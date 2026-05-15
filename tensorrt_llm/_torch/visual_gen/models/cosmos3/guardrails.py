# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import time
import warnings
from contextlib import contextmanager
from typing import Callable, Iterator, Protocol, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn

from tensorrt_llm._torch.visual_gen.offloading import OffloadPipelinePart
from tensorrt_llm.logger import logger


class TextGuardrailFn(Protocol):
    def __call__(self, prompt: str, profile_latency: bool = False) -> tuple[bool, str]:
        ...


class VideoGuardrailFn(Protocol):
    def __call__(self, frames: np.ndarray, profile_latency: bool = False) -> np.ndarray | None:
        ...


class TextGuardrail:
    def __init__(
        self,
        checkers: Sequence[Callable[[str, bool], tuple[bool, str]]],
        offload_parts: dict[str, OffloadPipelinePart],
    ) -> None:
        self.checkers = tuple(checkers)
        self._offload_parts = offload_parts

    def __call__(self, prompt: str, profile_latency: bool = False) -> tuple[bool, str]:
        for checker in self.checkers:
            is_safe, msg = checker(prompt, profile_latency)
            if not is_safe:
                return is_safe, msg
        return True, ""

    def offload_pipeline_parts(self) -> dict[str, OffloadPipelinePart]:
        return self._offload_parts


class VideoGuardrail:
    def __init__(
        self,
        safety_checker: Callable[[np.ndarray, bool], tuple[bool, str]] | None,
        face_blurrer: Callable[[np.ndarray, bool], np.ndarray] | None,
        offload_parts: dict[str, OffloadPipelinePart],
    ) -> None:
        self.safety_checker = safety_checker
        self.face_blurrer = face_blurrer
        self._offload_parts = offload_parts

    def __call__(self, frames: np.ndarray, profile_latency: bool = False) -> np.ndarray | None:
        if self.safety_checker is not None:
            is_safe, msg = self.safety_checker(frames, profile_latency)
            if not is_safe:
                logger.warning(f"Video content safety: {msg}")
                return None
        if self.face_blurrer is not None:
            frames = self.face_blurrer(frames, profile_latency)
        return frames

    def offload_pipeline_parts(self) -> dict[str, OffloadPipelinePart]:
        return self._offload_parts

GUARDRAIL_HF_REPO = "nvidia/Cosmos-Guardrail1"
GUARDRAIL_HF_REVISION = "d6d4bfa899a71454a700907664f3e88f503950cf"
CUTOFF_UNSAFE_FRAMES_PERCENT = 10


# ---------------------------------------------------------------------------
# Video safety classifier (matches reference: SigLIP so400m + 3-layer head)
# ---------------------------------------------------------------------------
class SafetyClassifier(nn.Module):
    """3-layer classifier with BatchNorm (1152 → 512 → 256 → 7)."""

    def __init__(self, input_size: int = 1152, num_classes: int = 7):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


CLASS_IDX_TO_NAME = {
    0: "Safe",
    1: "Sexual_Content",
    3: "Drugs",
    4: "Child_Abuse",
    5: "Hate_and_Harassment",
    6: "Self-Harm",
}


# ---------------------------------------------------------------------------
# Face pixelation utility
# ---------------------------------------------------------------------------
def _pixelate_face(face_img: np.ndarray, blocks: int = 5) -> np.ndarray:
    h, w = face_img.shape[:2]
    if h == 0 or w == 0:
        return face_img
    temp = cv2.resize(face_img, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


def _place_guardrail_module(
    module: nn.Module,
    device: torch.device,
    offload_to_cpu: bool,
    dtype: torch.dtype | None = None,
) -> nn.Module:
    target_device = torch.device("cpu") if offload_to_cpu else device
    if dtype is None:
        module = module.to(target_device)
    else:
        module = module.to(device=target_device, dtype=dtype)
    return module.eval()


def _synchronize_guardrail_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def _measure_guardrail_latency(
    name: str,
    device: torch.device,
    enabled: bool,
) -> Iterator[None]:
    if not enabled:
        yield
        return

    _synchronize_guardrail_device(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        _synchronize_guardrail_device(device)
        logger.info("Guardrail latency [%s]: %.3fs", name, time.perf_counter() - start)


# ---------------------------------------------------------------------------
# Default guardrail builders
# ---------------------------------------------------------------------------
def download_guardrail_checkpoint() -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            GUARDRAIL_HF_REPO,
            revision=GUARDRAIL_HF_REVISION,
            local_files_only=True,
        )
    except FileNotFoundError:
        logger.warning(
            f"Guardrail checkpoint not found, downloading from {GUARDRAIL_HF_REPO} {GUARDRAIL_HF_REVISION}"
        )
        return snapshot_download(
            GUARDRAIL_HF_REPO,
            revision=GUARDRAIL_HF_REVISION,
        )


def build_text_guardrail(
    guardrail_ckpt_dir: str,
    device: torch.device | str = "cuda",
    offload_parts: set[str] | None = None,
) -> TextGuardrailFn:
    device = torch.device(device)
    checkers: list[Callable[[str, bool], tuple[bool, str]]] = []
    parts: dict[str, OffloadPipelinePart] = {}
    offload_parts = offload_parts or set()

    # 1. Blocklist
    try:
        import nltk
        from better_profanity import profanity as profanity_filter

        blocklist_dir = os.path.join(guardrail_ckpt_dir, "blocklist")
        nltk.data.path.append(os.path.join(blocklist_dir, "nltk_data"))

        def _read_keywords(dirpath: str) -> list[str]:
            words: list[str] = []
            if not os.path.isdir(dirpath):
                return words
            for fname in sorted(os.listdir(dirpath)):
                fpath = os.path.join(dirpath, fname)
                if os.path.isfile(fpath):
                    with open(fpath, encoding="utf-8") as f:
                        words.extend(line.strip() for line in f if line.strip())
            return words

        blocklist_words = _read_keywords(os.path.join(blocklist_dir, "custom"))
        whitelist_words = _read_keywords(os.path.join(blocklist_dir, "whitelist"))
        profanity_filter.load_censor_words(
            custom_words=blocklist_words, whitelist_words=whitelist_words
        )

        def _blocklist_check(prompt: str, profile_latency: bool = False) -> tuple[bool, str]:
            with _measure_guardrail_latency("text blocklist guardrail", device, profile_latency):
                if profanity_filter.contains_profanity(prompt):
                    return False, "Blocked by keyword filter"
                return True, ""

        checkers.append(_blocklist_check)
        logger.info("Blocklist guardrail loaded (%d keywords)", len(blocklist_words))
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.warning("Could not load blocklist guardrail: %s", e)

    # 2. Qwen3Guard
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "Qwen/Qwen3Guard-Gen-0.6B"
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_id)
        qwen_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        qwen_offload = "guardrail.text.qwen" in offload_parts
        qwen_model = _place_guardrail_module(qwen_model, device, qwen_offload)
        parts["guardrail.text.qwen"] = OffloadPipelinePart(
            module=qwen_model,
            hook_modules=(qwen_model,),
        )

        def _qwen_check(prompt: str, profile_latency: bool = False) -> tuple[bool, str]:
            with _measure_guardrail_latency("text Qwen3Guard guardrail", device, profile_latency):
                conversations = [{"role": "user", "content": prompt}]
                input_ids = qwen_tokenizer.apply_chat_template(
                    conversations,
                    tokenize=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    return_dict=False,
                ).to(device)
                with torch.no_grad():
                    output_ids = qwen_model.generate(input_ids, max_new_tokens=128)
                response_ids = output_ids[0][input_ids.shape[1] :].detach().cpu()
                response = qwen_tokenizer.decode(response_ids, skip_special_tokens=True)
                if "unsafe" in response.lower():
                    return False, f"Qwen3Guard: {response.strip()}"
                return True, ""

        checkers.append(_qwen_check)
        logger.info("Qwen3Guard guardrail loaded")
    except (ImportError, OSError, RuntimeError, ValueError) as e:
        logger.warning("Could not load Qwen3Guard guardrail: %s", e)

    return TextGuardrail(checkers, parts)


def build_video_guardrail(
    guardrail_ckpt_dir: str,
    device: torch.device | str = "cuda",
    offload_parts: set[str] | None = None,
) -> VideoGuardrailFn:
    device = torch.device(device)
    safety_checker: Callable[[np.ndarray, bool], tuple[bool, str]] | None = None
    face_blurrer: Callable[[np.ndarray, bool], np.ndarray] | None = None
    parts: dict[str, OffloadPipelinePart] = {}
    offload_parts = offload_parts or set()

    # 1. Video content safety filter: SigLIP so400m + SafetyClassifier
    try:
        from PIL import Image
        from transformers import SiglipModel, SiglipProcessor

        siglip_id = "google/siglip-so400m-patch14-384"
        siglip_model = SiglipModel.from_pretrained(siglip_id)
        safety_offload = "guardrail.video.safety" in offload_parts
        siglip_model = _place_guardrail_module(
            siglip_model, device, safety_offload, dtype=torch.float32
        )
        siglip_processor = SiglipProcessor.from_pretrained(siglip_id)

        classifier = SafetyClassifier(input_size=1152, num_classes=7)
        ckpt_path = os.path.join(
            guardrail_ckpt_dir, "video_content_safety_filter", "safety_filter.pt"
        )
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = {k.removeprefix("network."): v for k, v in checkpoint["model"].items()}
        classifier.load_state_dict(state)
        classifier = _place_guardrail_module(classifier, device, safety_offload, dtype=torch.float32)
        parts["guardrail.video.safety"] = OffloadPipelinePart(
            module=nn.ModuleList([siglip_model, classifier]),
            hook_modules=(getattr(siglip_model, "vision_model", siglip_model), classifier),
        )

        def _safety_check(frames: np.ndarray, profile_latency: bool = False) -> tuple[bool, str]:
            with _measure_guardrail_latency("video content safety guardrail", device, profile_latency):
                unsafe_count = 0
                total = len(frames)
                for frame in frames:
                    img = Image.fromarray(frame)
                    inputs = siglip_processor(images=img, return_tensors="pt").to(
                        device, dtype=torch.float32
                    )
                    with torch.no_grad():
                        features = siglip_model.get_image_features(**inputs)
                        features = features / features.norm(dim=-1, keepdim=True)
                        logits = classifier(features)
                        pred = logits.argmax(dim=-1).item()
                    class_name = CLASS_IDX_TO_NAME.get(pred, "Unknown")
                    if class_name != "Safe":
                        unsafe_count += 1

                if unsafe_count / total > CUTOFF_UNSAFE_FRAMES_PERCENT / 100:
                    return False, f"Video content safety: {unsafe_count}/{total} frames unsafe"
                return True, ""

        safety_checker = _safety_check
        logger.info("Video content safety filter loaded (SigLIP so400m + classifier)")
    except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError) as e:
        logger.warning("Could not load video safety filter: %s", e)

    # 2. Face blur: RetinaFace + pixelation
    try:
        from retinaface.data import cfg_re50
        from retinaface.layers.functions.prior_box import PriorBox
        from retinaface.models.retinaface import RetinaFace
        from retinaface.utils.nms.py_cpu_nms import py_cpu_nms

        face_ckpt = os.path.join(guardrail_ckpt_dir, "face_blur_filter", "Resnet50_Final.pth")
        if not os.path.exists(face_ckpt):
            raise FileNotFoundError(face_ckpt)

        cfg = dict(cfg_re50)
        cfg["pretrain"] = False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            retinaface_net = RetinaFace(cfg=cfg, phase="test")

        # Load weights (strip 'module.' prefix if present)
        pretrained_dict = torch.load(face_ckpt, map_location="cpu", weights_only=True)
        if "state_dict" in pretrained_dict:
            pretrained_dict = pretrained_dict["state_dict"]
        pretrained_dict = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in pretrained_dict.items()
        }
        retinaface_net.load_state_dict(pretrained_dict, strict=False)
        face_offload = "guardrail.video.face_blur" in offload_parts
        retinaface_net = _place_guardrail_module(
            retinaface_net, device, face_offload, dtype=torch.float32
        )
        parts["guardrail.video.face_blur"] = OffloadPipelinePart(
            module=retinaface_net,
            hook_modules=(retinaface_net,),
        )

        CONF_THRESH = 0.7
        NMS_THRESH = 0.4
        TOP_K = 5000
        KEEP_TOP_K = 750

        def _decode_batch(loc, priors, variances):
            batch_size = loc.size(0)
            p = priors.unsqueeze(0).expand(batch_size, -1, -1)
            boxes = torch.cat(
                (
                    p[:, :, :2] + loc[:, :, :2] * variances[0] * p[:, :, 2:],
                    p[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1]),
                ),
                dim=2,
            )
            boxes[:, :, :2] -= boxes[:, :, 2:] / 2
            boxes[:, :, 2:] += boxes[:, :, :2]
            return boxes

        def _face_blur(frames: np.ndarray, profile_latency: bool = False) -> np.ndarray:
            with _measure_guardrail_latency("video face blur guardrail", device, profile_latency):
                prior_data = None
                scale = None
                result_frames = []

                for frame in frames:
                    frame_t = torch.from_numpy(frame).to(device, dtype=torch.float32)
                    frame_t = frame_t.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
                    frame_t = frame_t[:, [2, 1, 0], :, :]  # RGB to BGR
                    means = torch.tensor(
                        [104.0, 117.0, 123.0], device=device, dtype=torch.float32
                    ).view(1, 3, 1, 1)
                    frame_t = frame_t - means

                    h, w = frame_t.shape[2], frame_t.shape[3]
                    if prior_data is None:
                        priorbox = PriorBox(cfg, image_size=(h, w))
                        prior_data = priorbox.forward().to(device, dtype=torch.float32)
                    if scale is None:
                        scale = torch.tensor([w, h, w, h], device=device, dtype=torch.float32)

                    with torch.no_grad():
                        loc, conf, _ = retinaface_net(frame_t)

                    boxes = _decode_batch(loc, prior_data, cfg["variance"])
                    boxes = (boxes * scale).squeeze(0).cpu().numpy()
                    scores = conf.squeeze(0)[:, 1].cpu().numpy()

                    # Filter by confidence
                    inds = np.where(scores > CONF_THRESH)[0]
                    boxes_f = boxes[inds]
                    scores_f = scores[inds]
                    order = scores_f.argsort()[::-1][:TOP_K]
                    boxes_f = boxes_f[order]
                    scores_f = scores_f[order]

                    # NMS
                    dets = np.hstack((boxes_f, scores_f[:, np.newaxis])).astype(np.float32)
                    keep = py_cpu_nms(dets, NMS_THRESH)
                    dets = dets[keep][:KEEP_TOP_K]

                    out_frame = frame.copy()
                    for det in dets:
                        x1, y1, x2, y2 = map(int, det[:4])
                        if x2 - x1 < 20 or y2 - y1 < 20:
                            continue
                        max_h, max_w = out_frame.shape[:2]
                        y1c, y2c = max(y1, 0), min(y2, max_h)
                        x1c, x2c = max(x1, 0), min(x2, max_w)
                        out_frame[y1c:y2c, x1c:x2c] = _pixelate_face(
                            out_frame[y1c:y2c, x1c:x2c]
                        )

                    result_frames.append(out_frame)

                return np.array(result_frames)

        face_blurrer = _face_blur
        logger.info("Face blur filter loaded (RetinaFace Resnet50)")
    except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError) as e:
        logger.warning("Could not load face blur filter: %s", e)

    return VideoGuardrail(safety_checker, face_blurrer, parts)


def check_video_safety(
    video_tensor: torch.Tensor,
    video_guardrail: VideoGuardrailFn,
    profile_latency: bool = False,
) -> torch.Tensor | None:
    v = video_tensor.detach().cpu()
    was_batched = v.dim() == 5
    if was_batched:
        v = v[0]
    frames_np = v.numpy()
    frames_np = video_guardrail(frames_np, profile_latency=profile_latency)
    if frames_np is None:
        return None

    result = torch.from_numpy(frames_np)
    if was_batched:
        result = result.unsqueeze(0)
    return result.to(video_tensor.device)
