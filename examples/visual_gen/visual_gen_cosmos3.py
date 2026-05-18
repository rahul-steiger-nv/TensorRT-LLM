#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3 text-to-video and image-to-video generation with TensorRT-LLM VisualGen."""

import argparse
import time

import torch

from tensorrt_llm import VisualGen, VisualGenArgs, VisualGenParams, logger
from tensorrt_llm.serve.media_storage import MediaStorage

logger.set_level("info")


def parse_args():
    parser = argparse.ArgumentParser(
        description="TRTLLM VisualGen - Cosmos3 Text-to-Video/Image-to-Video Inference Example"
    )

    # Model & input
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Local path or HuggingFace Hub model ID for a Cosmos3 diffusers checkpoint.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="HuggingFace Hub revision (branch, tag, or commit SHA).",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for generation.")
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt. Default is model-specific.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Optional input image path for image-to-video conditioning.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="cosmos3_output.mp4",
        help="Path to save the output video, GIF, or first frame PNG.",
    )

    # Generation params
    parser.add_argument(
        "--height", type=int, default=None, help="Video height (default: model default)."
    )
    parser.add_argument(
        "--width", type=int, default=None, help="Video width (default: model default)."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=None,
        help="Number of frames to generate (default: model default).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of denoising steps (default: model default).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale (default: model default).",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=None,
        help="Maximum prompt token length (default: model default).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Frame rate for output video (default: model default).",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=1,
        help="Number of generation requests to run.",
    )
    parser.add_argument(
        "--disable_duration_template",
        action="store_true",
        help="Disable duration metadata template in the prompt.",
    )
    parser.add_argument(
        "--disable_resolution_template",
        action="store_true",
        help="Disable resolution metadata template in the prompt.",
    )
    parser.add_argument(
        "--use_system_prompt",
        action="store_true",
        help="Use the Cosmos3 system prompt in the chat template.",
    )

    # Quantization
    parser.add_argument(
        "--linear_type",
        type=str,
        default="default",
        choices=["default", "trtllm-fp8-per-tensor", "trtllm-fp8-blockwise", "trtllm-nvfp4"],
        help=(
            "Dynamic quantization shortcut. ModelOpt quantized checkpoints are preferred "
            "for Cosmos3 accuracy."
        ),
    )

    # Attention backend and parallelism
    parser.add_argument(
        "--attention_backend",
        type=str,
        default="VANILLA",
        choices=["VANILLA", "TRTLLM", "FA4"],
        help="Attention backend.",
    )
    parser.add_argument(
        "--cfg_size",
        type=int,
        default=1,
        choices=[1, 2],
        help="CFG parallel size. Distributes conditional/unconditional prompts across GPUs.",
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help=(
            "Ulysses sequence parallel size. Cosmos3 has 8 KV heads, so this must divide 8."
        ),
    )
    parser.add_argument("--disable_parallel_vae", action="store_true", help="Disable parallel VAE.")

    # CUDA graph / torch.compile
    parser.add_argument(
        "--enable_cudagraph", action="store_true", help="Enable CUDA graph acceleration."
    )
    parser.add_argument(
        "--disable_torch_compile", action="store_true", help="Disable torch.compile."
    )
    parser.add_argument(
        "--enable_fullgraph", action="store_true", help="Enable fullgraph for torch.compile."
    )
    parser.add_argument(
        "--disable_autotune", action="store_true", help="Disable autotuning during warmup."
    )

    # Offloading
    parser.add_argument(
        "--enable_offloading",
        action="store_true",
        help="Enable Cosmos3 language-model/gen-layer CPU offloading.",
    )
    parser.add_argument(
        "--offload_guardrails",
        "--offload-guardrails",
        action="store_true",
        help="Keep Cosmos3 guardrail weights on CPU and stage them to GPU only when needed.",
    )
    parser.add_argument(
        "--offload_share_memory",
        "--offload-share-memory",
        action="store_true",
        help=(
            "Share Cosmos3 CPU offload weights between distributed processes. "
            "Requires --enable_offloading and does not support guardrail offload yet."
        ),
    )
    parser.add_argument(
        "--offload_shared_memory_path",
        "--offload-shared-memory-path",
        type=str,
        default="",
        help="Optional backing file path for --offload_share_memory.",
    )
    parser.add_argument(
        "--offload_shared_memory_scope",
        "--offload-shared-memory-scope",
        type=str,
        default="global",
        choices=("global", "numa"),
        help=(
            "Shared offload storage scope: 'global' uses one copy for all ranks, "
            "'numa' uses one copy per NUMA node."
        ),
    )

    # Guardrails / profiling
    parser.add_argument(
        "--guardrail_checkpoint_dir",
        type=str,
        default="",
        help="Optional local Cosmos3 guardrail checkpoint directory.",
    )
    parser.add_argument(
        "--disable_guardrails",
        action="store_true",
        help="NOT RECOMMENDED: disable text/video guardrails.",
    )
    parser.add_argument(
        "--profile_memory",
        nargs="?",
        const=2,
        default=0,
        type=int,
        choices=(0, 1, 2),
        help=(
            "Memory profiling level: 0 disables profiling, 1 logs guardrail latency, "
            "2 also logs detailed Cosmos3 forward memory stats."
        ),
    )
    parser.add_argument(
        "--profile_generation_loop",
        action="store_true",
        help=(
            "Start and stop CUDA profiler collection around generation. "
            "Use with nsys --capture-range=cudaProfilerApi."
        ),
    )
    parser.add_argument(
        "--enable_layerwise_nvtx_marker",
        action="store_true",
        help="Enable layerwise NVTX markers.",
    )

    return parser.parse_args()


def _linear_type_to_quant_config(linear_type: str):
    common = {"ignore": ["language_model.*", "vae2llm", "llm2vae", "time_embedder.*"]}
    mapping = {
        "trtllm-fp8-per-tensor": {"quant_algo": "FP8", "dynamic": True, **common},
        "trtllm-fp8-blockwise": {"quant_algo": "FP8_BLOCK_SCALES", "dynamic": True, **common},
        "trtllm-nvfp4": {"quant_algo": "NVFP4", "dynamic": True, **common},
    }
    return mapping.get(linear_type)


def _build_visual_gen_args(args) -> VisualGenArgs:
    kwargs = dict(
        revision=args.revision,
        guardrail_checkpoint_dir=args.guardrail_checkpoint_dir,
        attention={"backend": args.attention_backend},
        parallel={
            "dit_cfg_size": args.cfg_size,
            "dit_ulysses_size": args.ulysses_size,
            "enable_parallel_vae": not args.disable_parallel_vae,
        },
        torch_compile={
            "enable_torch_compile": not args.disable_torch_compile,
            "enable_fullgraph": args.enable_fullgraph,
            "enable_autotune": not args.disable_autotune,
        },
        cuda_graph={"enable_cuda_graph": args.enable_cudagraph},
        pipeline={
            "enable_layerwise_nvtx_marker": args.enable_layerwise_nvtx_marker,
            "enable_offloading": args.enable_offloading,
            "offload_device": "cpu",
            "offload_shared_memory": args.offload_share_memory,
            "offload_shared_memory_path": args.offload_shared_memory_path,
            "offload_shared_memory_scope": args.offload_shared_memory_scope,
            "offload_guardrails": args.offload_guardrails,
        },
    )

    quant_config = _linear_type_to_quant_config(args.linear_type)
    if quant_config is not None:
        logger.info(f"Using {args.linear_type} dynamic quantization")
        kwargs["quant_config"] = quant_config

    return VisualGenArgs(**kwargs)


def main():
    args = parse_args()
    n_workers = args.cfg_size * args.ulysses_size

    if args.ulysses_size > 1:
        num_kv_heads = 8
        logger.info(
            f"Using Ulysses sequence parallelism: "
            f"{num_kv_heads} KV heads / {args.ulysses_size} ranks = "
            f"{num_kv_heads // args.ulysses_size} KV heads per GPU"
        )

    diffusion_args = _build_visual_gen_args(args)

    logger.info(
        f"Initializing VisualGen: world_size={n_workers}, "
        f"cfg_size={diffusion_args.parallel.dit_cfg_size}, "
        f"ulysses_size={diffusion_args.parallel.dit_ulysses_size}"
    )
    visual_gen = VisualGen(model=args.model_path, args=diffusion_args)

    try:
        defaults = visual_gen.default_params
        frame_rate = args.fps if args.fps is not None else defaults.frame_rate
        height = args.height if args.height is not None else defaults.height
        width = args.width if args.width is not None else defaults.width
        num_frames = args.num_frames if args.num_frames is not None else defaults.num_frames
        steps = args.steps if args.steps is not None else defaults.num_inference_steps
        guidance_scale = (
            args.guidance_scale if args.guidance_scale is not None else defaults.guidance_scale
        )

        logger.info(f"Generating video for prompt: '{args.prompt}'")
        logger.info(f"Resolution: {height}x{width}, Frames: {num_frames}, Steps: {steps}")
        logger.info(f"Guidance: {guidance_scale}, Frame rate: {frame_rate}")

        extra_params = {
            "use_duration_template": not args.disable_duration_template,
            "use_resolution_template": not args.disable_resolution_template,
            "use_system_prompt": args.use_system_prompt,
            "use_guardrails": not args.disable_guardrails,
            "profile_memory": args.profile_memory,
        }

        start_time = time.time()
        output = None

        if args.profile_generation_loop:
            torch.cuda.cudart().cudaProfilerStart()

        try:
            for generation_idx in range(args.num_generations):
                torch.cuda.nvtx.range_push(f"cosmos3_generation_{generation_idx}")
                try:
                    output = visual_gen.generate(
                        inputs=args.prompt,
                        params=VisualGenParams(
                            height=args.height,
                            width=args.width,
                            num_inference_steps=args.steps,
                            guidance_scale=args.guidance_scale,
                            max_sequence_length=args.max_sequence_length,
                            seed=args.seed,
                            num_frames=args.num_frames,
                            frame_rate=args.fps,
                            negative_prompt=args.negative_prompt,
                            image=args.image_path,
                            extra_params=extra_params,
                        ),
                    )
                finally:
                    torch.cuda.nvtx.range_pop()
        finally:
            if args.profile_generation_loop:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()

        time_taken = time.time() - start_time
        logger.info(f"Generation completed in {time_taken:.2f}s")
        logger.info(f"Average time per generation: {time_taken / args.num_generations:.2f}s")

        if output is None or output.video is None:
            logger.warning("No video was generated.")
            return

        if args.output_path.endswith(".png"):
            frame = output.video[0]
            MediaStorage.save_image(frame, args.output_path)
        else:
            MediaStorage.save_video(output.video, args.output_path, frame_rate=frame_rate)

    finally:
        visual_gen.shutdown()


if __name__ == "__main__":
    main()
