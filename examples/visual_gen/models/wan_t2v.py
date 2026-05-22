#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Wan Text-to-Video generation.

Usage:
    python wan_t2v.py
    python wan_t2v.py --enable_offloading --offload_stages text_encoder,transformer.blocks,transformer_2.blocks,vae
"""

import argparse

from tensorrt_llm import VisualGen, VisualGenArgs
from tensorrt_llm.serve.media_storage import MediaStorage


def _parse_offload_stages(offload_stages: str | None) -> list[str] | None:
    if offload_stages is None:
        return None
    stages = [stage.strip() for stage in offload_stages.split(",") if stage.strip()]
    if not stages:
        raise ValueError("--offload_stages must contain at least one stage name")
    return stages


def main():
    parser = argparse.ArgumentParser(description="Wan Text-to-Video example")
    parser.add_argument(
        "--model",
        type=str,
        default="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        help="Model path or HuggingFace Hub ID",
    )
    parser.add_argument(
        "--enable_offloading",
        action="store_true",
        help="Enable Wan T2V text-encoder/transformer-block CPU offloading.",
    )
    parser.add_argument(
        "--offload_stages",
        type=str,
        default=None,
        help=(
            "Comma-separated offload stage names, e.g. "
            "text_encoder,transformer.blocks,transformer_2.blocks,vae. "
            "Implies --enable_offloading."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="wan_t2v_output.avi",
        help="Path to save the output video",
    )
    args = parser.parse_args()

    kwargs = {}
    offload_stages = _parse_offload_stages(args.offload_stages)
    if args.enable_offloading or offload_stages is not None:
        pipeline_kwargs = {
            "enable_offloading": True,
            "offload_device": "cpu",
        }
        if offload_stages is not None:
            pipeline_kwargs["offload_stages"] = offload_stages
        kwargs["pipeline"] = pipeline_kwargs

    visual_gen_args = VisualGenArgs(**kwargs) if kwargs else None
    visual_gen = VisualGen(model=args.model, args=visual_gen_args)

    # --- Model-specific: T2V request construction ---
    # Query per-model defaults (resolution, steps, guidance, seed, etc.).
    params = visual_gen.default_params

    output = visual_gen.generate(
        inputs="A cat playing piano in a sunny room",
        params=params,
    )

    # --- Model-specific: video output ---
    MediaStorage.save_video(output.video, args.output_path, frame_rate=params.frame_rate)
    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
