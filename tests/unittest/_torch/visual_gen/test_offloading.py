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
"""Tests for visual generation module parameter offloading."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from tensorrt_llm._torch.visual_gen.config import PipelineConfig, VisualGenArgs
from tensorrt_llm._torch.visual_gen.offloading import (
    ModuleOffloadManager,
    OffloadPipeline,
)
from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline


class _ToyModule(nn.Module):

    def __init__(self, weight_value: float, bias_value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((2, 2), weight_value))
        self.register_buffer("bias", torch.full((2,), bias_value))


class _AlignmentToyModule(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.prefix = nn.Parameter(torch.ones((1,), dtype=torch.bfloat16))
        self.alignment_sensitive = nn.Parameter(torch.ones((3,), dtype=torch.bfloat16))


class _ToyTransformer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.blocks = _ToyModule(weight_value=3.0, bias_value=30.0)

    @property
    def device(self):
        return next(self.parameters()).device


class _OffloadCudaGraphPipeline(BasePipeline):

    def _init_transformer(self) -> None:
        self.transformer = _ToyModule(weight_value=1.0, bias_value=10.0)

    def default_offload_stages(self) -> tuple[tuple[str, ...], ...]:
        return (("transformer",),)


class _CustomOffloadPipeline(BasePipeline):

    def _init_transformer(self) -> None:
        self.transformer = _ToyTransformer()
        self.text_encoder = _ToyModule(weight_value=1.0, bias_value=10.0)
        self.vae = _ToyModule(weight_value=2.0, bias_value=20.0)


def _make_config(
    *,
    enable_offloading: bool = True,
    offload_stages: list[str | list[str]] | None = None,
):
    return SimpleNamespace(
        pretrained_config=SimpleNamespace(),
        cuda_graph=SimpleNamespace(enable_cuda_graph=False),
        torch_compile=SimpleNamespace(enable_torch_compile=False),
        pipeline=PipelineConfig(
            enable_offloading=enable_offloading,
            offload_stages=offload_stages,
        ),
    )


def _make_manager() -> tuple[ModuleOffloadManager, _ToyModule, _ToyModule]:
    group_a = _ToyModule(weight_value=1.0, bias_value=10.0)
    group_b = _ToyModule(weight_value=2.0, bias_value=20.0)
    manager = ModuleOffloadManager(
        groups={
            "group_a": group_a,
            "group_b": group_b,
        },
        device="cpu",
        pin_memory=False,
    )
    manager.initialize()
    return manager, group_a, group_b


def _storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


def test_cuda_graphs_with_offload_raise_not_implemented():
    config = _make_config()
    config.cuda_graph.enable_cuda_graph = True

    with pytest.raises(
        NotImplementedError,
        match="CUDA graphs are not supported with visual generation offloading",
    ):
        _OffloadCudaGraphPipeline(config)


def test_offload_context_requires_initialized_pipeline_when_configured():
    pipeline = _OffloadCudaGraphPipeline(_make_config())

    with pipeline.offload_context("transformer", enable=False):
        pass

    with pytest.raises(
        RuntimeError,
        match="offload pipeline has not been initialized",
    ):
        with pipeline.offload_context("transformer"):
            pass


def test_configured_offload_stages_override_model_defaults_and_expose_vae():
    pipeline = _CustomOffloadPipeline(
        _make_config(
            offload_stages=[
                "text_encoder",
                ["transformer.blocks", "vae"],
                "transformer_2.blocks",
            ]
        )
    )

    assert pipeline.offload_stages() == (
        ("text_encoder",),
        ("transformer.blocks", "vae"),
        ("transformer_2.blocks",),
    )

    parts = pipeline.collect_offload_pipeline_parts()
    assert parts["text_encoder"] is pipeline.text_encoder
    assert parts["transformer.blocks"] is pipeline.transformer.blocks
    assert parts["vae"] is pipeline.vae

    assert pipeline._filter_available_offload_stages(pipeline.offload_stages(), parts) == (
        ("text_encoder",),
        ("transformer.blocks", "vae"),
    )


def test_offload_context_resolves_part_to_grouped_stage():
    pipeline = _CustomOffloadPipeline(
        _make_config(offload_stages=[["transformer.blocks", "vae"]])
    )
    pipeline.initialize_offload_pipeline()

    with pipeline.offload_context("vae"):
        assert pipeline._offload_pipeline is not None
        assert pipeline._offload_pipeline.manager.active_group_name == "transformer.blocks+vae"

    pipeline.cleanup()


def test_offload_context_uses_filtered_group_when_stage_parts_are_unavailable():
    pipeline = _CustomOffloadPipeline(
        _make_config(offload_stages=[["transformer.blocks", "transformer_2.blocks"]])
    )
    pipeline.initialize_offload_pipeline()

    with pipeline.offload_context("transformer.blocks"):
        assert pipeline._offload_pipeline is not None
        assert pipeline._offload_pipeline.manager.active_group_name == "transformer.blocks"

    pipeline.cleanup()


def test_visual_gen_args_loads_yaml_offload_stages(tmp_path):
    config_path = tmp_path / "visual_gen.yaml"
    config_path.write_text(
        """
pipeline:
  enable_offloading: true
  offload_stages:
    - text_encoder
    - [transformer.blocks, vae]
""",
        encoding="utf-8",
    )

    args = VisualGenArgs.from_yaml(config_path)

    assert args.pipeline.enable_offloading is True
    assert args.pipeline.offload_stages == [
        "text_encoder",
        ["transformer.blocks", "vae"],
    ]


def test_initialize_reports_cpu_storage_allocation_context():
    group = _ToyModule(weight_value=1.0, bias_value=10.0)
    manager = ModuleOffloadManager(
        groups={"group": group},
        device="cpu",
        pin_memory=False,
    )
    gpu_arena = torch.empty(1024, dtype=torch.uint8, device="cpu")

    def fail_cpu_storage(*args, **kwargs):
        raise RuntimeError("cpu allocation failed")

    with (
        patch.object(manager, "_allocate_gpu_arena", return_value=gpu_arena),
        patch("torch.empty", side_effect=fail_cpu_storage),
    ):
        with pytest.raises(
            RuntimeError,
            match="Failed to allocate packed CPU storage for visual generation offload",
        ) as exc_info:
            manager.initialize()

    message = str(exc_info.value)
    assert "Failed to allocate packed CPU storage for visual generation offload" in message
    assert "pin_memory=False" in message
    assert "groups=[group=" in message


def test_initialize_reports_cuda_arena_allocation_hint():
    group = _ToyModule(weight_value=1.0, bias_value=10.0)
    manager = ModuleOffloadManager(
        groups={"group": group},
        device="cuda",
        pin_memory=False,
    )
    original_empty = torch.empty

    def fail_cuda_arena(*args, **kwargs):
        device = torch.device(kwargs.get("device", "cpu"))
        if device.type == "cuda":
            raise RuntimeError("cuda allocation failed")
        return original_empty(*args, **kwargs)

    with patch("torch.empty", side_effect=fail_cuda_arena):
        with pytest.raises(
            RuntimeError,
            match="Failed to allocate GPU arena for visual generation offload",
        ) as exc_info:
            manager.initialize()

    message = str(exc_info.value)
    assert "Failed to allocate GPU arena for visual generation offload" in message
    assert "device=cuda" in message
    assert "groups=[group=" in message
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in message


def test_initialize_packs_group_before_rebinding_to_cpu():
    group_a = _ToyModule(weight_value=1.0, bias_value=10.0)
    group_b = _ToyModule(weight_value=2.0, bias_value=20.0)
    manager = ModuleOffloadManager(
        groups={
            "group_a": group_a,
            "group_b": group_b,
        },
        device="cpu",
        pin_memory=False,
    )
    events = []
    original_rebind_to_cpu = manager._rebind_to_cpu
    original_allocate_cpu_storage = manager._allocate_cpu_storage

    def record_rebind_to_cpu(name):
        events.append(("rebind_cpu", name))
        return original_rebind_to_cpu(name)

    def record_allocate_cpu_storage(num_bytes, group_name=None):
        events.append(("allocate_cpu", group_name))
        return original_allocate_cpu_storage(num_bytes, group_name=group_name)

    with (
        patch.object(manager, "_rebind_to_cpu", side_effect=record_rebind_to_cpu),
        patch.object(
            manager,
            "_allocate_cpu_storage",
            side_effect=record_allocate_cpu_storage,
        ),
    ):
        manager.initialize()

    assert events.index(("allocate_cpu", "group_a")) < events.index(("rebind_cpu", "group_a"))
    assert events.index(("allocate_cpu", "group_b")) < events.index(("rebind_cpu", "group_b"))


def test_packed_tensor_views_are_sufficiently_aligned():
    group = _AlignmentToyModule()
    manager = ModuleOffloadManager(
        groups={"group": group},
        device="cpu",
        pin_memory=False,
    )
    manager.initialize()

    layout = manager.layouts["group"]
    for spec in layout.specs:
        assert spec.offset % 16 == 0

    manager.stage("group")
    assert group.prefix.data_ptr() % 16 == 0
    assert group.alignment_sensitive.data_ptr() % 16 == 0


def test_offload_pipeline_context_stages_requested_group():
    group_a = _ToyModule(weight_value=1.0, bias_value=10.0)
    group_b = _ToyModule(weight_value=2.0, bias_value=20.0)
    pipeline = OffloadPipeline(
        stages=(("group_a",), ("group_b",)),
        parts={
            "group_a": group_a,
            "group_b": group_b,
        },
        device="cpu",
        pin_memory=False,
    )
    pipeline.initialize()

    assert pipeline.manager.active_group_name is None
    with pipeline.context("group_a"):
        assert pipeline.manager.active_group_name == "group_a"
    assert pipeline.manager.active_group_name == "group_a"

    with pipeline.context("group_b"):
        assert pipeline.manager.active_group_name == "group_b"


def test_inactive_group_stays_cpu_backed_after_staging_another_group():
    manager, group_a, group_b = _make_manager()
    assert manager.gpu_arena is not None
    group_a_cpu_storage = manager.layouts["group_a"].cpu_storage
    group_b_cpu_storage = manager.layouts["group_b"].cpu_storage
    assert group_a_cpu_storage is not None
    assert group_b_cpu_storage is not None
    group_a_cpu_storage_ptr = _storage_ptr(group_a_cpu_storage)
    group_b_cpu_storage_ptr = _storage_ptr(group_b_cpu_storage)
    gpu_arena_ptr = _storage_ptr(manager.gpu_arena)

    manager.stage("group_a")

    assert manager.active_group_name == "group_a"
    assert _storage_ptr(group_a.weight) == gpu_arena_ptr
    assert _storage_ptr(group_a.bias) == gpu_arena_ptr
    assert _storage_ptr(group_b.weight) == group_b_cpu_storage_ptr
    assert _storage_ptr(group_b.bias) == group_b_cpu_storage_ptr
    torch.testing.assert_close(group_b.weight, torch.full((2, 2), 2.0))
    torch.testing.assert_close(group_b.bias, torch.full((2,), 20.0))

    manager.stage("group_b")

    assert manager.active_group_name == "group_b"
    assert _storage_ptr(group_a.weight) == group_a_cpu_storage_ptr
    assert _storage_ptr(group_a.bias) == group_a_cpu_storage_ptr
    assert _storage_ptr(group_b.weight) == gpu_arena_ptr
    assert _storage_ptr(group_b.bias) == gpu_arena_ptr
    torch.testing.assert_close(group_a.weight, torch.full((2, 2), 1.0))
    torch.testing.assert_close(group_a.bias, torch.full((2,), 10.0))


def test_state_dict_reads_correct_inactive_group_data():
    manager, group_a, group_b = _make_manager()

    manager.stage("group_a")
    group_b_state = group_b.state_dict()
    torch.testing.assert_close(group_b_state["weight"], torch.full((2, 2), 2.0))
    torch.testing.assert_close(group_b_state["bias"], torch.full((2,), 20.0))

    manager.stage("group_b")
    group_a_state = group_a.state_dict()
    torch.testing.assert_close(group_a_state["weight"], torch.full((2, 2), 1.0))
    torch.testing.assert_close(group_a_state["bias"], torch.full((2,), 10.0))


def test_rebinding_reuses_cached_view_objects():
    manager, group_a, _ = _make_manager()
    cpu_weight = group_a.weight
    cpu_bias = group_a.bias

    manager.stage("group_a")
    gpu_weight = group_a.weight
    gpu_bias = group_a.bias

    assert gpu_weight is not cpu_weight
    assert gpu_bias is not cpu_bias

    manager.stage("group_b")
    assert group_a.weight is cpu_weight
    assert group_a.bias is cpu_bias

    manager.stage("group_a")
    assert group_a.weight is gpu_weight
    assert group_a.bias is gpu_bias
