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

from tensorrt_llm._torch.visual_gen.pipeline import BasePipeline
from tensorrt_llm._torch.visual_gen.offloading import (
    ModuleOffloadManager,
    OffloadPipeline,
    OffloadPipelinePart,
)


class _ToyModule(nn.Module):

    def __init__(self, weight_value: float, bias_value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((2, 2), weight_value))
        self.register_buffer("bias", torch.full((2,), bias_value))


class _OffloadCudaGraphPipeline(BasePipeline):

    def _init_transformer(self) -> None:
        self.transformer = _ToyModule(weight_value=1.0, bias_value=10.0)

    def default_offload_stages(self) -> tuple[tuple[str, ...], ...]:
        return (("transformer",),)


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
    config = SimpleNamespace(
        pretrained_config=SimpleNamespace(),
        cuda_graph=SimpleNamespace(enable_cuda_graph=True),
        torch_compile=SimpleNamespace(enable_torch_compile=False),
    )

    with pytest.raises(
        NotImplementedError,
        match="CUDA graphs are not supported with visual generation offloading",
    ):
        _OffloadCudaGraphPipeline(config)


def test_initialize_reports_cpu_storage_allocation_context():
    group = _ToyModule(weight_value=1.0, bias_value=10.0)
    manager = ModuleOffloadManager(
        groups={"group": group},
        device="cpu",
        pin_memory=False,
    )

    def fail_cpu_storage(*args, **kwargs):
        raise RuntimeError("cpu allocation failed")

    with patch("torch.empty", side_effect=fail_cpu_storage):
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


def test_copy_to_cpu_storage_reports_tensor_context():
    group = _ToyModule(weight_value=1.0, bias_value=10.0)
    manager = ModuleOffloadManager(
        groups={"group": group},
        device="cpu",
        pin_memory=False,
    )
    layout = manager._collect_group_layout("group", group, 0)
    manager.cpu_storage = torch.empty(0, dtype=torch.uint8, device="cpu")

    with pytest.raises(
        RuntimeError,
        match="Failed to copy offload tensor 'group.weight'",
    ) as exc_info:
        manager._copy_group_to_cpu_storage(layout)

    message = str(exc_info.value)
    assert "Failed to copy offload tensor 'group.weight'" in message
    assert "to packed CPU storage at offset 0" in message
    assert "dtype=torch.float32" in message


def test_offload_pipeline_context_stages_requested_group():
    group_a = _ToyModule(weight_value=1.0, bias_value=10.0)
    group_b = _ToyModule(weight_value=2.0, bias_value=20.0)
    pipeline = OffloadPipeline(
        stages=(("group_a",), ("group_b",)),
        parts={
            "group_a": OffloadPipelinePart(group_a),
            "group_b": OffloadPipelinePart(group_b),
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
    assert manager.cpu_storage is not None
    assert manager.gpu_arena is not None
    cpu_storage_ptr = _storage_ptr(manager.cpu_storage)
    gpu_arena_ptr = _storage_ptr(manager.gpu_arena)

    manager.stage("group_a")

    assert manager.active_group_name == "group_a"
    assert _storage_ptr(group_a.weight) == gpu_arena_ptr
    assert _storage_ptr(group_a.bias) == gpu_arena_ptr
    assert _storage_ptr(group_b.weight) == cpu_storage_ptr
    assert _storage_ptr(group_b.bias) == cpu_storage_ptr
    torch.testing.assert_close(group_b.weight, torch.full((2, 2), 2.0))
    torch.testing.assert_close(group_b.bias, torch.full((2,), 20.0))

    manager.stage("group_b")

    assert manager.active_group_name == "group_b"
    assert _storage_ptr(group_a.weight) == cpu_storage_ptr
    assert _storage_ptr(group_a.bias) == cpu_storage_ptr
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
