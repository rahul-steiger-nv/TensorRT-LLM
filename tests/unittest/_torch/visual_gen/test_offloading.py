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

import torch
import torch.nn as nn

from tensorrt_llm._torch.visual_gen.offloading import _OffloadGroup, ModuleOffloadManager


class _ToyModule(nn.Module):

    def __init__(self, weight_value: float, bias_value: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((2, 2), weight_value))
        self.register_buffer("bias", torch.full((2,), bias_value))


def _make_manager() -> tuple[ModuleOffloadManager, _ToyModule, _ToyModule]:
    group_a = _ToyModule(weight_value=1.0, bias_value=10.0)
    group_b = _ToyModule(weight_value=2.0, bias_value=20.0)
    manager = ModuleOffloadManager(
        groups=[
            _OffloadGroup("group_a", group_a),
            _OffloadGroup("group_b", group_b),
        ],
        device="cpu",
        pin_memory=False,
    )
    manager.initialize()
    return manager, group_a, group_b


def _storage_ptr(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


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
