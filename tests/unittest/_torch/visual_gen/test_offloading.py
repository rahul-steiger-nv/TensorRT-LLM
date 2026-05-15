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

import torch
import torch.nn as nn

from tensorrt_llm._torch.visual_gen.offloading import (
    ForwardHookOffloadPipeline,
    OffloadPipelinePart,
    SharedOffloadStorageConfig,
)


def _make_module(fill_value: float) -> nn.Module:
    module = nn.Sequential(
        nn.Linear(3, 4, bias=False),
        nn.LayerNorm(4),
    )
    with torch.no_grad():
        for param in module.parameters():
            param.fill_(fill_value)
    return module


def test_shared_offload_storage_reader_does_not_overwrite_writer(tmp_path):
    storage_path = tmp_path / "offload.bin"
    writer_module = _make_module(3.0)
    reader_module = _make_module(7.0)
    expected = {name: param.detach().clone() for name, param in writer_module.named_parameters()}

    writer = ForwardHookOffloadPipeline(
        stages=("part",),
        parts={"part": OffloadPipelinePart(writer_module)},
        device="cpu",
        pin_memory=False,
        shared_storage=SharedOffloadStorageConfig(
            path=str(storage_path),
            is_writer=True,
        ),
    )
    writer.initialize()

    reader = ForwardHookOffloadPipeline(
        stages=("part",),
        parts={"part": OffloadPipelinePart(reader_module)},
        device="cpu",
        pin_memory=False,
        shared_storage=SharedOffloadStorageConfig(
            path=str(storage_path),
            is_writer=False,
        ),
    )
    reader.initialize()
    reader.manager.stage("part")

    for name, param in reader_module.named_parameters():
        torch.testing.assert_close(param, expected[name])

    writer.cleanup()
    reader.cleanup()
