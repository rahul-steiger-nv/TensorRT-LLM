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

import tensorrt_llm._torch.visual_gen.offloading as offloading
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


class _FakeCudaRuntime:
    def __init__(self) -> None:
        self.registered: list[tuple[int, int, int]] = []
        self.unregistered: list[int] = []

    def cudaHostRegister(self, ptr, nbytes, flags) -> int:
        self.registered.append((ptr, nbytes, flags))
        return 0

    def cudaHostUnregister(self, ptr) -> int:
        self.unregistered.append(ptr)
        return 0


def test_shared_offload_storage_cuda_host_registers_file_mapping(tmp_path, monkeypatch):
    storage_path = tmp_path / "offload.bin"
    with open(storage_path, "wb") as storage_file:
        storage_file.truncate(4096)

    fake_cudart = _FakeCudaRuntime()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(offloading, "_ensure_cuda_context", lambda device: None)
    monkeypatch.setattr(offloading, "_load_cuda_runtime", lambda: fake_cudart)

    pipeline = ForwardHookOffloadPipeline(
        stages=("part",),
        parts={"part": OffloadPipelinePart(_make_module(3.0))},
        device="cuda",
        pin_memory=True,
        shared_storage=SharedOffloadStorageConfig(
            path=str(storage_path),
            is_writer=False,
        ),
    )
    pipeline.manager.cpu_storage = torch.from_file(
        str(storage_path),
        shared=True,
        size=4096,
        dtype=torch.uint8,
    )

    pipeline.manager._register_shared_cpu_storage()

    assert pipeline.manager._cpu_storage_cuda_registered
    assert len(fake_cudart.registered) == 1
    registered_ptr, registered_nbytes, registered_flags = fake_cudart.registered[0]
    assert registered_ptr <= pipeline.manager.cpu_storage.data_ptr()
    assert registered_nbytes >= pipeline.manager.cpu_storage.numel()
    assert registered_flags == offloading._CUDA_HOST_REGISTER_PORTABLE
    assert pipeline.manager._cuda_runtime is fake_cudart

    pipeline.cleanup()

    assert not pipeline.manager._cpu_storage_cuda_registered
    assert fake_cudart.unregistered == [registered_ptr]
