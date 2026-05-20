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
"""Module parameter offloading utilities for visual generation pipelines.

The offload path keeps model loading and quantization unchanged: weights are
loaded into the modules first, then selected module groups are copied into
packed CPU storage. At runtime one group at a time is staged into a reusable GPU
arena and the original module parameters/buffers are rebound to views of that
storage.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from tensorrt_llm.logger import logger


def _align_offset(offset: int, alignment: int = 256) -> int:
    return ((offset + alignment - 1) // alignment) * alignment


def _format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


# FlashInfer and other custom kernels can require tensor data pointers to be at
# least 16-byte aligned even for smaller dtypes such as BF16.
_PACKED_TENSOR_ALIGNMENT = 16


OffloadPipelineStage = tuple[str, ...]


@dataclass
class _FlatTensorSpec:
    owner: nn.Module
    name: str
    qualified_name: str
    is_parameter: bool
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    requires_grad: bool
    persistent: bool
    cpu_offset: int
    gpu_offset: int
    nbytes: int


@dataclass
class _GroupLayout:
    """Packed storage layout and rebound views for one offload group."""

    name: str
    module: nn.Module
    cpu_offset: int
    nbytes: int
    specs: list[_FlatTensorSpec]
    cpu_views: tuple[nn.Parameter | torch.Tensor, ...] = ()
    gpu_views: tuple[nn.Parameter | torch.Tensor, ...] = ()


class ModuleOffloadManager:
    """Pack module groups into CPU storage and stage one group on GPU.

    The manager owns two flat byte buffers:
    - ``cpu_storage`` contains all offloaded groups.
    - ``gpu_arena`` is reused for whichever group is currently active.

    Initializing the manager temporarily keeps both the original CPU tensors and
    packed CPU storage alive. This preserves compatibility with quantization
    flows that may replace or repack parameters during loading.
    """

    def __init__(
        self,
        groups: Mapping[str, nn.Module],
        device: torch.device | str,
        pin_memory: bool = True,
    ) -> None:
        if not groups:
            raise ValueError("At least one offload group must be provided")

        self.groups = dict(groups)
        self.device = torch.device(device)
        self.pin_memory = pin_memory
        self.cpu_storage: torch.Tensor | None = None
        self.gpu_arena: torch.Tensor | None = None
        self.layouts: dict[str, _GroupLayout] = {}
        self.active_group_name: str | None = None

        for name, module in self.groups.items():
            if not name:
                raise ValueError("Offload group names must be non-empty")
            if not isinstance(module, nn.Module):
                raise TypeError(f"Offload group '{name}' must contain an nn.Module")

    @staticmethod
    def _owner_and_name(root: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
        if "." not in qualified_name:
            return root, qualified_name
        module_path, name = qualified_name.rsplit(".", 1)
        return root.get_submodule(module_path), name

    @staticmethod
    def _tensor_nbytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    @staticmethod
    def _storage_key(tensor: torch.Tensor) -> tuple[int, int] | None:
        if tensor.numel() == 0:
            return None
        storage_offset_bytes = tensor.storage_offset() * tensor.element_size()
        return tensor.untyped_storage().data_ptr(), storage_offset_bytes

    def _get_alias_spec(
        self,
        seen_tensors: dict[tuple[int, int], _FlatTensorSpec],
        tensor: torch.Tensor,
        display_name: str,
    ) -> _FlatTensorSpec | None:
        key = self._storage_key(tensor)
        if key is None:
            return None
        canonical = seen_tensors.get(key)
        if canonical is None:
            return None
        if self._tensor_nbytes(tensor) != canonical.nbytes or tensor.dtype != canonical.dtype:
            raise ValueError(
                "Shared parameters or buffers with different sizes or dtypes are "
                f"not supported by ModuleOffloadManager: '{display_name}' aliases "
                f"'{canonical.qualified_name}'"
            )
        return canonical

    def _build_spec(
        self,
        group_name: str,
        group_module: nn.Module,
        qualified_name: str,
        tensor: torch.Tensor,
        is_parameter: bool,
        cpu_offset: int,
        gpu_offset: int,
        absolute_cpu_offset: int | None = None,
    ) -> _FlatTensorSpec:
        display_name = f"{group_name}.{qualified_name}"
        if not tensor.is_contiguous():
            raise ValueError(
                f"Cannot offload non-contiguous tensor '{display_name}' "
                f"with stride {tuple(tensor.stride())}"
            )

        owner, name = self._owner_and_name(group_module, qualified_name)
        return _FlatTensorSpec(
            owner=owner,
            name=name,
            qualified_name=display_name,
            is_parameter=is_parameter,
            shape=tuple(tensor.shape),
            stride=tuple(tensor.stride()),
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad if is_parameter else False,
            persistent=True if is_parameter else name not in owner._non_persistent_buffers_set,
            cpu_offset=absolute_cpu_offset if absolute_cpu_offset is not None else cpu_offset + gpu_offset,
            gpu_offset=gpu_offset,
            nbytes=self._tensor_nbytes(tensor),
        )

    def _group_tensors(
        self, group_module: nn.Module
    ) -> list[tuple[str, torch.Tensor, bool]]:
        tensors: list[tuple[str, torch.Tensor, bool]] = []
        tensors.extend(
            (qualified_name, param.detach(), True)
            for qualified_name, param in group_module.named_parameters(
                recurse=True,
                remove_duplicate=False,
            )
        )
        tensors.extend(
            (qualified_name, buffer.detach(), False)
            for qualified_name, buffer in group_module.named_buffers(
                recurse=True,
                remove_duplicate=False,
            )
        )
        return tensors

    def _append_layout_spec(
        self,
        group_name: str,
        group_module: nn.Module,
        qualified_name: str,
        tensor: torch.Tensor,
        is_parameter: bool,
        cpu_offset: int,
        gpu_offset: int,
        seen_tensors: dict[tuple[int, int], _FlatTensorSpec],
        specs: list[_FlatTensorSpec],
    ) -> int:
        """Append a tensor spec and return the next GPU-local byte offset.

        This handles three layout concerns in one place: alias reuse, packed
        tensor alignment, and spec construction. CPU offsets are absolute within
        the full packed storage, while GPU offsets are relative to the reusable
        arena.
        """
        display_name = f"{group_name}.{qualified_name}"
        alias = self._get_alias_spec(seen_tensors, tensor, display_name)
        if alias is None:
            gpu_offset = _align_offset(gpu_offset, _PACKED_TENSOR_ALIGNMENT)

        spec = self._build_spec(
            group_name=group_name,
            group_module=group_module,
            qualified_name=qualified_name,
            tensor=tensor,
            is_parameter=is_parameter,
            cpu_offset=cpu_offset,
            gpu_offset=alias.gpu_offset if alias is not None else gpu_offset,
            absolute_cpu_offset=alias.cpu_offset if alias is not None else None,
        )
        specs.append(spec)

        if alias is not None:
            return gpu_offset

        key = self._storage_key(tensor)
        if key is not None:
            seen_tensors[key] = spec
        return gpu_offset + spec.nbytes

    def _collect_group_layout(
        self, group_name: str, group_module: nn.Module, cpu_offset: int
    ) -> _GroupLayout:
        """Build the packed storage layout for one named module group."""
        gpu_offset = 0
        specs: list[_FlatTensorSpec] = []
        seen_tensors: dict[tuple[int, int], _FlatTensorSpec] = {}

        for qualified_name, tensor, is_parameter in self._group_tensors(group_module):
            gpu_offset = self._append_layout_spec(
                group_name=group_name,
                group_module=group_module,
                qualified_name=qualified_name,
                tensor=tensor,
                is_parameter=is_parameter,
                cpu_offset=cpu_offset,
                gpu_offset=gpu_offset,
                seen_tensors=seen_tensors,
                specs=specs,
            )

        if not specs:
            raise ValueError(f"Offload group '{group_name}' has no parameters or buffers")

        return _GroupLayout(
            name=group_name,
            module=group_module,
            cpu_offset=cpu_offset,
            nbytes=_align_offset(gpu_offset),
            specs=specs,
        )

    def _copy_group_to_cpu_storage(self, layout: _GroupLayout) -> None:
        assert self.cpu_storage is not None
        for spec in layout.specs:
            if spec.nbytes == 0:
                continue
            try:
                tensor = getattr(spec.owner, spec.name).detach()
                tensor_bytes = tensor.reshape(-1).view(torch.uint8).cpu()
                self.cpu_storage.narrow(0, spec.cpu_offset, spec.nbytes).copy_(tensor_bytes)
            except RuntimeError as e:
                raise RuntimeError(
                    f"Failed to copy offload tensor '{spec.qualified_name}' "
                    f"({_format_bytes(spec.nbytes)}, shape={spec.shape}, dtype={spec.dtype}) "
                    f"to packed CPU storage at offset {spec.cpu_offset}."
                ) from e

    def _group_size_summary(self) -> str:
        return ", ".join(
            f"{name}={_format_bytes(layout.nbytes)}" for name, layout in self.layouts.items()
        )

    def _cuda_allocation_hint(self) -> str:
        if self.device.type != "cuda":
            return ""
        return (
            " If this is due to CUDA memory fragmentation, try setting "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before starting the process."
        )

    def _allocate_cpu_storage(self, num_bytes: int) -> torch.Tensor:
        try:
            return torch.empty(
                num_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=self.pin_memory,
            )
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to allocate packed CPU storage for visual generation offload "
                f"({_format_bytes(num_bytes)}, {num_bytes} bytes, "
                f"pin_memory={self.pin_memory}, groups=[{self._group_size_summary()}])."
            ) from e

    def _allocate_gpu_arena(self, num_bytes: int) -> torch.Tensor:
        try:
            return torch.empty(num_bytes, dtype=torch.uint8, device=self.device)
        except RuntimeError as e:
            raise RuntimeError(
                "Failed to allocate GPU arena for visual generation offload "
                f"({_format_bytes(num_bytes)}, {num_bytes} bytes, "
                f"device={self.device}, groups=[{self._group_size_summary()}])."
                f"{self._cuda_allocation_hint()}"
            ) from e

    def _typed_view(self, storage: torch.Tensor, offset: int, spec: _FlatTensorSpec) -> torch.Tensor:
        byte_view = storage.narrow(0, offset, spec.nbytes)
        typed_view = byte_view.view(spec.dtype)
        return typed_view.as_strided(spec.shape, spec.stride)

    def _make_views(
        self,
        layout: _GroupLayout,
        storage: torch.Tensor,
        use_gpu_offsets: bool,
    ) -> tuple[nn.Parameter | torch.Tensor, ...]:
        views: list[nn.Parameter | torch.Tensor] = []
        for spec in layout.specs:
            offset = spec.gpu_offset if use_gpu_offsets else spec.cpu_offset
            view = self._typed_view(storage, offset, spec)
            if spec.is_parameter:
                views.append(nn.Parameter(view, requires_grad=spec.requires_grad))
            else:
                views.append(view)
        return tuple(views)

    def _bind_views(
        self,
        layout: _GroupLayout,
        views: tuple[nn.Parameter | torch.Tensor, ...],
    ) -> None:
        for spec, view in zip(layout.specs, views, strict=True):
            if spec.is_parameter:
                assert isinstance(view, nn.Parameter)
                spec.owner.register_parameter(spec.name, view)
            else:
                assert isinstance(view, torch.Tensor)
                spec.owner.register_buffer(spec.name, view, persistent=spec.persistent)

    def initialize(self) -> None:
        """Allocate packed storage, copy current tensors, and bind CPU views."""
        if self.layouts:
            raise RuntimeError("ModuleOffloadManager has already been initialized")

        cpu_offset = 0
        for name, module in self.groups.items():
            layout = self._collect_group_layout(name, module, cpu_offset)
            self.layouts[name] = layout
            cpu_offset = _align_offset(layout.cpu_offset + layout.nbytes)

        total_cpu_bytes = _align_offset(cpu_offset)
        max_gpu_bytes = max(layout.nbytes for layout in self.layouts.values())
        logger.info(
            "Module offload storage layout: "
            f"cpu_total={_format_bytes(total_cpu_bytes)}, "
            f"gpu_arena={_format_bytes(max_gpu_bytes)}, "
            f"groups=[{self._group_size_summary()}], device={self.device}"
        )

        self.cpu_storage = self._allocate_cpu_storage(total_cpu_bytes)

        for layout in self.layouts.values():
            self._copy_group_to_cpu_storage(layout)

        self.gpu_arena = self._allocate_gpu_arena(max_gpu_bytes)

        for name, layout in self.layouts.items():
            assert self.cpu_storage is not None
            assert self.gpu_arena is not None
            layout.cpu_views = self._make_views(layout, self.cpu_storage, use_gpu_offsets=False)
            layout.gpu_views = self._make_views(layout, self.gpu_arena, use_gpu_offsets=True)
            self._rebind_to_cpu(name)

    def _get_layout(self, name: str) -> _GroupLayout:
        try:
            return self.layouts[name]
        except KeyError as e:
            raise KeyError(
                f"Unknown offload group '{name}'. Available groups: {sorted(self.layouts)}"
            ) from e

    def stage(self, name: str) -> None:
        """Stage one offload group into the GPU arena and rebind its tensors."""
        layout = self._get_layout(name)
        if self.active_group_name == name:
            return
        if self.cpu_storage is None or self.gpu_arena is None:
            raise RuntimeError("ModuleOffloadManager must be initialized before staging")

        if self.active_group_name is not None:
            self._rebind_to_cpu(self.active_group_name)
            self.active_group_name = None

        src = self.cpu_storage.narrow(0, layout.cpu_offset, layout.nbytes)
        dst = self.gpu_arena.narrow(0, 0, layout.nbytes)
        try:
            dst.copy_(src, non_blocking=self.cpu_storage.is_pinned())
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to stage offload group '{name}' ({_format_bytes(layout.nbytes)}) "
                f"to {self.device}"
            ) from e
        self._rebind_to_gpu(name)
        self.active_group_name = name

    def _rebind_to_cpu(self, name: str) -> None:
        layout = self._get_layout(name)
        if not layout.cpu_views:
            raise RuntimeError("ModuleOffloadManager must be initialized before staging")
        self._bind_views(layout, layout.cpu_views)

    def _rebind_to_gpu(self, name: str) -> None:
        layout = self._get_layout(name)
        if not layout.gpu_views:
            raise RuntimeError("ModuleOffloadManager must be initialized before staging")
        self._bind_views(layout, layout.gpu_views)


class OffloadPipeline:
    """Stage offload groups explicitly from model call-site contexts.

    This class intentionally does not use forward hooks. Pipeline code must wrap
    the relevant call site with ``with self.offload_context("group")`` so staging
    happens before the model invocation and outside any later CUDA graph capture.
    """

    def __init__(
        self,
        stages: Sequence[Sequence[str] | str],
        parts: Mapping[str, nn.Module],
        device: torch.device | str,
        pin_memory: bool = True,
    ) -> None:
        if not stages:
            raise ValueError("At least one offload pipeline stage must be provided")

        self.stages = tuple((stage,) if isinstance(stage, str) else tuple(stage) for stage in stages)
        self.parts = dict(parts)
        self.device = torch.device(device)
        self.pin_memory = pin_memory
        self.manager = ModuleOffloadManager(
            groups=self._build_groups(),
            device=self.device,
            pin_memory=self.pin_memory,
        )

    def _build_groups(self) -> dict[str, nn.Module]:
        groups: dict[str, nn.Module] = {}
        for stage in self.stages:
            group_name = self._stage_name(stage)
            if not stage:
                raise ValueError("Offload pipeline stages must have at least one part")
            if group_name in groups:
                raise ValueError(f"Duplicate offload pipeline stage: {group_name}")

            modules: list[nn.Module] = []
            for part_name in stage:
                try:
                    part = self.parts[part_name]
                except KeyError as e:
                    raise KeyError(
                        f"Unknown offload pipeline part '{part_name}' for stage "
                        f"'{group_name}'. Available parts: {sorted(self.parts)}"
                    ) from e
                modules.append(part)

            group_module = modules[0] if len(modules) == 1 else nn.ModuleList(modules)
            groups[group_name] = group_module

        return groups

    def initialize(self) -> None:
        """Allocate and populate backing storage for all configured stages."""
        self.manager.initialize()

    @staticmethod
    def _stage_name(stage: Sequence[str] | str) -> str:
        return stage if isinstance(stage, str) else "+".join(stage)

    def context(self, group_name: str):
        """Stage ``group_name`` and return a no-op context manager."""
        self.manager.stage(group_name)
        # The active group intentionally stays resident after the call site.
        # The next stage() call rebinds it back to CPU before staging another
        # group, and cleanup() handles the final rebind when the pipeline exits.
        return nullcontext()

    def cleanup(self) -> None:
        """Return the active group to CPU-backed views."""
        if self.manager.active_group_name is not None:
            self.manager._rebind_to_cpu(self.manager.active_group_name)
            self.manager.active_group_name = None
