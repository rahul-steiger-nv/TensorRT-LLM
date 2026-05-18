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
"""Module parameter offloading utilities for visual generation pipelines."""

import mmap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from tensorrt_llm.logger import logger


def _align_offset(offset: int, alignment: int = 256) -> int:
    return ((offset + alignment - 1) // alignment) * alignment


def _format_bytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


_CUDA_SUCCESS = 0
_CUDA_HOST_REGISTER_PORTABLE = 1


def _align_host_register_range(ptr: int, nbytes: int) -> tuple[int, int]:
    page_size = mmap.PAGESIZE
    aligned_ptr = ptr - (ptr % page_size)
    aligned_end = ((ptr + nbytes + page_size - 1) // page_size) * page_size
    return aligned_ptr, aligned_end - aligned_ptr


def _cuda_result_is_success(result: Any) -> bool:
    return result == _CUDA_SUCCESS or str(result) == "cudaError.success"


def _cuda_result_to_string(result: Any) -> str:
    return str(result)


def _load_cuda_runtime() -> Any:
    cudart = torch.cuda.cudart()
    if not hasattr(cudart, "cudaHostRegister") or not hasattr(cudart, "cudaHostUnregister"):
        raise AttributeError("torch.cuda.cudart() does not expose host registration APIs")
    return cudart


def _ensure_cuda_context(device: torch.device) -> None:
    with torch.cuda.device(device):
        torch.empty(0, device=device)


def _register_cuda_host_memory(
    ptr: int,
    nbytes: int,
) -> tuple[Any, str, int] | None:
    register_ptr, register_nbytes = _align_host_register_range(ptr, nbytes)

    try:
        cuda_runtime = _load_cuda_runtime()
        result = cuda_runtime.cudaHostRegister(
            register_ptr,
            register_nbytes,
            _CUDA_HOST_REGISTER_PORTABLE,
        )
    except (AttributeError, RuntimeError, TypeError, ValueError) as e:
        logger.warning(
            "Unable to CUDA-register shared module offload CPU storage. "
            "Falling back to pageable H2D copies: %s",
            e,
        )
        return None

    if _cuda_result_is_success(result):
        return cuda_runtime, "torch.cuda.cudart()", register_nbytes

    logger.warning(
        "Unable to CUDA-register shared module offload CPU storage. "
        "Falling back to pageable H2D copies: cudaHostRegister returned %s",
        _cuda_result_to_string(result),
    )
    return None


@dataclass(frozen=True)
class _OffloadGroup:
    name: str
    module: nn.Module


@dataclass(frozen=True)
class OffloadPipelinePart:
    """One model-defined offloadable module subtree."""

    module: nn.Module
    hook_modules: tuple[nn.Module, ...] = ()

    def modules_to_hook(self) -> tuple[nn.Module, ...]:
        return self.hook_modules or (self.module,)


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
    group: _OffloadGroup
    cpu_offset: int
    nbytes: int
    specs: list[_FlatTensorSpec]


@dataclass(frozen=True)
class SharedOffloadStorageConfig:
    """File-backed CPU storage shared by multiple visual-generation ranks."""

    path: str
    is_writer: bool
    process_group: Any | None = None
    unlink_on_cleanup: bool = False


class ModuleOffloadManager:
    """Pack module groups into CPU storage and stage one group on GPU."""

    def __init__(
        self,
        groups: Sequence[_OffloadGroup],
        device: torch.device | str,
        pin_memory: bool = True,
        shared_storage: SharedOffloadStorageConfig | None = None,
    ) -> None:
        if not groups:
            raise ValueError("At least one offload group must be provided")

        self.groups = list(groups)
        self.device = torch.device(device)
        self.pin_memory = pin_memory
        self.shared_storage = shared_storage
        self.cpu_storage: torch.Tensor | None = None
        self.gpu_arena: torch.Tensor | None = None
        self.layouts: dict[str, _GroupLayout] = {}
        self.active_group_name: str | None = None
        self._cpu_storage_cuda_registered = False
        self._cuda_host_register_ptr: int | None = None
        self._cuda_host_register_nbytes = 0
        self._cuda_runtime: Any | None = None

        seen_names: set[str] = set()
        for group in self.groups:
            if not group.name:
                raise ValueError("Offload group names must be non-empty")
            if group.name in seen_names:
                raise ValueError(f"Duplicate offload group name: {group.name}")
            if not isinstance(group.module, nn.Module):
                raise TypeError(f"Offload group '{group.name}' must contain an nn.Module")
            seen_names.add(group.name)

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
        group: _OffloadGroup,
        qualified_name: str,
        tensor: torch.Tensor,
        is_parameter: bool,
        cpu_offset: int,
        gpu_offset: int,
        absolute_cpu_offset: int | None = None,
    ) -> _FlatTensorSpec:
        display_name = f"{group.name}.{qualified_name}"
        if not tensor.is_contiguous():
            raise ValueError(
                f"Cannot offload non-contiguous tensor '{display_name}' "
                f"with stride {tuple(tensor.stride())}"
            )

        owner, name = self._owner_and_name(group.module, qualified_name)
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

    def _collect_group_layout(self, group: _OffloadGroup, cpu_offset: int) -> _GroupLayout:
        gpu_offset = 0
        specs: list[_FlatTensorSpec] = []
        seen_tensors: dict[tuple[int, int], _FlatTensorSpec] = {}

        for qualified_name, param in group.module.named_parameters(
            recurse=True,
            remove_duplicate=False,
        ):
            tensor = param.detach()
            display_name = f"{group.name}.{qualified_name}"
            alias = self._get_alias_spec(seen_tensors, tensor, display_name)
            if alias is None:
                gpu_offset = _align_offset(gpu_offset, tensor.element_size())
            specs.append(
                self._build_spec(
                    group=group,
                    qualified_name=qualified_name,
                    tensor=tensor,
                    is_parameter=True,
                    cpu_offset=cpu_offset,
                    gpu_offset=alias.gpu_offset if alias is not None else gpu_offset,
                    absolute_cpu_offset=alias.cpu_offset if alias is not None else None,
                )
            )
            if alias is None:
                key = self._storage_key(tensor)
                if key is not None:
                    seen_tensors[key] = specs[-1]
                gpu_offset += specs[-1].nbytes

        for qualified_name, buffer in group.module.named_buffers(
            recurse=True,
            remove_duplicate=False,
        ):
            tensor = buffer.detach()
            display_name = f"{group.name}.{qualified_name}"
            alias = self._get_alias_spec(seen_tensors, tensor, display_name)
            if alias is None:
                gpu_offset = _align_offset(gpu_offset, tensor.element_size())
            specs.append(
                self._build_spec(
                    group=group,
                    qualified_name=qualified_name,
                    tensor=tensor,
                    is_parameter=False,
                    cpu_offset=cpu_offset,
                    gpu_offset=alias.gpu_offset if alias is not None else gpu_offset,
                    absolute_cpu_offset=alias.cpu_offset if alias is not None else None,
                )
            )
            if alias is None:
                key = self._storage_key(tensor)
                if key is not None:
                    seen_tensors[key] = specs[-1]
                gpu_offset += specs[-1].nbytes

        if not specs:
            raise ValueError(f"Offload group '{group.name}' has no parameters or buffers")

        return _GroupLayout(
            group=group,
            cpu_offset=cpu_offset,
            nbytes=_align_offset(gpu_offset),
            specs=specs,
        )

    def _copy_group_to_cpu_storage(self, layout: _GroupLayout) -> None:
        assert self.cpu_storage is not None
        for spec in layout.specs:
            if spec.nbytes == 0:
                continue
            tensor = getattr(spec.owner, spec.name).detach()
            tensor_bytes = tensor.reshape(-1).view(torch.uint8).cpu()
            self.cpu_storage.narrow(0, spec.cpu_offset, spec.nbytes).copy_(tensor_bytes)

    def _typed_view(self, storage: torch.Tensor, offset: int, spec: _FlatTensorSpec) -> torch.Tensor:
        byte_view = storage.narrow(0, offset, spec.nbytes)
        typed_view = byte_view.view(spec.dtype)
        return typed_view.as_strided(spec.shape, spec.stride)

    def _bind_views(self, layout: _GroupLayout, storage: torch.Tensor, use_gpu_offsets: bool) -> None:
        for spec in layout.specs:
            offset = spec.gpu_offset if use_gpu_offsets else spec.cpu_offset
            view = self._typed_view(storage, offset, spec)
            if spec.is_parameter:
                spec.owner.register_parameter(
                    spec.name,
                    nn.Parameter(view, requires_grad=spec.requires_grad),
                )
            else:
                spec.owner.register_buffer(spec.name, view, persistent=spec.persistent)

    def _barrier(self) -> None:
        if self.shared_storage is None:
            return
        process_group = self.shared_storage.process_group
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size(group=process_group) > 1
        ):
            torch.distributed.barrier(group=process_group)

    def _allocate_cpu_storage(self, total_cpu_bytes: int) -> torch.Tensor:
        if self.shared_storage is None:
            return torch.empty(
                total_cpu_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=self.pin_memory,
            )

        path = Path(self.shared_storage.path)
        if self.shared_storage.is_writer:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as storage_file:
                storage_file.truncate(total_cpu_bytes)
        self._barrier()

        logger.info(
            "Using shared module offload CPU storage at %s (%s, writer=%s)",
            path,
            _format_bytes(total_cpu_bytes),
            self.shared_storage.is_writer,
        )
        return torch.from_file(
            str(path),
            shared=True,
            size=total_cpu_bytes,
            dtype=torch.uint8,
        )

    def _register_shared_cpu_storage(self) -> None:
        if (
            self.shared_storage is None
            or not self.pin_memory
            or self.device.type != "cuda"
            or not torch.cuda.is_available()
            or self.cpu_storage is None
        ):
            return

        ptr = self.cpu_storage.data_ptr()
        nbytes = self.cpu_storage.numel() * self.cpu_storage.element_size()
        if nbytes == 0:
            return

        try:
            _ensure_cuda_context(self.device)
        except RuntimeError as e:
            logger.warning(
                "Unable to initialize CUDA context for shared module offload CPU storage "
                "registration at %s: %s. Falling back to pageable H2D copies.",
                self.shared_storage.path,
                e,
            )
            return

        registration = _register_cuda_host_memory(ptr, nbytes)
        if registration is None:
            return
        cuda_runtime, cuda_host_register_api, register_nbytes = registration
        register_ptr, _ = _align_host_register_range(ptr, nbytes)

        self._cuda_runtime = cuda_runtime
        self._cuda_host_register_ptr = register_ptr
        self._cuda_host_register_nbytes = register_nbytes
        self._cpu_storage_cuda_registered = True
        logger.info(
            "CUDA-registered shared module offload CPU storage at %s (%s, api=%s)",
            self.shared_storage.path,
            _format_bytes(register_nbytes),
            cuda_host_register_api,
        )

    def _unregister_shared_cpu_storage(self) -> None:
        if (
            not self._cpu_storage_cuda_registered
            or self._cuda_runtime is None
            or self._cuda_host_register_ptr is None
        ):
            return

        result = self._cuda_runtime.cudaHostUnregister(self._cuda_host_register_ptr)
        if not _cuda_result_is_success(result):
            logger.warning(
                "Unable to CUDA-unregister shared module offload CPU storage at %s: "
                "%s returned %s.",
                self.shared_storage.path if self.shared_storage is not None else "<unknown>",
                "torch.cuda.cudart().cudaHostUnregister",
                _cuda_result_to_string(result),
            )
        self._cpu_storage_cuda_registered = False
        self._cuda_host_register_ptr = None
        self._cuda_host_register_nbytes = 0
        self._cuda_runtime = None

    def initialize(self, initial_group: str | None = None) -> None:
        if self.layouts:
            raise RuntimeError("ModuleOffloadManager has already been initialized")

        cpu_offset = 0
        for group in self.groups:
            layout = self._collect_group_layout(group, cpu_offset)
            self.layouts[group.name] = layout
            cpu_offset = _align_offset(layout.cpu_offset + layout.nbytes)

        total_cpu_bytes = _align_offset(cpu_offset)
        max_gpu_bytes = max(layout.nbytes for layout in self.layouts.values())
        group_sizes = ", ".join(
            f"{name}={_format_bytes(layout.nbytes)}" for name, layout in self.layouts.items()
        )
        logger.info(
            "Module offload storage layout: "
            f"cpu_total={_format_bytes(total_cpu_bytes)}, "
            f"gpu_arena={_format_bytes(max_gpu_bytes)}, "
            f"groups=[{group_sizes}], device={self.device}"
        )

        self.cpu_storage = self._allocate_cpu_storage(total_cpu_bytes)

        if self.shared_storage is None or self.shared_storage.is_writer:
            for layout in self.layouts.values():
                self._copy_group_to_cpu_storage(layout)
        if self.shared_storage is not None:
            self._barrier()
            self._register_shared_cpu_storage()

        for layout in self.layouts.values():
            assert self.cpu_storage is not None
            self._bind_views(layout, self.cpu_storage, use_gpu_offsets=False)

        self.gpu_arena = torch.empty(max_gpu_bytes, dtype=torch.uint8, device=self.device)

        for layout in self.layouts.values():
            assert self.gpu_arena is not None
            self._bind_views(layout, self.gpu_arena, use_gpu_offsets=True)

        if initial_group is not None:
            self.stage(initial_group)

    def cleanup(self) -> None:
        self._unregister_shared_cpu_storage()
        if self.shared_storage is None or not self.shared_storage.unlink_on_cleanup:
            return
        try:
            Path(self.shared_storage.path).unlink(missing_ok=True)
        except OSError as e:
            logger.warning(
                "Failed to unlink shared module offload CPU storage %s: %s",
                self.shared_storage.path,
                e,
            )

    def _get_layout(self, name: str) -> _GroupLayout:
        try:
            return self.layouts[name]
        except KeyError as e:
            raise KeyError(
                f"Unknown offload group '{name}'. Available groups: {sorted(self.layouts)}"
            ) from e

    def stage(self, name: str) -> None:
        layout = self._get_layout(name)
        if self.active_group_name == name:
            return
        if self.cpu_storage is None or self.gpu_arena is None:
            raise RuntimeError("ModuleOffloadManager must be initialized before staging")

        src = self.cpu_storage.narrow(0, layout.cpu_offset, layout.nbytes)
        dst = self.gpu_arena.narrow(0, 0, layout.nbytes)
        try:
            non_blocking = self.cpu_storage.is_pinned() or self._cpu_storage_cuda_registered
            dst.copy_(src, non_blocking=non_blocking)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to stage offload group '{name}' ({_format_bytes(layout.nbytes)}) "
                f"to {self.device}"
            ) from e
        self.active_group_name = name


class ForwardHookOffloadPipeline:
    """Stage offload groups automatically from module forward pre-hooks."""

    def __init__(
        self,
        stages: Sequence[Sequence[str] | str],
        parts: Mapping[str, OffloadPipelinePart],
        device: torch.device | str,
        pin_memory: bool = True,
        shared_storage: SharedOffloadStorageConfig | None = None,
    ) -> None:
        if not stages:
            raise ValueError("At least one offload pipeline stage must be provided")

        self.stages = tuple((stage,) if isinstance(stage, str) else tuple(stage) for stage in stages)
        self.parts = dict(parts)
        self.device = torch.device(device)
        self.pin_memory = pin_memory
        self.shared_storage = shared_storage
        self._hooks: list[RemovableHandle] = []
        self._hook_groups: dict[int, set[str]] = {}
        self._stage_names = tuple("+".join(stage) for stage in self.stages)
        self._stage_hook_modules: dict[str, tuple[nn.Module, ...]] = {}
        self.manager = ModuleOffloadManager(
            groups=self._build_groups(),
            device=self.device,
            pin_memory=self.pin_memory,
            shared_storage=self.shared_storage,
        )

    def _build_groups(self) -> list[_OffloadGroup]:
        groups: list[_OffloadGroup] = []
        seen_names: set[str] = set()
        for stage, group_name in zip(self.stages, self._stage_names):
            if not stage:
                raise ValueError("Offload pipeline stages must have at least one part")
            if group_name in seen_names:
                raise ValueError(f"Duplicate offload pipeline stage: {group_name}")
            seen_names.add(group_name)

            modules: list[nn.Module] = []
            hook_modules: list[nn.Module] = []
            for part_name in stage:
                try:
                    part = self.parts[part_name]
                except KeyError as e:
                    raise KeyError(
                        f"Unknown offload pipeline part '{part_name}' for stage "
                        f"'{group_name}'. Available parts: {sorted(self.parts)}"
                    ) from e
                modules.append(part.module)
                hook_modules.extend(part.modules_to_hook())

            group_module = modules[0] if len(modules) == 1 else nn.ModuleList(modules)
            groups.append(_OffloadGroup(group_name, group_module))
            self._stage_hook_modules[group_name] = self._unique_modules(hook_modules)

        return groups

    @staticmethod
    def _unique_modules(modules: Sequence[nn.Module]) -> tuple[nn.Module, ...]:
        unique: list[nn.Module] = []
        seen: set[int] = set()
        for module in modules:
            module_id = id(module)
            if module_id in seen:
                continue
            unique.append(module)
            seen.add(module_id)
        return tuple(unique)

    def initialize(self, initial_group: str | None = None) -> None:
        self.manager.initialize(initial_group=initial_group)
        self._register_hooks()

    def _register_hooks(self) -> None:
        if self._hooks:
            return

        for group_name in self._stage_names:
            for module in self._stage_hook_modules[group_name]:
                module_id = id(module)
                groups = self._hook_groups.setdefault(module_id, set())
                if groups and group_name not in groups:
                    logger.warning(
                        "Registering multiple offload groups %s on module %s",
                        sorted([*groups, group_name]),
                        module.__class__.__name__,
                    )
                groups.add(group_name)

                def stage_group(_module, _args, group_name=group_name):
                    self.manager.stage(group_name)

                self._hooks.append(module.register_forward_pre_hook(stage_group))

    def remove_hooks(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        self._hook_groups.clear()

    def cleanup(self) -> None:
        self.remove_hooks()
        self.manager.cleanup()
