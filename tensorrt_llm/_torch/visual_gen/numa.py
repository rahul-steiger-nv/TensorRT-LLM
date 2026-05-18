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
"""NUMA helpers for visual-generation shared CPU offload storage."""

import hashlib
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.distributed as dist

_SYSFS_NODE_ROOT = Path("/sys/devices/system/node")


@dataclass(frozen=True)
class RankNumaInfo:
    hostname: str
    rank: int
    local_rank: int
    numa_node: int | None
    affinity_numa_nodes: tuple[int, ...]

    @property
    def group_key(self) -> tuple[str, int] | None:
        if self.numa_node is None:
            return None
        return self.hostname, self.numa_node


@dataclass(frozen=True)
class NumaSharedOffloadContext:
    rank_info: RankNumaInfo
    group_key: tuple[str, int]
    group_ranks: tuple[int, ...]
    writer_rank: int
    process_group: Any


def parse_cpu_list(cpu_list: str) -> set[int]:
    cpus: set[int] = set()
    for token in cpu_list.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, end = token.split("-", 1)
            cpus.update(range(int(start), int(end) + 1))
        else:
            cpus.add(int(token))
    return cpus


def read_numa_node_cpus(sysfs_root: Path = _SYSFS_NODE_ROOT) -> dict[int, set[int]]:
    numa_node_cpus: dict[int, set[int]] = {}
    if not sysfs_root.exists():
        return numa_node_cpus

    for node_dir in sysfs_root.iterdir():
        if not node_dir.name.startswith("node"):
            continue
        node_id_text = node_dir.name.removeprefix("node")
        if not node_id_text.isdigit():
            continue
        cpulist_path = node_dir / "cpulist"
        try:
            numa_node_cpus[int(node_id_text)] = parse_cpu_list(cpulist_path.read_text().strip())
        except OSError:
            continue
    return numa_node_cpus


def select_numa_node(
    affinity_cpus: set[int],
    numa_node_cpus: dict[int, set[int]],
) -> tuple[int | None, tuple[int, ...]]:
    overlaps = {
        node: len(affinity_cpus & cpus)
        for node, cpus in numa_node_cpus.items()
        if affinity_cpus & cpus
    }
    if not overlaps:
        return None, ()

    max_overlap = max(overlaps.values())
    selected_node = min(node for node, overlap in overlaps.items() if overlap == max_overlap)
    return selected_node, tuple(sorted(overlaps))


def _current_affinity_cpus() -> set[int]:
    try:
        return set(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return set()


def _current_local_rank(rank: int) -> int:
    return int(os.environ.get("LOCAL_RANK", rank))


def get_current_rank_numa_info(rank: int, local_rank: int | None = None) -> RankNumaInfo:
    numa_node, affinity_numa_nodes = select_numa_node(_current_affinity_cpus(), read_numa_node_cpus())
    if local_rank is None:
        local_rank = _current_local_rank(rank)
    return RankNumaInfo(
        hostname=socket.gethostname(),
        rank=rank,
        local_rank=local_rank,
        numa_node=numa_node,
        affinity_numa_nodes=affinity_numa_nodes,
    )


def _rank_numa_info_to_dict(info: RankNumaInfo) -> dict[str, Any]:
    return {
        "hostname": info.hostname,
        "rank": info.rank,
        "local_rank": info.local_rank,
        "numa_node": info.numa_node,
        "affinity_numa_nodes": list(info.affinity_numa_nodes),
    }


def _rank_numa_info_from_dict(info: dict[str, Any]) -> RankNumaInfo:
    return RankNumaInfo(
        hostname=info["hostname"],
        rank=info["rank"],
        local_rank=info["local_rank"],
        numa_node=info["numa_node"],
        affinity_numa_nodes=tuple(info["affinity_numa_nodes"]),
    )


def group_ranks_by_numa(
    rank_infos: list[RankNumaInfo],
) -> dict[tuple[str, int], tuple[int, ...]]:
    groups: dict[tuple[str, int], list[int]] = {}
    for info in rank_infos:
        if info.group_key is None:
            continue
        groups.setdefault(info.group_key, []).append(info.rank)
    return {key: tuple(sorted(ranks)) for key, ranks in groups.items()}


def get_numa_group_writer_rank(group_ranks: tuple[int, ...]) -> int:
    if not group_ranks:
        raise ValueError("NUMA shared offload group must contain at least one rank")
    return min(group_ranks)


def gather_rank_numa_infos() -> list[RankNumaInfo]:
    rank = dist.get_rank()
    rank_info = get_current_rank_numa_info(rank)
    rank_info_dicts: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(rank_info_dicts, _rank_numa_info_to_dict(rank_info))
    return [_rank_numa_info_from_dict(info) for info in rank_info_dicts if info is not None]


def _require_complete_numa_detection(rank_infos: list[RankNumaInfo]) -> None:
    missing_ranks = [info.rank for info in rank_infos if info.group_key is None]
    if missing_ranks:
        raise RuntimeError(
            "Could not determine NUMA node for ranks "
            f"{missing_ranks}. Check process CPU affinity and /sys/devices/system/node."
        )


def create_numa_shared_offload_context(
    require_complete_numa_detection: bool = False,
) -> NumaSharedOffloadContext | None:
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return None

    rank_infos = gather_rank_numa_infos()
    if require_complete_numa_detection:
        _require_complete_numa_detection(rank_infos)

    rank = dist.get_rank()
    rank_info = next(info for info in rank_infos if info.rank == rank)
    groups = group_ranks_by_numa(rank_infos)
    current_group_key = rank_info.group_key
    current_context: NumaSharedOffloadContext | None = None

    # All ranks must create groups in the same order, including groups they do
    # not join, to keep distributed setup deterministic.
    for group_key, group_ranks in sorted(groups.items()):
        process_group = dist.new_group(ranks=list(group_ranks))
        if group_key == current_group_key:
            current_context = NumaSharedOffloadContext(
                rank_info=rank_info,
                group_key=group_key,
                group_ranks=group_ranks,
                writer_rank=get_numa_group_writer_rank(group_ranks),
                process_group=process_group,
            )
    return current_context


def add_numa_suffix_to_path(path: str, hostname: str, numa_node: int) -> str:
    path_obj = Path(path)
    host_token = hashlib.sha1(hostname.encode("utf-8")).hexdigest()[:8]
    return str(path_obj.with_name(f"{path_obj.name}.host{host_token}.numa{numa_node}"))
