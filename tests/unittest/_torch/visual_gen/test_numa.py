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

from pathlib import Path

from tensorrt_llm._torch.visual_gen.numa import (
    RankNumaInfo,
    add_numa_suffix_to_path,
    get_numa_group_writer_rank,
    group_ranks_by_numa,
    parse_cpu_list,
    select_numa_node,
)


def test_parse_cpu_list() -> None:
    assert parse_cpu_list("0-3,8,10-11") == {0, 1, 2, 3, 8, 10, 11}


def test_select_numa_node_prefers_largest_affinity_overlap() -> None:
    numa_node_cpus = {
        0: {0, 1, 2, 3},
        1: {4, 5, 6, 7},
    }

    selected_node, affinity_numa_nodes = select_numa_node({1, 4, 5}, numa_node_cpus)

    assert selected_node == 1
    assert affinity_numa_nodes == (0, 1)


def test_select_numa_node_tie_breaks_to_lowest_node() -> None:
    numa_node_cpus = {
        0: {0, 1},
        1: {2, 3},
    }

    selected_node, affinity_numa_nodes = select_numa_node({1, 3}, numa_node_cpus)

    assert selected_node == 0
    assert affinity_numa_nodes == (0, 1)


def test_select_numa_node_returns_none_without_affinity_overlap() -> None:
    selected_node, affinity_numa_nodes = select_numa_node(
        {8, 9},
        {
            0: {0, 1},
            1: {2, 3},
        },
    )

    assert selected_node is None
    assert affinity_numa_nodes == ()


def test_group_ranks_by_numa_uses_host_and_node_key() -> None:
    rank_infos = [
        RankNumaInfo("host-a", 0, 0, 0, (0,)),
        RankNumaInfo("host-a", 1, 1, 0, (0,)),
        RankNumaInfo("host-a", 2, 2, 1, (1,)),
        RankNumaInfo("host-b", 3, 0, 0, (0,)),
        RankNumaInfo("host-b", 4, 1, None, ()),
    ]

    assert group_ranks_by_numa(rank_infos) == {
        ("host-a", 0): (0, 1),
        ("host-a", 1): (2,),
        ("host-b", 0): (3,),
    }


def test_get_numa_group_writer_rank_selects_lowest_rank() -> None:
    assert get_numa_group_writer_rank((5, 3, 7)) == 3


def test_add_numa_suffix_to_path_keeps_parent_and_makes_node_unique(tmp_path: Path) -> None:
    base_path = tmp_path / "offload.bin"

    node0_path = Path(add_numa_suffix_to_path(str(base_path), "host-a", 0))
    node1_path = Path(add_numa_suffix_to_path(str(base_path), "host-a", 1))
    other_host_path = Path(add_numa_suffix_to_path(str(base_path), "host-b", 0))

    assert node0_path.parent == tmp_path
    assert node0_path != node1_path
    assert node0_path != other_host_path
    assert "numa0" in node0_path.name
    assert "numa1" in node1_path.name
