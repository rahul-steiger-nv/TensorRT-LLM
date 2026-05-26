# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Cosmos3 torch.compile / CUDA graph setup."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from tensorrt_llm._torch.visual_gen.config import PipelineConfig
from tensorrt_llm._torch.visual_gen.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniMoTPipeline


class _ToyGenLayer(nn.Module):

    def forward(self, hidden_states, k_und, v_und, freqs):
        return hidden_states


class _ToyCosmosTransformer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.gen_layers = nn.ModuleList([_ToyGenLayer(), _ToyGenLayer()])
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([_ToyGenLayer(), _ToyGenLayer()])
        self.cached_kv = None
        self.cached_freqs_gen = None
        self.use_seq_parallel = False

    def forward(self, **kwargs):
        return kwargs["hidden_states"]


class _ToyCosmosPipeline(Cosmos3OmniMoTPipeline):

    def _init_transformer(self) -> None:
        self.transformer = _ToyCosmosTransformer()


def _make_config(*, enable_cuda_graph: bool = False, enable_torch_compile: bool = False):
    return SimpleNamespace(
        pretrained_config=SimpleNamespace(),
        device="cuda",
        cuda_graph=SimpleNamespace(enable_cuda_graph=enable_cuda_graph),
        torch_compile=SimpleNamespace(
            enable_torch_compile=enable_torch_compile,
            enable_fullgraph=False,
        ),
        pipeline=PipelineConfig(enable_offloading=False),
    )


def test_cosmos3_cuda_graph_wraps_forward_with_torch_compile():
    pipeline = _ToyCosmosPipeline(
        _make_config(enable_cuda_graph=True, enable_torch_compile=True)
    )

    assert "transformer" in pipeline._cuda_graph_runners
    assert pipeline._cuda_graph_runners["transformer"].enabled
    assert hasattr(pipeline.transformer, "_eager_forward")


def test_cosmos3_skips_cuda_graphs_when_ulysses_enabled():
    pipeline = _ToyCosmosPipeline(_make_config(enable_cuda_graph=True))
    pipeline.transformer.use_seq_parallel = True
    pipeline._cuda_graph_runners.clear()
    pipeline._setup_cuda_graphs()

    assert pipeline._cuda_graph_runners == {}
