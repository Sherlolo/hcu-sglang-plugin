# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import torch
from sglang.srt.eplb import expert_distribution as N
from hcu_sglang.utils.detection import is_hcu


def _on_forward_pass_end(self, forward_pass_id: int, outputs: Dict[str, Any]):
    if not self._recording or not self._record_current_pass:
        return
    if torch.get_device_module().is_current_stream_capturing():
        return
    for gatherer_key, gatherer in self._single_pass_gatherers.items():
        single_pass_data = gatherer.collect()
        self._accumulator.append(
            forward_pass_id, gatherer_key, single_pass_data, outputs
        )


def init_new(
    expert_location_metadata: ExpertLocationMetadata,
    rank: int,
) -> _SinglePassGatherer:
    if N.get_exec().moe.expert_distribution_recorder_mode == "per_token":
        return N._DetailSinglePassGatherer(expert_location_metadata, rank)

    if N.get_exec().moe.moe_a2a_backend == "mori":
        return N._DeepepLowLatencySinglePassGatherer(expert_location_metadata, rank)

    if N.get_exec().moe.expert_distribution_recorder_mode == "stat_approx":
        if N.get_exec().moe.moe_a2a_backend != "none" and (
            N.get_exec().moe.deepep_mode == "normal"
        ):
            return N._DeepepNormalSinglePassGatherer(expert_location_metadata, rank)
        else:
            raise NotImplementedError

    if N.get_exec().moe.moe_a2a_backend == "deepep":
        if N.get_exec().moe.deepep_mode == "normal":
            return N._SelectExpertsSinglePassGatherer(expert_location_metadata, rank)
        elif N.get_exec().moe.deepep_mode == "auto":
            # ``auto`` switches between normal dispatch for extend/prefill
            # and low-latency dispatch for decode.  Use the dispatch counts
            # from both paths: on HIP, CUDA-graph padding has a valid expert
            # id with zero routing weight, so counting top-k ids directly
            # would incorrectly attribute padded rows to expert 0.
            return _DeepepAutoSinglePassGatherer(
                expert_location_metadata,
                rank,
                elastic_ep_enabled=N.get_exec().moe.elastic_ep_backend is not None,
            )
        elif N.get_exec().moe.deepep_mode == "low_latency":
            return N._DeepepLowLatencySinglePassGatherer(
                expert_location_metadata,
                rank,
                elastic_ep_enabled=N.get_exec().moe.elastic_ep_backend is not None,
            )
        else:
            raise NotImplementedError

    # Non-DeepEP a2a backends (flashinfer / nixl / mooncake / megamoe) and
    # no-a2a path dispatch through the standard topk select_experts.
    return N._SelectExpertsSinglePassGatherer(expert_location_metadata, rank)


class _DeepepAutoSinglePassGatherer(N._DeepepLowLatencySinglePassGatherer):
    """Collect exact local dispatch counts across DeepEP auto-mode switches."""

    def on_deepep_dispatch_normal(
        self,
        layer_idx: int,
        local_physical_count_of_layer: List[int],
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
    ):
        local_count = torch.as_tensor(
            local_physical_count_of_layer,
            dtype=self._data.dtype,
            device=self._data.device,
        )
        if local_count.shape != self._data[layer_idx].shape:
            raise ValueError(
                "DeepEP normal dispatch count shape mismatch in auto recorder: "
                f"{tuple(local_count.shape)} vs {tuple(self._data[layer_idx].shape)}"
            )
        self._data[layer_idx, :] += local_count

def dump(self, output_mode: _OutputMode):
    buffered_global_physical_count = (
        self._global_physical_count_of_buffered_step.get_all()
    )
    N.logger.info(
        "[EPLB dump][rank=%s] start, buffered_steps=%s",
        self._rank,
        buffered_global_physical_count.shape[0],
    )

    # EPLB ultimately sums the recorded steps before rebalancing. Sum locally
    # first so the distributed collective has a small, fixed shape even when
    # different ranks recorded a different number of forward passes. The
    # per-step buffer is int32, but a 1,000-step window with 65K-token prefills
    # and cross-rank reduction can exceed int32, so accumulate in int64.
    global_physical_count = buffered_global_physical_count.sum(
        dim=0, keepdim=True, dtype=torch.int64
    )
    logical_count_of_buffered_step = N._convert_global_physical_count_to_logical_count(
        global_physical_count,
        num_layers=self._expert_location_metadata.num_layers,
        num_logical_experts=self._expert_location_metadata.num_logical_experts,
        physical_to_logical_map=self._expert_location_metadata.physical_to_logical_map,
    ).squeeze(0)

    if self._first_dump:
        self._first_dump = False
        torch.get_device_module().empty_cache()

    N.logger.info("[EPLB dump][rank=%s] before all_reduce", self._rank)
    torch.distributed.all_reduce(
        logical_count_of_buffered_step, op=torch.distributed.ReduceOp.SUM
    )
    N.logger.info("[EPLB dump][rank=%s] after all_reduce", self._rank)

    output = dict(
        rank=self._rank,
        logical_count=logical_count_of_buffered_step,
        average_utilization_rate_over_window=self._get_global_average_utilization_rate(),
    )

    if output_mode == "file":
        if self._rank == 0:
            _dump_to_file(f"expert_distribution_recorder_{N.time.time()}.pt", output)
    elif output_mode == "object":
        return output
    else:
        raise NotImplementedError


def _dump_to_file(name, data):
    save_dir = Path(N.envs.SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR.get())
    path_output = save_dir / name
    N.logger.info("Writing expert distribution to %s", path_output)
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, str(path_output))
    N.logger.info("Finished writing expert distribution to %s", path_output)


def _around(implementation):
    def hook(original, *args, **kwargs):
        if not is_hcu():
            return original(*args, **kwargs)
        return implementation(*args, **kwargs)
    return hook


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    for target, implementation in [
        ("sglang.srt.eplb.expert_distribution._ExpertDistributionRecorderReal._on_forward_pass_end", _on_forward_pass_end),
        ("sglang.srt.eplb.expert_distribution._SinglePassGatherer.init_new", init_new),
        ("sglang.srt.eplb.expert_distribution._StatAccumulator.dump", dump),
        ("sglang.srt.eplb.expert_distribution._dump_to_file", _dump_to_file),
    ]:
        HookRegistry.register(target, _around(implementation), HookType.AROUND)
