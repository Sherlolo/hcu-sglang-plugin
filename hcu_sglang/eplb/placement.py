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
import random
from typing import List, Optional
import torch
from sglang.srt.eplb import expert_location as N
from sglang.srt.arg_groups import parallel_hook as P
from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.config import ep_static_dispatch_policy, configure_ep_static_dispatch_policy


def static_dispatch_policy():
    cfg = N.get_exec().moe
    if hasattr(cfg, "ep_static_dispatch_policy"):
        return cfg.ep_static_dispatch_policy
    return ep_static_dispatch_policy()


def _compute_logical_to_all_physical_map(
    physical_to_logical_map: torch.Tensor,
    num_logical_experts: int,
    ep_size: int,
    moe_ep_rank: int,
):

    # This is rarely called, so we use for loops for maximum clarity

    num_layers, num_physical_experts = physical_to_logical_map.shape

    logical_to_all_physical_map = [
        [[] for _ in range(num_logical_experts)] for _ in range(num_layers)
    ]

    # Find out the candidate physical experts for each logical expert on each layer
    for layer_id in range(num_layers):
        for physical_expert_id in range(num_physical_experts):
            logical_expert_id = physical_to_logical_map[
                layer_id, physical_expert_id
            ].item()
            logical_to_all_physical_map[layer_id][logical_expert_id].append(
                physical_expert_id
            )

    # Replace by the physical expert on local GPU or node if possible. Skipped
    # without an a2a backend, where all EP ranks must agree on the pick: this
    # collapse is per-rank, and the full candidate list is what lets the dispatch
    # spread a hot expert over its replicas. See ExpertLocationDispatchInfo.
    # LP and locality_fair need the complete replica set. Collapsing it here
    # makes LP probabilities ineffective and prevents fair source-rank choices.
    if (
        moe_ep_rank is not None
        and N.get_exec().moe.moe_a2a_backend != "none"
        and N.get_exec().moe.ep_dispatch_algorithm == "static"
        and static_dispatch_policy() == "nearest"
    ):
        num_local_gpu_physical_experts = num_physical_experts // ep_size
        prefer_same_node = N._prefer_same_node_experts()
        num_gpus_per_node = (
            N.get_parallel().ep_size // N.get_parallel().nnodes
            if prefer_same_node
            else None
        )
        num_local_node_physical_experts = (
            num_local_gpu_physical_experts * num_gpus_per_node
            if num_gpus_per_node is not None
            else None
        )
        for layer_id in range(num_layers):
            for logical_expert_id in range(num_logical_experts):
                # Try to find the nearest physical expert
                nearest_expert = N._find_nearest_expert(
                    candidate_physical_expert_ids=logical_to_all_physical_map[layer_id][
                        logical_expert_id
                    ],
                    num_local_gpu_physical_experts=num_local_gpu_physical_experts,
                    moe_ep_rank=moe_ep_rank,
                    num_gpus_per_node=num_gpus_per_node,
                    num_local_node_physical_experts=num_local_node_physical_experts,
                )

                # Replace by the nearest physical expert
                if nearest_expert != -1:
                    logical_to_all_physical_map[layer_id][logical_expert_id] = [
                        nearest_expert
                    ]

    logical_to_all_physical_map = N._pad_nested_array(
        logical_to_all_physical_map, pad_value=-1
    )

    return torch.tensor(
        logical_to_all_physical_map, device=physical_to_logical_map.device
    )


def compute_logical_to_rank_dispatch_physical_map(
    logical_to_all_physical_map: torch.Tensor,
    ep_size: int,
    num_physical_experts: int,
    ep_rank: int,
    seed: int = 42,
):

    r = random.Random(seed)
    dispatch_policy = static_dispatch_policy()
    if dispatch_policy not in ("nearest", "locality_fair"):
        raise ValueError(f"Unknown static expert dispatch policy: {dispatch_policy}")

    device = logical_to_all_physical_map.device
    logical_to_all_physical_map = logical_to_all_physical_map.cpu()

    num_local_gpu_physical_experts = num_physical_experts // ep_size
    prefer_same_node = N._prefer_same_node_experts()
    num_gpus_per_node = (
        N.get_parallel().ep_size // N.get_parallel().nnodes if prefer_same_node else None
    )
    num_local_node_physical_experts = (
        num_local_gpu_physical_experts * num_gpus_per_node
        if num_gpus_per_node is not None
        else None
    )
    num_layers, num_logical_experts, _ = logical_to_all_physical_map.shape
    dtype = logical_to_all_physical_map.dtype

    result_list = [
        [[-1] * num_logical_experts for _ in range(num_layers)] for _ in range(ep_size)
    ]

    for layer_id in range(num_layers):
        for logical_expert_id in range(num_logical_experts):
            candidate_physical_expert_ids = N._logical_to_all_physical_raw(
                logical_to_all_physical_map, layer_id, logical_expert_id
            )

            if dispatch_policy == "locality_fair":
                choices = _assign_locality_fair_experts(
                    candidate_physical_expert_ids=candidate_physical_expert_ids,
                    ep_size=ep_size,
                    num_local_gpu_physical_experts=num_local_gpu_physical_experts,
                    num_gpus_per_node=num_gpus_per_node,
                    num_local_node_physical_experts=num_local_node_physical_experts,
                    r=r,
                )
                for moe_ep_rank, choice in enumerate(choices):
                    result_list[moe_ep_rank][layer_id][logical_expert_id] = choice
                continue

            remaining_ranks = []
            for moe_ep_rank in range(ep_size):
                val = N._find_nearest_expert(
                    candidate_physical_expert_ids=candidate_physical_expert_ids,
                    num_local_gpu_physical_experts=num_local_gpu_physical_experts,
                    moe_ep_rank=moe_ep_rank,
                    num_gpus_per_node=num_gpus_per_node,
                    num_local_node_physical_experts=num_local_node_physical_experts,
                )

                result_list[moe_ep_rank][layer_id][logical_expert_id] = val
                if val == -1:
                    remaining_ranks.append(moe_ep_rank)

            if remaining_ranks:
                choices = N._fair_choices(
                    candidate_physical_expert_ids, k=len(remaining_ranks), r=r
                )
                for moe_ep_rank, choice in zip(remaining_ranks, choices, strict=True):
                    result_list[moe_ep_rank][layer_id][logical_expert_id] = choice

    logical_to_rank_dispatch_physical_map = torch.tensor(result_list, dtype=dtype)
    assert torch.all(logical_to_rank_dispatch_physical_map != -1)

    return logical_to_rank_dispatch_physical_map[ep_rank, :, :].to(device)


def _assign_locality_fair_experts(
    candidate_physical_expert_ids: List[int],
    ep_size: int,
    num_local_gpu_physical_experts: int,
    num_gpus_per_node: Optional[int],
    num_local_node_physical_experts: Optional[int],
    r: random.Random,
) -> List[int]:
    """Build one deterministic static source-rank-to-replica assignment.

    Locality is prioritized before fairness: same GPU, then same node when the
    topology exposes node boundaries, then remote. Assignment counts are local
    to this logical expert; they balance static source-rank bindings, not live
    token or queue load.
    """
    if ep_size <= 0:
        raise ValueError(f"ep_size must be positive, got {ep_size}")
    if num_local_gpu_physical_experts <= 0:
        raise ValueError(
            "num_local_gpu_physical_experts must be positive, got "
            f"{num_local_gpu_physical_experts}"
        )
    if not candidate_physical_expert_ids:
        raise ValueError("locality_fair requires at least one physical replica")
    if (num_gpus_per_node is None) != (num_local_node_physical_experts is None):
        raise ValueError(
            "num_gpus_per_node and num_local_node_physical_experts must either "
            "both be set or both be None"
        )

    num_nodes = None
    if num_gpus_per_node is not None:
        if num_gpus_per_node <= 0 or ep_size % num_gpus_per_node != 0:
            raise ValueError(
                f"ep_size ({ep_size}) must be divisible by positive "
                f"num_gpus_per_node ({num_gpus_per_node})"
            )
        expected_node_experts = num_local_gpu_physical_experts * num_gpus_per_node
        if num_local_node_physical_experts != expected_node_experts:
            raise ValueError(
                "num_local_node_physical_experts must equal "
                "num_local_gpu_physical_experts * num_gpus_per_node, got "
                f"{num_local_node_physical_experts} != {expected_node_experts}"
            )
        num_nodes = ep_size // num_gpus_per_node

    assignments = [-1] * ep_size
    assignment_counts = {
        physical_expert_id: 0 for physical_expert_id in candidate_physical_expert_ids
    }

    candidates_by_gpu = [[] for _ in range(ep_size)]
    candidates_by_node = [[] for _ in range(num_nodes or 0)]
    for physical_expert_id in candidate_physical_expert_ids:
        gpu_id = N._compute_gpu_id_of_physical_expert(
            physical_expert_id, num_local_gpu_physical_experts
        )
        if physical_expert_id < 0 or gpu_id >= ep_size:
            max_physical_expert_id = ep_size * num_local_gpu_physical_experts - 1
            raise ValueError(
                f"physical expert {physical_expert_id} is outside topology "
                f"range [0, {max_physical_expert_id}]"
            )
        candidates_by_gpu[gpu_id].append(physical_expert_id)
        if num_nodes is not None:
            node_id = N._compute_node_id_of_physical_expert(
                physical_expert_id, num_local_node_physical_experts
            )
            candidates_by_node[node_id].append(physical_expert_id)

    # A same-GPU replica is always the strongest locality choice.
    for moe_ep_rank in range(ep_size):
        if candidates_by_gpu[moe_ep_rank]:
            choice = _choose_least_assigned_expert(
                candidates_by_gpu[moe_ep_rank], assignment_counts, r
            )
            assignments[moe_ep_rank] = choice
            assignment_counts[choice] += 1

    # Include same-GPU counts while balancing unassigned ranks within a node.
    if num_gpus_per_node is not None:
        for node_id, same_node_physical_expert_ids in enumerate(candidates_by_node):
            if not same_node_physical_expert_ids:
                continue

            node_rank_begin = node_id * num_gpus_per_node
            remaining_ranks = [
                moe_ep_rank
                for moe_ep_rank in range(
                    node_rank_begin, node_rank_begin + num_gpus_per_node
                )
                if assignments[moe_ep_rank] == -1
            ]
            r.shuffle(remaining_ranks)
            for moe_ep_rank in remaining_ranks:
                choice = _choose_least_assigned_expert(
                    same_node_physical_expert_ids, assignment_counts, r
                )
                assignments[moe_ep_rank] = choice
                assignment_counts[choice] += 1

    # Nodes without a replica, or flat topologies without node affinity, fall
    # back to all replicas while preserving counts from the locality passes.
    remaining_ranks = [
        moe_ep_rank
        for moe_ep_rank, assignment in enumerate(assignments)
        if assignment == -1
    ]
    r.shuffle(remaining_ranks)
    for moe_ep_rank in remaining_ranks:
        choice = _choose_least_assigned_expert(
            candidate_physical_expert_ids, assignment_counts, r
        )
        assignments[moe_ep_rank] = choice
        assignment_counts[choice] += 1

    return assignments


def _choose_least_assigned_expert(
    candidate_physical_expert_ids: List[int],
    assignment_counts: dict[int, int],
    r: random.Random,
) -> int:
    candidates = candidate_physical_expert_ids.copy()
    r.shuffle(candidates)
    return min(candidates, key=assignment_counts.__getitem__)


def _around(implementation):
    def hook(original, *args, **kwargs):
        if not is_hcu():
            return original(*args, **kwargs)
        return implementation(*args, **kwargs)
    return hook


def after_cli(result, parser):
    if is_hcu() and "--ep-static-dispatch-policy" not in parser._option_string_actions:
        parser.add_argument("--ep-static-dispatch-policy", dest="hcu_plugin_ep_static_dispatch_policy", choices=("nearest", "locality_fair"), default=None, help="Choose nearest or locality-fair static replica placement.")
    return result


def before_seed(cls, args):
    if is_hcu():
        configure_ep_static_dispatch_policy(getattr(args, "hcu_plugin_ep_static_dispatch_policy", None))


def around_validation(original, server_args):
    result = original(server_args)
    if is_hcu() and ep_static_dispatch_policy() != "nearest" and P.resolving_view(server_args).ep_dispatch_algorithm != "static":
        raise ValueError("locality_fair replica policy requires static expert dispatch")
    return result


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    for name in ("_compute_logical_to_all_physical_map", "compute_logical_to_rank_dispatch_physical_map"):
        HookRegistry.register("sglang.srt.eplb.expert_location." + name, _around(globals()[name]), HookType.AROUND)
    HookRegistry.register("sglang.srt.server_args.ServerArgs.add_cli_args", after_cli, HookType.AFTER)
    HookRegistry.register("sglang.srt.server_args.ServerArgs.from_cli_args", before_seed, HookType.BEFORE)
    HookRegistry.register("sglang.srt.arg_groups.parallel_hook.handle_eplb_and_dispatch", around_validation, HookType.AROUND)
