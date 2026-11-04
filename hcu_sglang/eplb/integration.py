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
import logging
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_location import get_global_expert_location_metadata
from sglang.srt.runtime_context import get_exec
from hcu_sglang.eplb.lplb_solver import LoadAwareReplicaSolver, LPLBSolver, assert_lplb_supported_model, clear_global_lplb_solvers, set_global_lplb_solver, get_global_lplb_solver
from sglang.srt.eplb import eplb_manager as M
from sglang.srt.arg_groups import parallel_hook as P
from hcu_sglang.utils.detection import is_hcu

logger = logging.getLogger(__name__)


def init_lplb_solvers(*, model_config: ModelConfig) -> None:
    """Initialize per-layer LPLB solvers from current expert location metadata."""
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.runtime_context import get_exec

    dispatch_algorithm = get_exec().moe.ep_dispatch_algorithm
    # Gate: refuse the Hopper LP solver for non-DeepSeek MoE families whose
    # empty-token paths don't participate in its EP all-reduce. The ROCm
    # load-aware path explicitly wires empty MiniMax ranks through solve().
    architectures = getattr(model_config.hf_config, "architectures", None)
    if dispatch_algorithm == "lp" and architectures:
        assert_lplb_supported_model(architectures[0])

    metadata = get_global_expert_location_metadata()
    if metadata is None:
        return
    clear_global_lplb_solvers()
    ep_group = get_parallel().moe_ep_group
    static_mass = None
    static_max_copies = None
    static_path = envs.SGLANG_EXPERIMENTAL_LPLB_STATIC_PROBS.get()
    if static_path:
        import torch

        if not torch.version.hip:
            raise NotImplementedError("Precomputed compact LP dispatch requires HIP")
        payload = torch.load(static_path, weights_only=True, map_location="cpu")
        if not torch.equal(
            payload["physical_to_logical_map"],
            metadata.physical_to_logical_map_cpu,
        ):
            raise ValueError(
                "Static LP artifact does not match the loaded expert layout"
            )
        if payload["physical_mass"].shape != metadata.physical_to_logical_map_cpu.shape:
            raise ValueError(
                "Static LP mass must match the physical expert layout shape"
            )
        static_mass = payload["physical_mass"].to(
            device=metadata.physical_to_logical_map.device, dtype=torch.float32
        )
        if not bool(torch.isfinite(static_mass).all() and (static_mass >= 0).all()):
            raise ValueError("Static LP mass must be finite and nonnegative")
        static_max_copies = int(
            metadata.logical_to_all_physical_map_num_valid.max().item()
        )
        logger.info(
            "Using precomputed LP replica probabilities from %s; replica table %d -> %d columns",
            static_path,
            metadata.logical_to_all_physical_map.shape[-1],
            static_max_copies,
        )
    for lid in range(metadata.num_layers):
        static_probabilities = None
        if static_mass is not None:
            replica_map = metadata.logical_to_all_physical_map[lid]
            static_probabilities = static_mass[lid][replica_map.clamp_min(0).long()]
            static_probabilities = static_probabilities * (replica_map >= 0)
        solver_cls = (
            LoadAwareReplicaSolver if dispatch_algorithm == "load_aware" else LPLBSolver
        )
        solver_kwargs = dict(
            phy2log=metadata.physical_to_logical_map[lid],
            log2phy=metadata.logical_to_all_physical_map[lid],
            num_gpus=metadata.ep_size,
            ep_group=ep_group,
            logical_to_all_physical_map_num_valid=(
                metadata.logical_to_all_physical_map_num_valid[lid]
            ),
        )
        if dispatch_algorithm == "load_aware":
            if static_mass is not None:
                raise ValueError("Static LP probabilities cannot configure load-aware dispatch")
        else:
            solver_kwargs.update(
                static_probabilities=static_probabilities,
                static_max_copies=static_max_copies,
            )
        solver = solver_cls(**solver_kwargs)
        set_global_lplb_solver(lid, solver)
    logger.info(
        "Initialized %s replica-dispatch solvers for %d layers",
        dispatch_algorithm,
        metadata.num_layers,
    )
    if dispatch_algorithm == "load_aware":
        counts = sorted({get_global_lplb_solver(lid).num_replicated for lid in range(metadata.num_layers)})
        logger.info("Load-aware replicated logical experts per layer: %s", counts)


def maybe_init_lplb_solvers(self):
    if (
        get_exec().moe.ep_dispatch_algorithm
        in (
            "lp",
            "load_aware",
        )
        and not self.is_draft_worker
    ):
        init_lplb_solvers(model_config=self.model_config)



def update_expert_location_with_recovery(
    *,
    expert_location_updater: ExpertLocationUpdater,
    model: nn.Module,
    new_expert_location_metadata: ExpertLocationMetadata,
    update_layer_ids: List[int],
    tp_rank: int,
    use_flat_topology: bool = False,
    expert_backup_client,
    update_weights_from_disk_callable,
    ep_dispatch_algorithm: str,
    init_lplb_solvers_callable,
):
    p2p_missing_logical_experts = expert_location_updater.update(
        model.routed_experts_weights_of_layer,
        new_expert_location_metadata,
        update_layer_ids=update_layer_ids,
        nnodes=M.get_parallel().nnodes,
        rank=tp_rank,
        use_flat_topology=use_flat_topology,
    )

    if len(p2p_missing_logical_experts) > 0:
        # Load the missing expert weights from disk
        if callable(getattr(model, "generate_weight_name_filter", None)):
            # Filter and load only missing expert weights
            weight_name_filter = model.generate_weight_name_filter(
                p2p_missing_logical_experts
            )
        else:
            # Do a full reload from disk/DRAM
            M.logger.info(
                "[Elastic EP] Model does not implement generate_weight_name_filter. "
                "Performing full weight reload."
            )
            weight_name_filter = None

        if expert_backup_client is not None and expert_backup_client.use_backup:
            # Load the missing weights from the DRAM backup
            expert_backup_client.update_weights(weight_name_filter)
        else:
            # Load the missing weights from disk
            update_weights_from_disk_callable(
                M.get_model().model_path,
                M.get_model().load_format,
                weight_name_filter=weight_name_filter,
            )

    # Re-init LPLB solvers after expert location update
    if ep_dispatch_algorithm in ("lp", "load_aware"):
        init_lplb_solvers_callable()


def _around(implementation):
    def hook(original, *args, **kwargs):
        if not is_hcu():
            return original(*args, **kwargs)
        return implementation(*args, **kwargs)
    return hook


def after_cli(result, parser):
    if is_hcu():
        action = parser._option_string_actions.get("--ep-dispatch-algorithm")
        if action is not None and action.choices is not None and "load_aware" not in action.choices:
            action.choices = list(action.choices) + ["load_aware"]
    return result


def around_dispatch_validation(original, server_args):
    result = original(server_args)
    if not is_hcu():
        return result
    cfg = P.resolving_view(server_args)
    if cfg.ep_dispatch_algorithm == "load_aware":
        if P.resolved_view(server_args).moe_a2a_backend == "none":
            raise ValueError("load_aware replica dispatch requires an a2a backend")
        if not P.get_platform().is_hip:
            raise ValueError("load_aware replica dispatch requires ROCm")
        if cfg.ep_num_redundant_experts <= 0:
            raise ValueError("load_aware replica dispatch requires redundant experts")
    return result


def around_info_init(original, cls, layer_id):
    if not is_hcu():
        return original(cls, layer_id)
    from hcu_sglang.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
    return ExpertLocationDispatchInfo.init_new(layer_id)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from hcu_sglang.eplb import expert_location_dispatch as D
    for target, implementation in [
        ("sglang.srt.model_executor.model_runner_components.moe_ep_setup.init_lplb_solvers", init_lplb_solvers),
        ("sglang.srt.model_executor.model_runner.ModelRunner.maybe_init_lplb_solvers", maybe_init_lplb_solvers),
        ("sglang.srt.eplb.expert_location_dispatch.topk_ids_logical_to_physical", D.topk_ids_logical_to_physical),
        ("sglang.srt.eplb.eplb_manager.update_expert_location_with_recovery", update_expert_location_with_recovery),
    ]:
        HookRegistry.register(target, _around(implementation), HookType.AROUND)
    HookRegistry.register("sglang.srt.eplb.expert_location_dispatch.ExpertLocationDispatchInfo.init_new", around_info_init, HookType.AROUND)
    HookRegistry.register("sglang.srt.server_args.ServerArgs.add_cli_args", after_cli, HookType.AFTER)
    HookRegistry.register("sglang.srt.arg_groups.parallel_hook.handle_eplb_and_dispatch", around_dispatch_validation, HookType.AROUND)
