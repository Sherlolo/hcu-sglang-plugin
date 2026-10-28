"""Fork-compatible HCU AITER W16A16 MoE routing."""

from __future__ import annotations

from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_dcu


def _enabled() -> bool:
    from sglang.srt.utils import get_bool_env_var

    return (
        is_dcu()
        and get_bool_env_var("SGLANG_ROCM_USE_AITER_MOE")
        and not get_bool_env_var("SGLANG_USE_MARLIN_W16A16_MOE")
    )


def _weight_tensor(weight: Any) -> torch.Tensor:
    return weight[0] if isinstance(weight, tuple) else weight


def around_forward_cuda(
    original: Callable[..., Any], method: Any, layer: Any, dispatch_output: Any
) -> Any:
    if not _enabled():
        return original(method, layer, dispatch_output)

    from hcu_sglang.compat.aiter import install_aiter_compat

    install_aiter_compat()
    from aiter.moe import (
        MoeQuantType,
        MoeSolutionType,
        aiter_moe,
        get_aiter_moe_config,
    )
    from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    w13_weight = _weight_tensor(layer.w13_weight)
    w2_weight = _weight_tensor(layer.w2_weight)
    config = method.moe_runner_config

    if config.apply_router_weight_on_input:
        if topk_weights.ndim != 2 or topk_weights.shape[1] != 1:
            raise RuntimeError(
                "HCU AITER W16A16 apply_router_weight_on_input requires topk=1"
            )
        hidden_states = hidden_states * topk_weights.to(hidden_states.dtype)
        topk_weights = torch.ones_like(topk_weights, dtype=torch.float32)

    if hidden_states.ndim != 2:
        raise RuntimeError(
            "HCU AITER W16A16 MoE expects 2D hidden_states, "
            f"got shape={tuple(hidden_states.shape)}"
        )
    if w13_weight.ndim != 3 or w2_weight.ndim != 3:
        raise RuntimeError(
            "HCU AITER W16A16 MoE expects 3D expert weights, "
            f"got w13={tuple(w13_weight.shape)}, w2={tuple(w2_weight.shape)}"
        )

    num_tokens, hidden_size = hidden_states.shape
    num_experts = config.num_experts
    if w13_weight.shape[0] != num_experts or w2_weight.shape[0] != num_experts:
        raise RuntimeError(
            "HCU AITER W16A16 expert count mismatch: "
            f"expected={num_experts}, w13={tuple(w13_weight.shape)}, "
            f"w2={tuple(w2_weight.shape)}"
        )
    if w13_weight.shape[2] != hidden_size or w2_weight.shape[1] != hidden_size:
        raise RuntimeError(
            "HCU AITER W16A16 hidden dimension mismatch: "
            f"hidden_size={hidden_size}, w13={tuple(w13_weight.shape)}, "
            f"w2={tuple(w2_weight.shape)}"
        )

    status, moe_config = get_aiter_moe_config(
        M=num_tokens,
        E=num_experts,
        N1=w13_weight.shape[1],
        N2=w2_weight.shape[2],
        K=hidden_size,
        top_k=topk_ids.shape[1],
        block_size=0,
        dtype=hidden_states.dtype,
        quant_type=MoeQuantType.W16A16,
    )
    if not status:
        raise RuntimeError(
            "HCU AITER W16A16 did not find a suitable backend for "
            f"M={num_tokens}, E={num_experts}, K={hidden_size}, "
            f"N1={w13_weight.shape[1]}, N2={w2_weight.shape[2]}, "
            f"topk={topk_ids.shape[1]}"
        )
    if moe_config.solution_type not in {
        MoeSolutionType.ASM,
        MoeSolutionType.TRITON,
        MoeSolutionType.CK,
    }:
        raise RuntimeError(
            f"Unsupported HCU AITER W16A16 solution: {moe_config.solution_type}"
        )
    if moe_config.quant_type != MoeQuantType.W16A16:
        raise RuntimeError(f"Unexpected HCU AITER quant type: {moe_config.quant_type}")

    output = aiter_moe(
        hidden_states=hidden_states,
        w1=w13_weight,
        w2=w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        moe_config=moe_config,
        inplace=True,
        w1_scale=None,
        w2_scale=None,
        activation="silu" if config.activation == "silu" else "gelu",
        block_shape=None,
        global_num_experts=num_experts,
        routed_scaling_factor=config.routed_scaling_factor or 1.0,
    )
    return StandardCombineInput(hidden_states=output)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.quantization.unquant."
        "UnquantizedFusedMoEMethod.forward_cuda",
        around_forward_cuda,
        HookType.AROUND,
    )
