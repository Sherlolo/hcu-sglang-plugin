"""AITER W4A8 MoE extension for the official SGLang runner."""

from __future__ import annotations

import os
from typing import Any, Callable

import torch
from torch.nn.parameter import Parameter


_MARKER = "_sglang_dcu_w4a8_int8"
_QUANT_MODE = "_sglang_dcu_aiter_quant_mode"
_W4A8_INT8 = "w4a8_int8"
_W8A8_INT8 = "w8a8_int8"
_W8A8_FP8 = "w8a8_fp8"


def get_aiter_w4a8_int8_quant_info(layer: torch.nn.Module):
    from sglang.srt.layers.moe.moe_runner.aiter import (
        AiterMoeQuantInfo,
        AiterQuantType,
    )

    dispatcher = getattr(layer, "dispatcher", None)
    expert_map = (
        getattr(dispatcher, "local_expert_mapping", None)
        if getattr(dispatcher, "expert_mask_gpu", None) is not None
        else None
    )
    quant_info = AiterMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        quant_type=AiterQuantType.NONE,
        w13_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        a13_scale=layer.w13_input_scale,
        a2_scale=layer.w2_input_scale,
    )
    setattr(quant_info, _MARKER, True)
    setattr(quant_info, _QUANT_MODE, _W4A8_INT8)
    setattr(quant_info, "global_num_experts", getattr(layer, "num_experts", None))
    setattr(quant_info, "expert_map", expert_map)
    setattr(
        quant_info,
        "original_w13_shape",
        getattr(
            layer, "_aiter_w4a8_original_w13_shape", tuple(layer.w13_weight.shape)
        ),
    )
    setattr(
        quant_info,
        "original_w2_shape",
        getattr(layer, "_aiter_w4a8_original_w2_shape", tuple(layer.w2_weight.shape)),
    )
    setattr(
        quant_info,
        "moe_config_cache",
        getattr(layer, "_aiter_w4a8_moe_config_cache", None),
    )
    setattr(quant_info, "layer", layer)
    return quant_info


def _get_expert_map(layer):
    dispatcher = getattr(layer, "dispatcher", None)
    if getattr(dispatcher, "expert_mask_gpu", None) is None:
        return None
    return getattr(dispatcher, "local_expert_mapping", None)


def _get_w8a8_quant_info(layer, mode):
    from sglang.srt.layers.moe.moe_runner.aiter import (
        AiterMoeQuantInfo,
        AiterQuantType,
    )

    prefix = "_aiter_w8a8_fp8" if mode == _W8A8_FP8 else "_aiter_w8a8_int8"
    original_w13_shape = f"{prefix}_original_w13_shape"
    original_w2_shape = f"{prefix}_original_w2_shape"
    config_cache = f"{prefix}_moe_config_cache"
    weight_layout = f"{prefix}_moe_c_weight_layout"
    if not hasattr(layer, original_w13_shape):
        setattr(layer, original_w13_shape, tuple(layer.w13_weight.shape))
    if not hasattr(layer, original_w2_shape):
        setattr(layer, original_w2_shape, tuple(layer.w2_weight.shape))
    if not hasattr(layer, config_cache):
        setattr(layer, config_cache, {})
    if not hasattr(layer, weight_layout):
        setattr(layer, weight_layout, False)

    quant_info = AiterMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        quant_type=AiterQuantType.NONE,
        w13_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        a13_scale=layer.w13_input_scale,
        a2_scale=layer.w2_input_scale,
    )
    setattr(quant_info, _QUANT_MODE, mode)
    setattr(quant_info, "global_num_experts", getattr(layer, "num_experts", None))
    setattr(quant_info, "expert_map", _get_expert_map(layer))
    setattr(quant_info, "moe_config_cache", getattr(layer, config_cache))
    setattr(quant_info, "moe_c_weight_layout", getattr(layer, weight_layout))
    setattr(quant_info, "original_w13_shape", getattr(layer, original_w13_shape))
    setattr(quant_info, "original_w2_shape", getattr(layer, original_w2_shape))
    setattr(quant_info, "layer", layer)
    return quant_info


def get_aiter_w8a8_int8_quant_info(layer):
    return _get_w8a8_quant_info(layer, _W8A8_INT8)


def get_aiter_w8a8_fp8_quant_info(layer):
    return _get_w8a8_quant_info(layer, _W8A8_FP8)


def _run_w4a8(runner_core, runner_input, quant_info):
    from aiter.ops.triton.fused_moe import fused_experts_impl
    from sglang.srt.layers.moe.moe_runner.aiter import AiterRunnerOutput

    runner_config = runner_core.config
    if runner_config.no_combine:
        raise RuntimeError("AITER W4A8 MoE does not support no_combine=True.")
    if runner_config.apply_router_weight_on_input:
        raise RuntimeError(
            "AITER W4A8 MoE does not support apply_router_weight_on_input=True."
        )
    if runner_input.hidden_states.shape[0] == 0:
        return AiterRunnerOutput(hidden_states=runner_input.hidden_states)

    _get_w4a8_original_dims(runner_input.hidden_states, quant_info)

    global_num_experts = getattr(quant_info, "global_num_experts", None)
    if global_num_experts is None:
        global_num_experts = quant_info.w13_weight.shape[0]
    routed_scaling_factor = runner_config.routed_scaling_factor
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    output = fused_experts_impl(
        hidden_states=runner_input.hidden_states,
        w1=quant_info.w13_weight,
        w2=quant_info.w2_weight,
        topk_weights=runner_input.topk_weights.to(torch.float32),
        topk_ids=runner_input.topk_ids.to(torch.int32),
        output_dtype=runner_input.hidden_states.dtype,
        inplace=False,
        activation=str(runner_config.activation),
        is_gated=runner_config.is_gated,
        use_int4_w4a8=True,
        per_channel_quant=True,
        global_num_experts=global_num_experts,
        expert_map=getattr(quant_info, "expert_map", None),
        w1_scale=quant_info.w13_scale,
        w2_scale=quant_info.w2_scale,
        a1_scale=quant_info.a13_scale,
        a2_scale=quant_info.a2_scale,
        routed_scaling_factor=float(routed_scaling_factor),
        gemm1_alpha=runner_config.gemm1_alpha,
        gemm1_limit=runner_config.gemm1_clamp_limit,
    )
    return AiterRunnerOutput(hidden_states=output)


def _get_w4a8_original_dims(hidden_states, quant_info):
    if hidden_states.dim() != 2:
        raise RuntimeError(
            "AITER W4A8 MoE expects 2D hidden_states, got "
            f"shape={tuple(hidden_states.shape)}."
        )
    hidden_size = hidden_states.shape[1]
    w13_shape = getattr(quant_info, "original_w13_shape", None)
    w2_shape = getattr(quant_info, "original_w2_shape", None)
    if w13_shape is None or w2_shape is None:
        raise RuntimeError("AITER W4A8 MoE is missing original packed weight shapes.")

    num_experts, intermediate_twice, packed_hidden = w13_shape
    w2_experts, w2_hidden, _ = w2_shape
    if (
        num_experts != w2_experts
        or packed_hidden != hidden_size // 2
        or w2_hidden != hidden_size
    ):
        raise RuntimeError(
            "AITER W4A8 MoE shape mismatch: "
            f"hidden_states={tuple(hidden_states.shape)}, "
            f"w1_original={tuple(w13_shape)}, w2_original={tuple(w2_shape)}, "
            f"w1_current={tuple(quant_info.w13_weight.shape)}, "
            f"w2_current={tuple(quant_info.w2_weight.shape)}."
        )
    return num_experts, intermediate_twice, w2_hidden, hidden_size


def _get_w8a8_original_dims(hidden_states, quant_info):
    if hidden_states.dim() != 2:
        raise RuntimeError(
            "AITER W8A8 MoE expects 2D hidden_states, got "
            f"shape={tuple(hidden_states.shape)}."
        )
    hidden_size = hidden_states.shape[1]
    w13_shape = quant_info.original_w13_shape
    w2_shape = quant_info.original_w2_shape
    num_experts, intermediate_twice, w13_hidden = w13_shape
    w2_experts, w2_hidden, _ = w2_shape
    if (
        num_experts != w2_experts
        or hidden_size != w13_hidden
        or hidden_size != w2_hidden
    ):
        raise RuntimeError(
            "AITER W8A8 MoE shape mismatch: "
            f"hidden_states={tuple(hidden_states.shape)}, "
            f"w1_original={tuple(w13_shape)}, w2_original={tuple(w2_shape)}, "
            f"w1_current={tuple(quant_info.w13_weight.shape)}, "
            f"w2_current={tuple(quant_info.w2_weight.shape)}."
        )
    return num_experts, intermediate_twice, w2_hidden, hidden_size


def _get_w8a8_moe_config(runner_input, quant_info, runner_config):
    from aiter.moe import MoeQuantType, MoeSolutionType, get_aiter_moe_config

    hidden_states = runner_input.hidden_states
    num_experts, intermediate_twice, w2_hidden, hidden_size = (
        _get_w8a8_original_dims(hidden_states, quant_info)
    )
    top_k = runner_input.topk_ids.shape[1]
    activation = str(runner_config.activation)
    cache_key = (hidden_states.shape[0], top_k, hidden_states.dtype, activation)
    cached = quant_info.moe_config_cache.get(cache_key)
    if cached is not None:
        return cached

    quant_type_name = (
        "FP8_W8A8"
        if getattr(quant_info, _QUANT_MODE) == _W8A8_FP8
        else "W8A8"
    )
    quant_type = getattr(MoeQuantType, quant_type_name, None)
    if quant_type is None:
        raise RuntimeError(
            f"The installed aiter package does not expose MoeQuantType.{quant_type_name}."
        )
    config_kwargs = {
        "M": hidden_states.shape[0],
        "E": num_experts,
        "N1": intermediate_twice,
        "N2": w2_hidden,
        "K": hidden_size,
        "top_k": top_k,
        "block_size": 0,
        "dtype": hidden_states.dtype,
        "quant_type": quant_type,
        "activation": activation,
    }
    try:
        status, moe_config = get_aiter_moe_config(**config_kwargs)
    except TypeError:
        config_kwargs.pop("activation")
        status, moe_config = get_aiter_moe_config(**config_kwargs)
    if os.environ.get("PRINT_MOE_ARGS", "0") == "1":
        print(f"AITER W8A8 MoE args: moe_config={moe_config}, {config_kwargs}", flush=True)
    if not status:
        raise RuntimeError(f"AITER W8A8 MoE did not find a valid config: {config_kwargs}.")
    if moe_config.solution_type not in {
        MoeSolutionType.MOE_C,
        MoeSolutionType.ASM,
        MoeSolutionType.TRITON,
        MoeSolutionType.CK,
    }:
        raise RuntimeError(
            f"Unsupported AITER MoE solution_type: {moe_config.solution_type}"
        )
    if moe_config.quant_type != quant_type:
        raise RuntimeError(f"Unexpected AITER MoE quant_type: {moe_config.quant_type}")
    quant_info.moe_config_cache[cache_key] = moe_config
    return moe_config


def _get_w8a8_weights_for_solution(quant_info, moe_config):
    from aiter.moe import MoeSolutionType
    from aiter.ops.shuffle import moe_layout_shuffle_gemm2

    need_shuffle = getattr(
        moe_config,
        "need_shuffle",
        moe_config.solution_type == MoeSolutionType.MOE_C,
    )
    if not need_shuffle:
        if quant_info.moe_c_weight_layout:
            raise RuntimeError(
                "AITER W8A8 weights are already shuffled, but the selected "
                f"solution does not use that layout: {moe_config.solution_type}."
            )
        return quant_info.w13_weight, quant_info.w2_weight
    if quant_info.moe_c_weight_layout:
        return quant_info.w13_weight, quant_info.w2_weight

    with torch.no_grad():
        w13_weight = moe_layout_shuffle_gemm2(quant_info.w13_weight).view(
            *quant_info.w13_weight.shape
        )
        w2_weight = moe_layout_shuffle_gemm2(quant_info.w2_weight).view(
            *quant_info.w2_weight.shape
        )
    layer = quant_info.layer
    if layer is not None:
        layer.w13_weight = Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = Parameter(w2_weight, requires_grad=False)
        prefix = (
            "_aiter_w8a8_fp8"
            if getattr(quant_info, _QUANT_MODE) == _W8A8_FP8
            else "_aiter_w8a8_int8"
        )
        setattr(layer, f"{prefix}_moe_c_weight_layout", True)
        w13_weight = layer.w13_weight
        w2_weight = layer.w2_weight
    quant_info.w13_weight = w13_weight
    quant_info.w2_weight = w2_weight
    quant_info.moe_c_weight_layout = True
    return w13_weight, w2_weight


def _run_w8a8(runner_core, runner_input, quant_info):
    from aiter.moe import aiter_moe
    from sglang.srt.layers.moe.moe_runner.aiter import AiterRunnerOutput

    runner_config = runner_core.config
    if runner_config.no_combine:
        raise RuntimeError("AITER W8A8 MoE does not support no_combine=True.")
    if runner_config.apply_router_weight_on_input:
        raise RuntimeError(
            "AITER W8A8 MoE does not support apply_router_weight_on_input=True."
        )
    if runner_input.hidden_states.shape[0] == 0:
        return AiterRunnerOutput(hidden_states=runner_input.hidden_states)

    moe_config = _get_w8a8_moe_config(runner_input, quant_info, runner_config)
    w13_weight, w2_weight = _get_w8a8_weights_for_solution(quant_info, moe_config)
    global_num_experts = (
        quant_info.global_num_experts
        if quant_info.expert_map is not None and quant_info.global_num_experts is not None
        else w13_weight.shape[0]
    )
    routed_scaling_factor = runner_config.routed_scaling_factor or 1.0
    output = aiter_moe(
        hidden_states=runner_input.hidden_states,
        w1=w13_weight,
        w2=w2_weight,
        topk_weights=runner_input.topk_weights.to(torch.float32),
        topk_ids=runner_input.topk_ids.to(torch.int32),
        moe_config=moe_config,
        inplace=runner_config.inplace,
        activation=str(runner_config.activation),
        w1_scale=quant_info.w13_scale,
        w2_scale=quant_info.w2_scale,
        w1_zp=None,
        w2_zp=None,
        a1_scale=quant_info.a13_scale,
        a2_scale=quant_info.a2_scale,
        block_shape=None,
        global_num_experts=global_num_experts,
        expert_map=quant_info.expert_map,
        routed_scaling_factor=float(routed_scaling_factor),
        output_dtype=runner_input.hidden_states.dtype,
        gemm1_alpha=runner_config.gemm1_alpha,
        gemm1_limit=runner_config.gemm1_clamp_limit,
    )
    return AiterRunnerOutput(hidden_states=output)


def around_aiter_run(
    original: Callable[..., Any],
    runner_core,
    runner_input,
    quant_info,
    running_state,
    hooks=None,
):
    quant_mode = getattr(quant_info, _QUANT_MODE, None)
    if quant_mode is None and getattr(quant_info, _MARKER, False):
        quant_mode = _W4A8_INT8
    if quant_mode is not None:
        if hooks is not None:
            raise AssertionError("AITER HCU quantized MoE does not support LoRA hooks.")
        if quant_mode == _W4A8_INT8:
            return _run_w4a8(runner_core, runner_input, quant_info)
        if quant_mode in {_W8A8_INT8, _W8A8_FP8}:
            return _run_w8a8(runner_core, runner_input, quant_info)
        raise RuntimeError(f"Unknown plugin AITER quantization mode: {quant_mode}")
    return original(runner_core, runner_input, quant_info, running_state, hooks)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.moe.moe_runner.aiter.AiterRunnerCore.run",
        around_aiter_run,
        HookType.AROUND,
    )
