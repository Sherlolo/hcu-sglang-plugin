"""LightOp expert computation for DeepEP normal dispatch."""

from __future__ import annotations

from typing import Any

import torch


def _copy_counts_to_device(counts: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(counts, dtype=torch.int32, device=device)


def _scatter(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    counts: torch.Tensor,
    scattered: torch.Tensor,
    scattered_scale: torch.Tensor,
    all_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from lightop import moe as lightop_moe

    if any(int(count) % 256 for count in counts.cpu().tolist()):
        raise RuntimeError(
            "LightOp DeepEP normal mode requires 256-aligned expert counts"
        )

    output_index = torch.full(
        topk_ids.shape, -1, device=topk_ids.device, dtype=torch.int32
    )
    m_indices = torch.full(
        (all_tokens,), -1, device=hidden_states.device, dtype=torch.int32
    )
    lightop_moe.ep_scatter(
        hidden_states,
        hidden_states_scale,
        topk_ids,
        None,
        counts,
        scattered,
        scattered_scale,
        m_indices,
        output_index,
        counts.numel(),
        256,
    )
    return m_indices, output_index


def _gather(
    expert_output: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output_index: torch.Tensor,
    output: torch.Tensor,
) -> None:
    from lightop import moe as lightop_moe

    lightop_moe.ep_gather(
        expert_output,
        topk_ids,
        topk_weights,
        output_index,
        None,
        output,
    )


def _run_int8(
    dispatch_output: Any, quant_info: Any, config: Any, all_tokens: int
) -> torch.Tensor:
    import lightop
    from lightop.activation import fuse_silu_mul_quant

    hidden_states = dispatch_output.hidden_states
    hidden_states_scale = dispatch_output.hidden_states_scale
    if hidden_states_scale is None:
        raise RuntimeError("LightOp INT8 DeepEP requires dispatched activation scales")

    w13_weight = quant_info.deepep_w13_weight
    w2_weight = quant_info.deepep_w2_weight
    _, hidden_size = hidden_states.shape
    intermediate_twice = w13_weight.shape[1]
    counts = _copy_counts_to_device(
        dispatch_output.num_recv_tokens_per_expert, hidden_states.device
    )
    scattered = torch.empty(
        (all_tokens, hidden_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    scattered_scale = torch.empty(
        (all_tokens, hidden_states_scale.shape[-1]),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    m_indices, output_index = _scatter(
        hidden_states,
        hidden_states_scale,
        dispatch_output.topk_ids,
        counts,
        scattered,
        scattered_scale,
        all_tokens,
    )

    gate_up = torch.empty(
        (all_tokens, intermediate_twice),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    lightop.m_grouped_w8a8_gemm_nt_contig_asm(
        (scattered, scattered_scale),
        (w13_weight, quant_info.w13_scale),
        gate_up,
        m_indices,
    )
    activated, activated_scale = fuse_silu_mul_quant(gate_up)
    expert_output = torch.empty(
        (all_tokens, hidden_size),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    lightop.m_grouped_w8a8_gemm_nt_contig_asm(
        (activated, activated_scale),
        (w2_weight, quant_info.w2_scale),
        expert_output,
        m_indices,
    )
    output = torch.zeros_like(hidden_states, dtype=torch.bfloat16)
    _gather(
        expert_output,
        dispatch_output.topk_ids,
        dispatch_output.topk_weights,
        output_index,
        output,
    )
    return output


def _run_fp8(
    dispatch_output: Any, quant_info: Any, config: Any, all_tokens: int
) -> torch.Tensor:
    del config
    from deepgemm import m_grouped_fp8_gemm_nt_contiguous
    from lightop.activation import fuse_silu_mul_fp8_quant

    hidden_states = dispatch_output.hidden_states
    hidden_states_scale = dispatch_output.hidden_states_scale
    if hidden_states_scale is None:
        raise RuntimeError("LightOp FP8 DeepEP requires dispatched activation scales")

    w13_weight = quant_info.deepep_w13_weight
    w2_weight = quant_info.deepep_w2_weight
    _, hidden_size = hidden_states.shape
    intermediate_twice = w13_weight.shape[1]
    counts = _copy_counts_to_device(
        dispatch_output.num_recv_tokens_per_expert, hidden_states.device
    )
    scattered = torch.empty(
        (all_tokens, hidden_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    scattered_scale = torch.empty(
        (all_tokens, hidden_states_scale.shape[-1]),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    m_indices, output_index = _scatter(
        hidden_states,
        hidden_states_scale,
        dispatch_output.topk_ids,
        counts,
        scattered,
        scattered_scale,
        all_tokens,
    )

    gate_up = torch.empty(
        (all_tokens, intermediate_twice),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    m_grouped_fp8_gemm_nt_contiguous(
        (scattered, scattered_scale),
        (w13_weight, quant_info.w13_scale),
        gate_up,
        m_indices,
    )
    activated, activated_scale = fuse_silu_mul_fp8_quant(gate_up, fp8type=0)
    expert_output = torch.empty(
        (all_tokens, hidden_size),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    m_grouped_fp8_gemm_nt_contiguous(
        (activated, activated_scale),
        (w2_weight, quant_info.w2_scale),
        expert_output,
        m_indices,
    )
    output = torch.zeros_like(hidden_states, dtype=torch.bfloat16)
    _gather(
        expert_output,
        dispatch_output.topk_ids,
        dispatch_output.topk_weights,
        output_index,
        output,
    )
    return output


def _run_int8_ll(dispatch_output: Any, quant_info: Any) -> torch.Tensor:
    import lightop
    from lightop.activation import fuse_silu_mul_quant_ep

    hidden_states = dispatch_output.hidden_states
    hidden_states_scale = dispatch_output.hidden_states_scale
    if hidden_states_scale is None:
        raise RuntimeError("LightOp INT8 DeepEP-LL requires activation scales")
    expected_m = min(hidden_states.shape[1], dispatch_output.expected_m)
    w13_weight = quant_info.deepep_w13_weight
    w2_weight = quant_info.deepep_w2_weight
    gate_up = torch.empty(
        (*hidden_states.shape[:2], w13_weight.shape[1]),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    lightop.m_grouped_w8a8_gemm_nt_masked(
        (hidden_states, hidden_states_scale),
        (w13_weight, quant_info.w13_scale),
        gate_up,
        dispatch_output.masked_m,
        expected_m,
    )
    activated, activated_scale = fuse_silu_mul_quant_ep(
        gate_up,
        tokens_per_expert=dispatch_output.masked_m,
        expect_m=expected_m,
    )
    output = torch.empty(
        (*hidden_states.shape[:2], w2_weight.shape[1]),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    lightop.m_grouped_w8a8_gemm_nt_masked(
        (activated, activated_scale),
        (w2_weight, quant_info.w2_scale),
        output,
        dispatch_output.masked_m,
        expected_m,
    )
    return output


def _run_fp8_ll(dispatch_output: Any, quant_info: Any) -> torch.Tensor:
    import lightop
    from lightop.activation import fuse_silu_mul_fp8_quant_ep

    hidden_states = dispatch_output.hidden_states
    hidden_states_scale = dispatch_output.hidden_states_scale
    if hidden_states_scale is None:
        raise RuntimeError("LightOp FP8 DeepEP-LL requires activation scales")
    expected_m = min(hidden_states.shape[1], dispatch_output.expected_m)
    w13_weight = quant_info.deepep_w13_weight
    w2_weight = quant_info.deepep_w2_weight
    gate_up = torch.empty(
        (*hidden_states.shape[:2], w13_weight.shape[1]),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    lightop.m_grouped_fp8_gemm_nt_masked(
        (hidden_states, hidden_states_scale),
        (w13_weight, quant_info.w13_scale),
        gate_up,
        dispatch_output.masked_m,
        expected_m,
    )
    activated, activated_scale = fuse_silu_mul_fp8_quant_ep(
        gate_up,
        fp8type=0,
        tokens_per_expert=dispatch_output.masked_m,
        expect_m=expected_m,
    )
    output = torch.empty(
        (*hidden_states.shape[:2], w2_weight.shape[1]),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    lightop.m_grouped_fp8_gemm_nt_masked(
        (activated, activated_scale),
        (w2_weight, quant_info.w2_scale),
        output,
        dispatch_output.masked_m,
        expected_m,
    )
    return output


def fused_deepep_lightop(dispatch_output: Any, quant_info: Any, config: Any) -> Any:
    dispatch_format = dispatch_output.format.value
    if dispatch_format not in {"deepep_normal", "deepep_ll"}:
        raise NotImplementedError(f"Unsupported LightOp dispatch: {dispatch_format}")
    if quant_info.deepep_w13_weight is None or quant_info.deepep_w2_weight is None:
        raise RuntimeError("LightOp DeepEP original expert weights were not preserved")

    if dispatch_format == "deepep_normal":
        counts = dispatch_output.num_recv_tokens_per_expert
        if counts is None or sum(counts) <= 0:
            output = dispatch_output.hidden_states.bfloat16()
        elif quant_info.use_fp8_w8a8:
            output = _run_fp8(dispatch_output, quant_info, config, sum(counts))
        elif quant_info.use_int8_w8a8:
            output = _run_int8(dispatch_output, quant_info, config, sum(counts))
        else:
            raise NotImplementedError("Unsupported LightOp DeepEP quantization mode")
    else:
        if quant_info.use_fp8_w8a8:
            output = _run_fp8_ll(dispatch_output, quant_info)
        elif quant_info.use_int8_w8a8:
            output = _run_int8_ll(dispatch_output, quant_info)
        else:
            raise NotImplementedError("Unsupported LightOp DeepEP-LL quantization mode")

    from hcu_sglang.compat.aiter import install_aiter_compat

    install_aiter_compat()
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPNormalCombineInput,
    )

    combine_cls = (
        DeepEPNormalCombineInput
        if dispatch_format == "deepep_normal"
        else DeepEPLLCombineInput
    )
    return combine_cls(
        hidden_states=output,
        topk_ids=dispatch_output.topk_ids,
        topk_weights=dispatch_output.topk_weights,
    )


def register_fused_func() -> None:
    from sglang.srt.layers.moe.moe_runner.base import FusedOpPool

    key = ("deepep", "lightop")
    if key not in FusedOpPool._fused_funcs:
        FusedOpPool.register_fused_func(*key, fused_deepep_lightop)
