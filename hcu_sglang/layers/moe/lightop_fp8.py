"""Two-stage LightOp FP8-W8A8 MoE wrapper."""

from __future__ import annotations

from typing import Optional

import torch


def fused_moe_fp8_w8a8(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int = -1,
    origin_w1_shape: tuple | torch.Size | None = None,
    origin_w2_shape: tuple | torch.Size | None = None,
    inplace: bool = False,
    routed_scaling_factor: Optional[float] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if origin_w1_shape is None or origin_w2_shape is None:
        raise ValueError("Original FP8 MoE weight shapes are required")
    if (
        not hidden_states.is_contiguous()
        or not w1.is_contiguous()
        or not w2.is_contiguous()
    ):
        raise ValueError("LightOp FP8 MoE inputs and weights must be contiguous")

    from lightop.activation import fuse_silu_mul_fp8_quant
    from lightop.moe import (
        get_moe_cuda_marlin_config,
        moe_align_block_size,
        moe_gemm_marlin_w8a8_fp8,
        moe_sum,
    )
    from lightop.quant import per_token_quant_fp8

    hidden_states_fp8, hidden_states_scale_fp8 = per_token_quant_fp8(hidden_states)
    num_experts = w1.shape[0]
    num_tokens, hidden_size = hidden_states.shape
    n1, k1 = origin_w1_shape[1:]
    n2, k2 = origin_w2_shape[1:]
    top_k = topk_ids.shape[1]
    device_name = torch.cuda.get_device_name().replace(" ", "_")
    num_cus = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    cuda_config1, cuda_config2, _ = get_moe_cuda_marlin_config(
        num_experts,
        num_tokens,
        n1,
        k1,
        n2,
        k2,
        top_k,
        device_name,
        num_cus,
        hidden_states.dtype,
    )
    if "BLOCK_SIZE_M" not in cuda_config1:
        raise RuntimeError("No LightOp FP8-W8A8 MoE implementation is available")

    if global_num_experts == -1:
        global_num_experts = num_experts
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids,
        cuda_config1["BLOCK_SIZE_M"],
        global_num_experts,
    )
    workspace = torch.empty(
        (num_tokens * top_k * max(n1, hidden_size),),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = workspace[: num_tokens * top_k * n1].view(-1, n1)
    intermediate_cache3 = workspace[: num_tokens * top_k * hidden_size].view(
        -1, hidden_size
    )
    intermediate_cache1 = moe_gemm_marlin_w8a8_fp8(
        hidden_states_fp8,
        w1,
        intermediate_cache1,
        hidden_states_scale_fp8,
        w1_scale,
        None,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k,
        cuda_config1,
    )
    fp8_cache2, fp8_cache2_scale = fuse_silu_mul_fp8_quant(
        intermediate_cache1,
        fp8type=0,
    )
    intermediate_cache3 = moe_gemm_marlin_w8a8_fp8(
        fp8_cache2,
        w2,
        intermediate_cache3,
        fp8_cache2_scale,
        w2_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        1,
        cuda_config2,
    ).view(-1, top_k, hidden_size)
    output = hidden_states if inplace else torch.empty_like(hidden_states)
    moe_sum(
        intermediate_cache3,
        output,
        bias=bias,
        expert_mask=None,
        num_local_tokens=None,
        factor=routed_scaling_factor or 1.0,
        expect_m=-1,
    )
    return output
