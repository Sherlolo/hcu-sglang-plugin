"""LightOp W8A8 INT8 Marlin custom-op bridge."""

from __future__ import annotations

from typing import List, Optional

import torch

from sglang.srt.utils import direct_register_custom_op


def w8a8_2_marlin_weight(weight: torch.Tensor) -> torch.Tensor:
    weight = weight.T
    size_k, size_n = weight.shape
    if size_k % 64:
        raise ValueError("W8A8 FP8 Marlin K dimension must be divisible by 64")
    return weight.reshape(size_k // 64, 64, size_n).transpose(1, 2).reshape(
        size_k // 64, size_n * 64
    )


def fused_experts_impl_int8_marlin_wrapper(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    use_nn_moe: Optional[bool] = False,
    routed_scaling_factor: Optional[float] = 1.0,
    shared_output: Optional[torch.Tensor] = None,
    i_q: Optional[torch.Tensor] = None,
    i_s: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from lightop.moe import fused_experts_impl_int8_marlin

    return fused_experts_impl_int8_marlin(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=inplace,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        w1_zp=w1_zp,
        w2_zp=w2_zp,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        use_nn_moe=use_nn_moe,
        routed_scaling_factor=routed_scaling_factor,
        shared_output=shared_output,
        i_q=i_q,
        i_s=i_s,
    )


def fused_experts_impl_int8_marlin_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[List[int]] = None,
    use_nn_moe: Optional[bool] = False,
    routed_scaling_factor: Optional[float] = 1.0,
    shared_output: Optional[torch.Tensor] = None,
    i_q: Optional[torch.Tensor] = None,
    i_s: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def get_w8a8_int8_marlin_weights(
    weight: torch.Tensor, k_tile: int = 64
) -> torch.Tensor:
    weight = weight.T
    size_k, size_n = weight.shape
    if size_k % k_tile:
        raise ValueError("W8A8 INT8 Marlin K dimension must be divisible by 64")
    return (
        weight.reshape(size_k // k_tile, k_tile, size_n)
        .transpose(1, 2)
        .reshape(size_k // k_tile, size_n * k_tile)
    )


direct_register_custom_op(
    op_name="fused_experts_impl_int8_marlin",
    op_func=fused_experts_impl_int8_marlin_wrapper,
    mutates_args=[],
    fake_impl=fused_experts_impl_int8_marlin_fake,
)
