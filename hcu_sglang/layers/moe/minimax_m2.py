"""MiniMax M2 sequence-parallel LightOp MoE helper."""

from __future__ import annotations

from typing import Optional

import torch


def fused_experts_impl_int8_marlin_minimax_m2(
    num_tokens: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    global_num_experts: int,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: Optional[torch.Tensor],
    a2_scale: Optional[torch.Tensor],
    i_q: torch.Tensor,
    i_s: torch.Tensor,
    routed_scaling_factor: float = 1.0,
) -> torch.Tensor:
    hidden_states = torch.empty(
        (num_tokens, hidden_size),
        device=device,
        dtype=dtype,
    )
    return torch.ops.sglang.fused_experts_impl_int8_marlin(
        hidden_states,
        w1,
        w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        inplace=False,
        activation="silu",
        apply_router_weight_on_input=False,
        use_fp8_w8a8=False,
        use_int8_w8a8=True,
        per_channel_quant=True,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        use_nn_moe=False,
        routed_scaling_factor=routed_scaling_factor,
        i_q=i_q,
        i_s=i_s,
    )
