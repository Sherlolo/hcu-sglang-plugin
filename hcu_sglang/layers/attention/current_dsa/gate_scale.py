from __future__ import annotations
import torch
import triton
import triton.language as tl

@triton.jit
def _fused_gate_scale_kernel(
    weights_ptr,
    q_scale_ptr,
    out_ptr,
    scale,
    M,
    K,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    w = tl.load(weights_ptr + pid_m)
    w_scaled = w * scale
    row_start_idx = pid_m * K
    for k_offset in range(0, K, BLOCK_K):
        cols = k_offset + tl.arange(0, BLOCK_K)
        mask = cols < K

        q_ptrs = q_scale_ptr + row_start_idx + cols
        out_ptrs = out_ptr + row_start_idx + cols
        q = tl.load(q_ptrs, mask=mask)
        out = w_scaled * q
        tl.store(out_ptrs, out, mask=mask)

def fused_get_logits_head_gate_triton(
    weights: torch.Tensor,
    q_scale: torch.Tensor,
    n_heads: int,
    softmax_scale: float,
) -> torch.Tensor:
    weights = weights.contiguous()
    q_scale = q_scale.contiguous()

    K = q_scale.size(-1)
    M = weights.numel()

    out_dtype = torch.promote_types(weights.dtype, q_scale.dtype)
    out = torch.empty_like(q_scale, dtype=out_dtype)

    scale = softmax_scale * (n_heads**-0.5)
    block_k = triton.next_power_of_2(K)
    if block_k > 1024:
        block_k = 1024

    grid = (M,)
    _fused_gate_scale_kernel[grid](
        weights,
        q_scale,
        out,
        scale,
        M,
        K,
        BLOCK_K=block_k,
    )
    return out
