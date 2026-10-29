"""HCU static-shape masked MLA scatter; preserves source kernel semantics."""
from __future__ import annotations
import torch
import triton
import triton.language as tl

@triton.jit
def set_mla_kv_buffer_masked_kernel(
    kv_buffer_ptr,
    cache_k_nope_ptr,
    cache_k_rope_ptr,
    loc_ptr,
    valid_ptr,
    buffer_stride: tl.constexpr,
    nope_stride: tl.constexpr,
    rope_stride: tl.constexpr,
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Static-shape BF16 scatter that skips padded compressor-plan rows."""
    pid_loc = tl.program_id(0)
    pid_blk = tl.program_id(1)

    base = pid_blk * BLOCK
    offs = base + tl.arange(0, BLOCK)
    total_dim = nope_dim + rope_dim
    valid = tl.load(valid_ptr + pid_loc)
    mask = (offs < total_dim) & valid

    loc = tl.load(loc_ptr + pid_loc).to(tl.int64)
    dst_ptr = kv_buffer_ptr + loc * buffer_stride + offs

    if base + BLOCK <= nope_dim:
        src = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask,
            other=0,
        )
    elif base >= nope_dim:
        offs_rope = offs - nope_dim
        src = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + offs_rope,
            mask=mask,
            other=0,
        )
    else:
        is_nope = offs < nope_dim
        is_rope = (offs >= nope_dim) & (offs < total_dim)
        src_nope = tl.load(
            cache_k_nope_ptr + pid_loc * nope_stride + offs,
            mask=mask & is_nope,
            other=0,
        )
        src_rope = tl.load(
            cache_k_rope_ptr + pid_loc * rope_stride + (offs - nope_dim),
            mask=mask & is_rope,
            other=0,
        )
        src = tl.where(is_nope, src_nope, src_rope)

    tl.store(dst_ptr, src, mask=mask)


def set_mla_kv_buffer_triton_masked(
    kv_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    valid: torch.Tensor,
):
    """Scatter only valid rows from a static-shape compressor plan."""
    n_loc = loc.numel()
    nope_dim = cache_k_nope.shape[-1]
    rope_dim = cache_k_rope.shape[-1]
    total_dim = nope_dim + rope_dim
    BLOCK = triton.next_power_of_2(total_dim)
    grid = (n_loc, 1)
    set_mla_kv_buffer_masked_kernel[grid](
        kv_buffer,
        cache_k_nope,
        cache_k_rope,
        loc,
        valid,
        kv_buffer.stride(0),
        cache_k_nope.stride(0),
        cache_k_rope.stride(0),
        nope_dim,
        rope_dim,
        BLOCK=BLOCK,
    )

