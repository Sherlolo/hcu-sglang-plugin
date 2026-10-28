"""HCU DeepSeek-V4 KV-pool adaptations."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from hcu_sglang.utils.detection import is_dcu


def single_is_bf16_attention_kv_cache(pool: Any) -> bool:
    return is_dcu() and pool.dtype == torch.bfloat16


def single_logical_kv_dim(pool: Any) -> int:
    return pool.qk_nope_head_dim + pool.qk_rope_head_dim


def token_is_bf16_attention_kv_cache(pool: Any) -> bool:
    return pool._unified_kv or (
        pool.swa_kv_pool is not None
        and single_is_bf16_attention_kv_cache(pool.swa_kv_pool)
    )


def around_get_bytes_per_token(original: Callable[..., int], pool: Any) -> int:
    if not single_is_bf16_attention_kv_cache(pool):
        return original(pool)
    return single_logical_kv_dim(pool) * torch.bfloat16.itemsize


def around_create_buffer(
    original: Callable[..., torch.Tensor], pool: Any, *, num_pages: int
):
    if not single_is_bf16_attention_kv_cache(pool):
        return original(pool, num_pages=num_pages)
    logical_kv_dim = single_logical_kv_dim(pool)
    pool.kv_cache_total_dim = logical_kv_dim
    pool.bytes_per_page_padded = pool.page_size * pool.get_bytes_per_token()
    return torch.zeros(
        num_pages,
        pool.page_size * logical_kv_dim,
        dtype=pool.store_dtype,
        device=pool.device,
    )


def around_set_key_buffer(original, pool: Any, layer_id, loc, packed) -> None:
    assert not single_is_bf16_attention_kv_cache(pool)
    return original(pool, layer_id, loc, packed)


def set_key_buffer_bf16(
    pool: Any,
    layer_id: int,
    loc: torch.Tensor,
    cache_k: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> None:
    assert single_is_bf16_attention_kv_cache(pool)
    logical_kv_dim = single_logical_kv_dim(pool)
    values = cache_k.to(torch.bfloat16).contiguous().view(-1, logical_kv_dim)
    assert loc.numel() == values.shape[0], (
        "BF16 DSV4 KV locations must match cache rows: "
        f"{loc.numel()=} rows={values.shape[0]}"
    )
    if valid_mask is not None:
        valid_mask = valid_mask.reshape(-1).bool()
        loc = loc.reshape(-1)[valid_mask]
        values = values[valid_mask]
    if loc.numel() == 0:
        return None

    from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_triton

    kv_buffer = pool.kv_buffer[layer_id].view(-1, logical_kv_dim)
    set_mla_kv_buffer_triton(
        kv_buffer,
        loc.long(),
        values[..., : pool.qk_nope_head_dim],
        values[..., pool.qk_nope_head_dim :],
    )


def around_set_key_buffer_fused(
    original,
    pool: Any,
    layer_id: int,
    loc: torch.Tensor,
    cache_k: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> None:
    if single_is_bf16_attention_kv_cache(pool):
        return set_key_buffer_bf16(pool, layer_id, loc, cache_k, valid_mask)
    return original(pool, layer_id, loc, cache_k)


def set_key_buffer_lightop_fused(
    pool: Any,
    layer_id: int,
    loc: torch.Tensor,
    cache_k: torch.Tensor,
    eps: float = 1e-8,
) -> None:
    assert not single_is_bf16_attention_kv_cache(pool)
    from lightop import op

    op.quantize_nope_fp8_rope_bf16_pack_store(
        cache_k,
        pool.kv_buffer[layer_id],
        loc,
        pool.page_size,
        eps,
    )


def around_swa_fused_norm_rope(
    original,
    pool: Any,
    layer_id: int,
    swa_loc: torch.Tensor,
    kv: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    if not token_is_bf16_attention_kv_cache(pool):
        return original(
            pool,
            layer_id,
            swa_loc,
            kv,
            kv_weight,
            eps,
            freqs_cis,
            positions,
        )
    from sglang.kernels.ops.attention.deepseek_v4_rope import (
        fused_norm_rope_inplace_triton,
    )

    fused_norm_rope_inplace_triton(
        kv,
        kv_weight,
        eps,
        freqs_cis,
        positions=positions,
    )
    return pool.swa_kv_pool.set_key_buffer_fused(
        pool._swa_local_layer_id(layer_id), swa_loc, kv
    )


def set_swa_key_buffer_radix_lightop_fused(
    pool: Any,
    layer_id: int,
    swa_loc: torch.Tensor,
    cache_k: torch.Tensor,
    eps: float = 1e-8,
) -> None:
    return pool.swa_kv_pool.set_key_buffer_lightop_fused(
        pool._swa_local_layer_id(layer_id), swa_loc, cache_k, eps
    )


def around_set_extra_key_buffer_fused(
    original,
    pool: Any,
    layer_id: int,
    loc: torch.Tensor,
    cache_k: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> None:
    if valid_mask is None:
        return original(pool, layer_id, loc, cache_k)
    _, compress_layer_id, compress_kv_pool = pool.layer_mapping[layer_id]
    assert compress_kv_pool is not None
    return compress_kv_pool.set_key_buffer_fused(
        compress_layer_id, loc, cache_k, valid_mask
    )


def set_extra_key_buffer_lightop_fused(
    pool: Any,
    layer_id: int,
    loc: torch.Tensor,
    cache_k: torch.Tensor,
    eps: float = 1e-8,
) -> None:
    _, compress_layer_id, compress_kv_pool = pool.layer_mapping[layer_id]
    assert compress_kv_pool is not None
    return compress_kv_pool.set_key_buffer_lightop_fused(
        compress_layer_id, loc, cache_k, eps
    )


def _install_compat_members() -> None:
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        DeepSeekV4SingleKVPool,
        DeepSeekV4TokenToKVPool,
    )

    DeepSeekV4SingleKVPool.is_bf16_attention_kv_cache = property(
        single_is_bf16_attention_kv_cache
    )
    DeepSeekV4SingleKVPool.logical_kv_dim = property(single_logical_kv_dim)
    DeepSeekV4TokenToKVPool.is_bf16_attention_kv_cache = property(
        token_is_bf16_attention_kv_cache
    )
    if not hasattr(DeepSeekV4SingleKVPool, "set_key_buffer_bf16"):
        DeepSeekV4SingleKVPool.set_key_buffer_bf16 = set_key_buffer_bf16
    if not hasattr(DeepSeekV4SingleKVPool, "set_key_buffer_lightop_fused"):
        DeepSeekV4SingleKVPool.set_key_buffer_lightop_fused = (
            set_key_buffer_lightop_fused
        )
    if not hasattr(DeepSeekV4TokenToKVPool, "set_swa_key_buffer_radix_lightop_fused"):
        DeepSeekV4TokenToKVPool.set_swa_key_buffer_radix_lightop_fused = (
            set_swa_key_buffer_radix_lightop_fused
        )
    if not hasattr(DeepSeekV4TokenToKVPool, "set_extra_key_buffer_lightop_fused"):
        DeepSeekV4TokenToKVPool.set_extra_key_buffer_lightop_fused = (
            set_extra_key_buffer_lightop_fused
        )


def register_hooks() -> None:
    if is_dcu():
        from hcu_sglang.mem_cache.hcu_v4_integration import register_hooks as register_current
        register_current()
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _install_compat_members()
    single = "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4SingleKVPool"
    HookRegistry.register(
        f"{single}.get_bytes_per_token", around_get_bytes_per_token, HookType.AROUND
    )
    HookRegistry.register(
        f"{single}.create_buffer", around_create_buffer, HookType.AROUND
    )
    HookRegistry.register(
        f"{single}.set_key_buffer", around_set_key_buffer, HookType.AROUND
    )
    HookRegistry.register(
        f"{single}.set_key_buffer_fused",
        around_set_key_buffer_fused,
        HookType.AROUND,
    )
    token = "sglang.srt.mem_cache.deepseek_v4_memory_pool.DeepSeekV4TokenToKVPool"
    HookRegistry.register(
        f"{token}.set_swa_key_buffer_radix_fused_norm_rope",
        around_swa_fused_norm_rope,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{token}.set_extra_key_buffer_fused",
        around_set_extra_key_buffer_fused,
        HookType.AROUND,
    )
