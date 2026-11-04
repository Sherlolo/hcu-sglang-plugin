"""HCU MLA KV-pool compatibility and write hooks."""

from __future__ import annotations

from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_hcu


def get_key_buffer_deepseek_v2(pool: Any, layer_id: int):
    if pool.layer_transfer_counter is not None:
        pool.layer_transfer_counter.wait_until(layer_id - pool.start_layer)
    buffer = pool.kv_buffer[layer_id - pool.start_layer]
    if pool.store_dtype != pool.dtype and pool.dtype not in (
        torch.float8_e5m2,
        torch.float8_e4m3fn,
    ):
        return buffer.view(pool.dtype)
    return buffer, pool.dtype


def set_kv_buffer_opt(
    pool: Any,
    layer: Any,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
):
    layer_id = layer.layer_id
    assert not (pool.use_dsa and pool.dsa_kv_cache_store_fp8)
    cache_k = torch.cat([cache_k_nope, cache_k_rope], dim=-1)
    if cache_k.dtype != pool.dtype:
        cache_k = cache_k.to(pool.dtype)
    if pool.store_dtype != pool.dtype:
        cache_k = cache_k.view(pool.store_dtype)
    pool.kv_buffer[layer_id - pool.start_layer][loc] = cache_k


def _write_triton(
    dst_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
) -> None:
    from sglang.kernels.ops.kvcache.mla_buffer import set_mla_kv_buffer_triton

    set_mla_kv_buffer_triton(
        dst_buffer,
        loc,
        cache_k_nope,
        cache_k_rope,
    )


def _write_lightop(
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
    dst_buffer: torch.Tensor,
    loc: torch.Tensor,
) -> None:
    from lightop import kvcache

    kvcache.fused_quantize_and_store_mla_kv_cache(
        cache_k_nope,
        cache_k_rope,
        dst_buffer,
        loc,
        "fp8_e4m3",
        1e-6,
    )


def around_write_mla_kv_buffer(
    original: Callable[..., None],
    pool: Any,
    dst_buffer: torch.Tensor,
    loc: torch.Tensor,
    cache_k_nope: torch.Tensor,
    cache_k_rope: torch.Tensor,
) -> None:
    if not is_hcu():
        return original(pool, dst_buffer, loc, cache_k_nope, cache_k_rope)
    if pool.dsa_kv_cache_store_fp8:
        _write_lightop(cache_k_nope, cache_k_rope, dst_buffer, loc)
        return None

    if cache_k_nope.dtype != pool.dtype:
        cache_k_nope = cache_k_nope.to(pool.dtype)
        cache_k_rope = cache_k_rope.to(pool.dtype)
    if pool.store_dtype != pool.dtype:
        cache_k_nope = cache_k_nope.view(pool.store_dtype)
        cache_k_rope = cache_k_rope.view(pool.store_dtype)
    _write_triton(dst_buffer, loc, cache_k_nope, cache_k_rope)
    return None


def _install_missing_methods() -> None:
    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    if not hasattr(MLATokenToKVPool, "get_key_buffer_DeepSeekV2"):
        MLATokenToKVPool.get_key_buffer_DeepSeekV2 = get_key_buffer_deepseek_v2
    if not hasattr(MLATokenToKVPool, "set_kv_buffer_opt"):
        MLATokenToKVPool.set_kv_buffer_opt = set_kv_buffer_opt


def around_calculate_mla_kv_cache_dim(original, *, model_config, kv_cache_dtype):
    if not is_hcu():
        return original(model_config=model_config, kv_cache_dtype=kv_cache_dtype)
    from sglang.srt.configs.model_config import is_deepseek_dsa

    if is_deepseek_dsa(model_config.hf_config):
        return original(model_config=model_config, kv_cache_dtype=kv_cache_dtype)
    from sglang.srt.runtime_context import get_exec
    from hcu_sglang.layers.attention.hcu_mla_utils import get_hcu_mla_fp8_kv_cache_dim

    kernel = get_exec().kernel
    return get_hcu_mla_fp8_kv_cache_dim(
        model_config.kv_lora_rank + model_config.qk_rope_head_dim,
        model_config.qk_rope_head_dim,
        kv_cache_dtype,
        uses_hcu_mla=any(
            getattr(kernel, name, None) == "hcu_mla"
            for name in (
                "attention_backend",
                "prefill_attention_backend",
                "decode_attention_backend",
            )
        ),
    )


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _install_missing_methods()
    HookRegistry.register(
        "sglang.srt.mem_cache.memory_pool.MLATokenToKVPool._write_mla_kv_buffer",
        around_write_mla_kv_buffer,
        HookType.AROUND,
    )

    HookRegistry.register(
        "sglang.srt.mem_cache.kv_cache_configurator.calculate_mla_kv_cache_dim",
        around_calculate_mla_kv_cache_dim,
        HookType.AROUND,
    )
