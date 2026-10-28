"""Qwen2 HCU fused split-QKV, rotary, and KV-store path."""

from __future__ import annotations

import logging
import os

from hcu_sglang.models.fused_rotary import (
    get_kv_cache_dtype,
    mark_fused_kv_store,
)
from hcu_sglang.utils.detection import is_dcu

logger = logging.getLogger(__name__)


def after_init(result, attention, *args, **kwargs):
    if not is_dcu():
        return result
    layer_id = kwargs.get("layer_id")
    if layer_id is None and len(args) >= 5:
        layer_id = args[4]
    attention.layer_id = 0 if layer_id is None else layer_id
    attention.page_size = 64
    attention.kv_cache_dtype = get_kv_cache_dtype(attention.qkv_proj.weight.dtype)
    return result


def around_forward(original, attention, positions, hidden_states, forward_batch):
    if (
        not is_dcu()
        or os.getenv("SGLANG_USE_FUSED_SPLIT_QKV_RMS_ROTARY_EMBEDDING", "0")
        != "1"
    ):
        return original(attention, positions, hidden_states, forward_batch)

    try:
        from lightop import split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant
    except Exception as error:
        logger.warning("Qwen2 HCU fused rotary is unavailable: %s", error)
        return original(attention, positions, hidden_states, forward_batch)

    qkv, _ = attention.qkv_proj(hidden_states)
    cos_sin_cache = attention.rotary_emb.cos_sin_cache
    if cos_sin_cache.device != qkv.device or cos_sin_cache.dtype != qkv.dtype:
        cos_sin_cache = cos_sin_cache.to(
            qkv.device, dtype=qkv.dtype, non_blocking=True
        )
        attention.rotary_emb.cos_sin_cache = cos_sin_cache
    k_buffer, v_buffer = forward_batch.token_to_kv_pool.get_kv_buffer(
        attention.layer_id
    )
    q, k, v = split_qkv_rms_rotary_embedding_fuse_with_kv_store_quant(
        positions,
        qkv.contiguous(),
        attention.q_size,
        attention.kv_size,
        cos_sin_cache,
        head_dim=attention.head_dim,
        page_size=attention.page_size,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        kv_cache_loc=forward_batch.out_cache_loc,
        is_neox=True,
        weight_q=None,
        weight_k=None,
        output_dtype=qkv.dtype,
        kv_cache_dtype=attention.kv_cache_dtype,
        epsilon=1e-5,
        residual_q=None,
        residual_k=None,
        k_scale=None,
        v_scale=None,
    )
    num_tokens = qkv.shape[0]
    q = q.contiguous().view(
        num_tokens, attention.q_size // attention.head_dim, attention.head_dim
    )
    k = k.contiguous().view(
        num_tokens, attention.kv_size // attention.head_dim, attention.head_dim
    )
    v = v.contiguous().view(
        num_tokens, attention.kv_size // attention.head_dim, attention.head_dim
    )
    mark_fused_kv_store(forward_batch, attention.layer_id)
    attn_output = attention.attn(q, k, v, forward_batch, save_kv_cache=False)
    output, _ = attention.o_proj(attn_output)
    return output


def register() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    target = "sglang.srt.models.qwen2.Qwen2Attention"
    HookRegistry.register(f"{target}.__init__", after_init, HookType.AFTER)
    HookRegistry.register(f"{target}.forward", around_forward, HookType.AROUND)
