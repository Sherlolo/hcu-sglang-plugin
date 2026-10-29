"""Shared DCU fused RMSNorm + RoPE + KV-store helpers."""

from __future__ import annotations

from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

import logging

import torch

from sglang.srt.runtime_context import get_model
from sglang.srt.utils import get_bool_env_var
from hcu_sglang.utils.detection import is_dcu

logger = logging.getLogger(__name__)


_use_fused_qwen_bailing_rotary = get_bool_env_var("SGLANG_USE_FUSED_RMS_ROTARY")

_FUSED_KV_LAYERS_ATTR = "_dcu_fused_rms_rotary_kv_layers"


def use_fused_rms_rotary() -> bool:
    return is_dcu() and _use_fused_qwen_bailing_rotary


def mark_fused_kv_store(forward_batch, layer_id: int) -> None:
    layers = getattr(forward_batch, _FUSED_KV_LAYERS_ATTR, None)
    if layers is None:
        layers = set()
        setattr(forward_batch, _FUSED_KV_LAYERS_ATTR, layers)
    layers.add(layer_id)


def has_fused_kv_store(forward_batch, layer_id: int) -> bool:
    layers = getattr(forward_batch, _FUSED_KV_LAYERS_ATTR, None)
    return layers is not None and layer_id in layers


def get_kv_cache_dtype(default: torch.dtype) -> torch.dtype:
    kv_cache_dtype = get_model().kv_cache_dtype
    if kv_cache_dtype == "fp8_e4m3":
        return torch.float8_e4m3fn
    if kv_cache_dtype == "fp8_e5m2":
        return torch.float8_e5m2
    if kv_cache_dtype in ("bf16", "bfloat16"):
        return torch.bfloat16
    return default


def get_layer_id(module) -> int:
    layer_id = getattr(module, "layer_id", None)
    if layer_id is not None:
        return layer_id
    return module.attn.layer_id


def fused_rms_rotary_kv_store(
    fused_op,
    module,
    positions,
    q,
    k,
    v,
    forward_batch,
    q_norm,
    k_norm,
):
    cos_sin_cache = module.rotary_emb.cos_sin_cache
    if cos_sin_cache.device != q.device or cos_sin_cache.dtype != q.dtype:
        cos_sin_cache = cos_sin_cache.to(q.device, dtype=q.dtype, non_blocking=True)
        module.rotary_emb.cos_sin_cache = cos_sin_cache

    layer_id = get_layer_id(module)
    k_buffer, v_buffer = get_token_to_kv_pool().get_kv_buffer(layer_id)

    result = fused_op(
        positions,
        q,
        k,
        v,
        cos_sin_cache,
        module.head_dim,
        getattr(module, "page_size", 64),
        k_buffer,
        v_buffer,
        forward_batch.out_cache_loc,
        is_neox=True,
        weight_q=q_norm.weight,
        weight_k=k_norm.weight,
        output_dtype=getattr(module, "kv_cache_dtype", get_kv_cache_dtype(q.dtype)),
        residual_q=None,
        residual_k=None,
        k_scale=None,
        v_scale=None,
        epsilon=q_norm.variance_epsilon,
    )
    mark_fused_kv_store(forward_batch, layer_id)
    return result
