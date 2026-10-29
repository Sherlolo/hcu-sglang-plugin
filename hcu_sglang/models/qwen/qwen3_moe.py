"""Qwen3-MoE model hooks for DCU fused RMSNorm + RoPE KV-store."""

import logging

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from hcu_sglang.models.fused_rotary import (
    fused_rms_rotary_kv_store,
    use_fused_rms_rotary,
)
from hcu_sglang.utils.module_patch import get_original_attr

logger = logging.getLogger(__name__)

_TARGET = "sglang.srt.models.qwen3_moe.Qwen3MoeAttention.apply_qk_norm_rope"
_MODULE = "sglang.srt.models.qwen3_moe"


def register() -> None:
    logger.info("[hcu_sglang] register REPLACE hook: %s", _TARGET)
    HookRegistry.register(_TARGET, replacement_apply_qk_norm_rope, HookType.REPLACE)


def replacement_apply_qk_norm_rope(self, qkv, positions, forward_batch):
    original = get_original_attr(_MODULE, "Qwen3MoeAttention.apply_qk_norm_rope")
    use_fused = (
        self.use_fused_qk_norm_rope and qkv.dtype == torch.bfloat16
    ) or (
        self.use_fused_qk_norm_rope_cpu
        and qkv.dtype in (torch.bfloat16, torch.float16)
    )
    if use_fused or not use_fused_rms_rotary():
        return original(self, qkv, positions, forward_batch)

    try:
        from lightop import rms_rotary_embedding_fuse_with_kv_store
    except Exception as exc:
        logger.warning(
            "[hcu_sglang] qwen3_moe fused RMS rotary is unavailable, fallback: %s",
            exc,
        )
        return original(self, qkv, positions, forward_batch)

    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k, v = fused_rms_rotary_kv_store(
        rms_rotary_embedding_fuse_with_kv_store,
        self,
        positions,
        q,
        k,
        v,
        forward_batch,
        self.q_norm,
        self.k_norm,
    )
    self._used_fused_qk_norm_rope_last_call = False
    return q, k, v
