"""Qwen3 model hooks for DCU fused RMSNorm + RoPE KV-store."""

import logging

import torch

from sglang.srt.runtime_context import get_exec
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from hcu_sglang.models.fused_rotary import (
    fused_rms_rotary_kv_store,
    use_fused_rms_rotary,
)

logger = logging.getLogger(__name__)

_TARGET = "sglang.srt.models.qwen3.Qwen3Attention.forward"


def register() -> None:
    logger.info("[hcu_sglang] register AROUND hook: %s", _TARGET)
    HookRegistry.register(_TARGET, replacement_forward, HookType.AROUND)


def replacement_forward(original_fn, self, positions, hidden_states, forward_batch):
    if not use_fused_rms_rotary():
        return original_fn(self, positions, hidden_states, forward_batch)

    determinism = get_exec().deterministic
    if (
        getattr(self, "use_fused_qk_norm_mrope", False)
        and forward_batch.forward_mode.is_decode()
        and determinism.rl_on_policy_target is None
    ):
        return original_fn(self, positions, hidden_states, forward_batch)
    if determinism.rl_on_policy_target is not None:
        return original_fn(self, positions, hidden_states, forward_batch)

    try:
        from lightop import rms_rotary_embedding_fuse_with_kv_store
    except ImportError as exc:
        logger.warning(
            "[hcu_sglang] qwen3 fused RMS rotary is unavailable, fallback: %s",
            exc,
        )
        return original_fn(self, positions, hidden_states, forward_batch)

    qkv, _ = self.qkv_proj(hidden_states)
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
    attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=True)
    output, _ = self.o_proj(attn_output)
    return output
