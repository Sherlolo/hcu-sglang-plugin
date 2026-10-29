"""Qwen3.5 model hooks for DCU fused RMSNorm + RoPE KV-store."""

import logging

import torch

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from hcu_sglang.models.fused_rotary import (
    fused_rms_rotary_kv_store,
    use_fused_rms_rotary,
)
from hcu_sglang.utils.module_patch import get_original_attr

logger = logging.getLogger(__name__)

_TARGET = "sglang.srt.models.qwen3_5.Qwen3_5AttentionDecoderLayer.self_attention"
_MODULE = "sglang.srt.models.qwen3_5"


def register() -> None:
    logger.info("[hcu_sglang] register REPLACE hook: %s", _TARGET)
    HookRegistry.register(_TARGET, replacement_self_attention, HookType.REPLACE)


def replacement_self_attention(self, positions, hidden_states, forward_batch):
    original = get_original_attr(
        _MODULE, "Qwen3_5AttentionDecoderLayer.self_attention"
    )
    if not use_fused_rms_rotary():
        return original(self, positions, hidden_states, forward_batch)

    try:
        from lightop import gammarms_rotary_embedding_fuse_with_kv_store
    except Exception as exc:
        logger.warning(
            "[hcu_sglang] qwen3_5 fused RMS rotary is unavailable, fallback: %s",
            exc,
        )
        return original(self, positions, hidden_states, forward_batch)

    qkv, _ = self.qkv_proj(hidden_states)

    if self.attn_output_gate:
        q_gate, k, v = qkv.split(
            [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
        )
        orig_shape = q_gate.shape[:-1]
        q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        q = q.reshape(*orig_shape, -1)
        gate = gate.reshape(*orig_shape, -1)
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        gate = None

    q, k, v = fused_rms_rotary_kv_store(
        gammarms_rotary_embedding_fuse_with_kv_store,
        self,
        positions,
        q,
        k,
        v,
        forward_batch,
        self.q_norm,
        self.k_norm,
    )
    attn_output = self.attn(q, k, v, forward_batch)

    if gate is not None:
        attn_output = attn_output * torch.sigmoid(gate)

    output, _ = self.o_proj(attn_output)
    return output
