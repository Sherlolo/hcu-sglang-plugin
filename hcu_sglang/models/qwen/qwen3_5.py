"""Qwen3.5 model hooks for HCU fused RMSNorm + RoPE KV-store."""

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
    logger.info("[hcu_sglang] register AROUND hook: %s", _TARGET)
    HookRegistry.register(_TARGET, replacement_self_attention, HookType.AROUND)


def replacement_self_attention(original, self, positions, hidden_states, forward_batch):
    return original(self, positions, hidden_states, forward_batch)
