"""Qwen3 model hooks for HCU fused RMSNorm + RoPE KV-store."""

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
    HookRegistry.register("sglang.srt.models.qwen3.Qwen3Attention.__init__", around_init, HookType.AROUND)
    HookRegistry.register(_TARGET, replacement_forward, HookType.AROUND)


def around_init(original, attention, *args, **kwargs):
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        return original(attention, *args, **kwargs)
    import types
    from hcu_sglang.models.qwen.qwen3_source import Qwen3Attention
    Qwen3Attention.__init__(attention, *args, **kwargs)
    for name in ("forward_prepare_native", "forward_prepare_npu", "forward_prepare_aiter_fused"):
        setattr(attention, name, types.MethodType(getattr(Qwen3Attention, name), attention))


def replacement_forward(original_fn, self, positions, hidden_states, forward_batch):
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        return original_fn(self, positions, hidden_states, forward_batch)
    from hcu_sglang.models.qwen.qwen3_source import Qwen3Attention
    return Qwen3Attention.forward(self, positions, hidden_states, forward_batch)
