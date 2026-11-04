"""HCU base attention backend compatibility."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from hcu_sglang.utils.detection import is_hcu


def around_forward(
    original: Callable[..., Any],
    backend: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer: Any,
    forward_batch: Any,
    *args,
    **kwargs,
) -> Any:
    if is_hcu() and forward_batch.forward_mode.is_idle():
        return query.new_empty(
            query.shape[0],
            layer.tp_q_head_num * layer.v_head_dim,
            dtype=torch.bfloat16,
        )
    return original(
        backend,
        query,
        key,
        value,
        layer,
        forward_batch,
        *args,
        **kwargs,
    )


def get_fused_kv_write_buffers(
    self, layer: RadixAttention, forward_batch: ForwardBatch
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Return token-major K/V buffers and physical slots for a fused writer."""
    return None

def late_layer_num_tokens(self, forward_batch: ForwardBatch) -> Optional[int]:
    """Rows this rank runs from a late-layer row switch on (DeepSeek-V4.1
    decoder SWA bounded replay); None for backends without one."""
    return None


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    if is_hcu():
        from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
        for name in ("get_fused_kv_write_buffers", "late_layer_num_tokens"):
            if not hasattr(AttentionBackend, name):
                setattr(AttentionBackend, name, globals()[name])

    HookRegistry.register(
        "sglang.srt.layers.attention.base_attn_backend.AttentionBackend.forward",
        around_forward,
        HookType.AROUND,
    )
