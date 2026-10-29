"""HCU base attention backend compatibility."""

from __future__ import annotations

from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_dcu


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
    if is_dcu() and forward_batch.forward_mode.is_idle():
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


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.attention.base_attn_backend.AttentionBackend.forward",
        around_forward,
        HookType.AROUND,
    )
