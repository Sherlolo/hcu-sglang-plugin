"""DCU attention backend handler additions."""

from __future__ import annotations

from sglang.srt.models.deepseek_common.attention_backend_handler import (
    _handle_attention_backend,
)


def handle_attention_dcu_mla(attn, forward_batch):
    return _handle_attention_backend(attn, forward_batch, "dcu_mla")
