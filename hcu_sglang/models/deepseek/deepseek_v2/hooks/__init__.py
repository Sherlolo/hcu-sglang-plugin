"""DeepSeek V2 method-level plugin hooks."""

from __future__ import annotations

from hcu_sglang.models.deepseek.deepseek_v2.hooks import (
    attention,
    decoder,
    mlp,
    moe,
)


def register_hooks() -> int:
    n = 0
    n += mlp.register_hooks()
    n += moe.register_hooks()
    n += decoder.register_hooks()
    n += attention.register_hooks()
    return n
