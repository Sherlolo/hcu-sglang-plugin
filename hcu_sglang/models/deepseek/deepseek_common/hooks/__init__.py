"""deepseek_common hook registration."""

from __future__ import annotations

from hcu_sglang.models.deepseek.deepseek_common.hooks import weight_loader


def register_hooks() -> int:
    n = 0
    n += weight_loader.register_hooks()
    return n
