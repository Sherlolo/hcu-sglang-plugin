"""DCU model utility adaptations."""

from __future__ import annotations

import logging

from hcu_sglang.models.deepseek.hook_utils import (
    register_function_replaces_from_path,
)

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    n = register_function_replaces_from_path(
        "sglang.srt.models.utils",
        ("enable_fused_set_kv_buffer",),
        "hcu_sglang.models.model_utils.methods.utils",
    )
    logger.info("[hcu_sglang] Registered %d models.utils hooks", n)
