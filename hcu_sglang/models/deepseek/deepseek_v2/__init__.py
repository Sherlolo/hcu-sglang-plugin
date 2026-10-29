"""DCU DeepSeek V2 plugin package."""

from __future__ import annotations

import logging

from . import hooks as v2_hooks
from hcu_sglang.models.deepseek.hook_utils import ensure_post_apply_module_patch

logger = logging.getLogger(__name__)

def register_hooks() -> None:
    """Register method-level REPLACE hooks for DCU DeepSeek V2 fork delta."""
    n = v2_hooks.register_hooks()

    ensure_post_apply_module_patch()

    logger.info(
        "[hcu_sglang] Registered %d precise DeepSeek V2 hooks",
        n,
    )
