"""DCU deepseek_common plugin package."""

from __future__ import annotations

import logging

from . import hooks as common_hooks

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    """Register hooks for deepseek_common DCU fork delta."""
    n = common_hooks.register_hooks()

    logger.info(
        "[hcu_sglang] Registered %d deepseek_common hooks",
        n,
    )
