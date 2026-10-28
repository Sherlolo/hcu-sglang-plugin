"""DCU deepseek_nextn plugin package."""

from __future__ import annotations

import logging

from . import hooks as nextn_hooks

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    """Register hooks for deepseek_nextn DCU fork delta."""
    n = nextn_hooks.register_hooks()

    logger.info(
        "[hcu_sglang] Registered %d deepseek_nextn hooks",
        n,
    )
