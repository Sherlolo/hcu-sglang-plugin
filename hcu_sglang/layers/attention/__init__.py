"""DCU attention adaptations."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from hcu_sglang.layers.attention import (
        dcu_mla_backend,
        current_dsa,
        flashattention_patch,
        nsa_patch,
    )

    flashattention_patch.register_hooks()
    nsa_patch.register_hooks()
    dcu_mla_backend.register_hooks()
    current_dsa.register_hooks()
    logger.debug("[hcu_sglang] attention hooks registered")
