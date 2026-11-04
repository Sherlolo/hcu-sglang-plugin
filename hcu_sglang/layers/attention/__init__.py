"""HCU attention adaptations."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from hcu_sglang.layers.attention import (
        hcu_mla_registry,
        current_dsa,
        flashattention_backend,
        nsa_patch,
    )

    flashattention_backend.register()
    nsa_patch.register_hooks()
    hcu_mla_registry.register()
    current_dsa.register_hooks()
    logger.debug("[hcu_sglang] attention hooks registered")
