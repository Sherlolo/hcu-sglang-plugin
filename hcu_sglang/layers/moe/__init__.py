"""DCU MoE adaptations."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from hcu_sglang.layers.moe import aiter_w4a8, lightop_runner

    aiter_w4a8.register_hooks()
    lightop_runner.register_hooks()
    logger.debug("[hcu_sglang] moe hooks registered")
