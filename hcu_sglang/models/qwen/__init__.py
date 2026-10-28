"""DCU Qwen model adaptations."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from hcu_sglang.models.qwen import qwen3, qwen3_moe

    qwen3.register_hooks()
    qwen3_moe.register_hooks()
    logger.debug("[hcu_sglang] qwen hooks registered")
