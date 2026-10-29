"""DCU model adaptations and hook registration."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from hcu_sglang.models import deepseek, qwen, utils

    deepseek.register_hooks()
    qwen.register_hooks()
    utils.register_hooks()
    logger.info("[hcu_sglang] model hooks registered")
