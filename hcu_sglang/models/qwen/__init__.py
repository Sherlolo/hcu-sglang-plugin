"""HCU Qwen model adaptations."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from hcu_sglang.models.qwen import qwen2, qwen3, qwen3_moe, qwen3_5

    qwen2.register()
    qwen3.register()
    qwen3_moe.register()
    qwen3_5.register()
    logger.debug("[hcu_sglang] qwen hooks registered")
