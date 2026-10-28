"""DCU DeepSeek model adaptations."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from hcu_sglang.models.deepseek import (
        deepseek_common,
        deepseek_nextn,
        deepseek_v2,
        deepseek_v4,
    )

    deepseek_v2.register_hooks()
    deepseek_v4.register_hooks()
    deepseek_common.register_hooks()
    deepseek_nextn.register_hooks()
    logger.debug("[hcu_sglang] deepseek hooks registered")
