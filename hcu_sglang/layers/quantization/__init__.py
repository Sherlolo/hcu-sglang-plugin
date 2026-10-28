"""DCU quantization adaptations."""

import logging

logger = logging.getLogger(__name__)


def register_hooks() -> None:
    from hcu_sglang.layers.quantization import (
        fp8,
        gptq,
        registry,
        unquantized_aiter,
        unquantized_marlin,
        w8a8_aiter,
        w8a8_fp8,
        w8a8_lightop,
    )

    fp8.register_hooks()
    gptq.register_hooks()
    w8a8_fp8.register_hooks()
    registry.register_hooks()
    unquantized_aiter.register_hooks()
    unquantized_marlin.register_hooks()
    w8a8_aiter.register_hooks()
    w8a8_lightop.register_hooks()
    logger.debug("[hcu_sglang] quantization hooks registered")
