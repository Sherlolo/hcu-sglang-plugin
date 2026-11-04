"""HCU MLA attention backend registration."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_REGISTERED = False


def register_attention_hcu_mla_backend() -> None:
    """Add handle_attention_hcu_mla and register the hcu_mla backend."""
    global _REGISTERED
    if _REGISTERED:
        return

    import sglang.srt.models.deepseek_common.attention_backend_handler as handler_mod
    from sglang.srt.models.deepseek_common.attention_backend_handler import (
        AttentionBackendRegistry,
    )

    from hcu_sglang.models.deepseek.deepseek_common.methods import (
        attention_backend_handler,
    )

    handler_mod.handle_attention_hcu_mla = (
        attention_backend_handler.handle_attention_hcu_mla
    )
    AttentionBackendRegistry.register(
        "hcu_mla",
        attention_backend_handler.handle_attention_hcu_mla,
    )
    _REGISTERED = True
    logger.debug("[hcu_sglang] Registered hcu_mla attention backend")


def register_hooks() -> int:
    register_attention_hcu_mla_backend()
    return 0
