"""Runtime namespace for extracted DeepSeek common methods."""

from __future__ import annotations

import sglang.srt.models.deepseek_common.deepseek_weight_loader as _upstream

from hcu_sglang.models.deepseek.deepseek_common import constants as C

globals().update(
    {
        name: value
        for name, value in vars(_upstream).items()
        if not name.startswith("__")
    }
)

_is_hcu = C._is_dcu

__all__ = [name for name in globals() if not name.startswith("__") and name != "_upstream"]
