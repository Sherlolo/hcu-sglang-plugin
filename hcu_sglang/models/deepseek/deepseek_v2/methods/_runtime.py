"""Runtime namespace for extracted DeepSeek V2 methods."""

from __future__ import annotations

import sglang.srt.models.deepseek_v2 as _upstream

from hcu_sglang.models.deepseek.deepseek_v2 import constants as C

globals().update(
    {
        name: value
        for name, value in vars(_upstream).items()
        if not name.startswith("__")
    }
)

_is_hcu = C._is_dcu

__all__ = [name for name in globals() if not name.startswith("__") and name != "_upstream"]
