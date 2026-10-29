"""Runtime namespace for extracted DeepSeek NextN methods."""

from __future__ import annotations

import sglang.srt.models.deepseek_nextn as _upstream
from sglang.srt.layers.moe.utils import is_sbo_enabled

globals().update(
    {
        name: value
        for name, value in vars(_upstream).items()
        if not name.startswith("__")
    }
)

__all__ = [name for name in globals() if not name.startswith("__") and name != "_upstream"]
