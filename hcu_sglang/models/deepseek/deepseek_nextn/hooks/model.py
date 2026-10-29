"""DeepseekModelNextN hooks."""

from __future__ import annotations

from hcu_sglang.models.deepseek.hook_utils import (
    register_method_replaces_from_path,
)

_BASE = "sglang.srt.models.deepseek_nextn"
_SRC = "hcu_sglang.models.deepseek.deepseek_nextn.methods.model"

_METHODS = (
    "__init__",
    "forward",
)


def register_hooks() -> int:
    return register_method_replaces_from_path(
        _BASE,
        "DeepseekModelNextN",
        _METHODS,
        _SRC,
    )
