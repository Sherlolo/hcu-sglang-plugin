"""DeepseekV2MoE method hooks."""

from __future__ import annotations

from hcu_sglang.models.deepseek.hook_utils import (
    register_method_replaces_from_path,
)

_BASE = "sglang.srt.models.deepseek_v2"
_SRC = "hcu_sglang.models.deepseek.deepseek_v2.methods.moe"

_MOE_METHODS = (
    "__init__",
    "forward",
    "forward_normal",
    "forward_deepep",
    "forward_normal_dual_stream",
    "_forward_shared_experts",
)


def register_hooks() -> int:
    return register_method_replaces_from_path(
        _BASE,
        "DeepseekV2MoE",
        _MOE_METHODS,
        _SRC,
    )
