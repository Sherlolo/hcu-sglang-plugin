"""DeepseekV4DecoderLayer method hooks."""

from __future__ import annotations

from hcu_sglang.models.deepseek.hook_utils import (
    register_method_replaces_from_path,
)

_BASE = "sglang.srt.models.deepseek_v4"
_SRC = "hcu_sglang.models.deepseek.deepseek_v4.methods.decoder"


def register_hooks() -> int:
    return register_method_replaces_from_path(
        _BASE,
        "DeepseekV4DecoderLayer",
        ("hc_pre", "hc_post", "forward"),
        _SRC,
    )
