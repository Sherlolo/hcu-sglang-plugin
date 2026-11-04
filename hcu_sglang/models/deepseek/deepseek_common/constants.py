"""HCU-specific module-level flags for deepseek_common.utils patch."""

from __future__ import annotations

from hcu_sglang.utils.detection import is_hcu

_is_hcu = is_hcu()

MODULE_PATCH_ATTRS = ("_is_hcu",)
