"""DCU-specific module-level flags for deepseek_common.utils patch."""

from __future__ import annotations

from hcu_sglang.utils.detection import is_dcu

_is_dcu = is_dcu()

MODULE_PATCH_ATTRS = ("_is_dcu",)
