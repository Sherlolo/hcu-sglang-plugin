"""Runtime namespace for extracted DeepSeek V4 methods."""

from __future__ import annotations

import sglang.srt.models.deepseek_v4 as _upstream
from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
    try_fused_hc_post_pre,
)
from sglang.srt.utils.common import is_sm120_supported

from hcu_sglang.models.deepseek.deepseek_v4 import constants as C

globals().update(
    {
        name: value
        for name, value in vars(_upstream).items()
        if not name.startswith("__")
    }
)

_is_hcu = C._is_dcu
_use_aiter_tilelang_mhc = C._use_aiter_tilelang_mhc


def _is_fused_mhc_post_pre_enabled() -> bool:
    return (
        not _is_hcu
        and envs.SGLANG_OPT_FUSE_MHC_POST_PRE.get()
        and envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get()
        and (envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get() or is_sm120_supported())
    )


__all__ = [name for name in globals() if not name.startswith("__") and name != "_upstream"]
