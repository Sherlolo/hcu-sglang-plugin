"""DCU-specific module-level flags for DeepSeek V4."""

from __future__ import annotations

import logging

from sglang.srt.utils import get_bool_env_var
from hcu_sglang.utils.detection import is_dcu

logger = logging.getLogger(__name__)

_is_dcu = is_dcu()
_use_dpskv4_lightop_rmsnorm = get_bool_env_var("SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM")
_use_fused_qnorm_rope_kv_rope_quant = get_bool_env_var(
    "SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT"
)
_use_aiter_tilelang_mhc = get_bool_env_var("SGLANG_ROCM_USE_AITER_TILELANG_MHC")

op = None
if _is_dcu:
    try:
        from lightop import op as _lightop_op

        op = _lightop_op
    except ImportError:
        logger.warning("lightop not installed; DPSK V4 lightop paths disabled")
    if _use_aiter_tilelang_mhc:
        try:
            from aiter.ops.tilelang import mhc_post_fwd, mhc_pre_big_fuse  # noqa: F401
        except ImportError:
            logger.warning(
                "aiter tilelang MHC not available; SGLANG_ROCM_USE_AITER_TILELANG_MHC ignored"
            )

MODULE_PATCH_ATTRS = (
    "_is_dcu",
    "_use_dpskv4_lightop_rmsnorm",
    "_use_fused_qnorm_rope_kv_rope_quant",
    "_use_aiter_tilelang_mhc",
    "op",
)
