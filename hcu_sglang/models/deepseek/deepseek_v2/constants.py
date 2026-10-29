"""DCU-specific module-level flags for DeepSeek V2 (mirrors fork globals)."""

from __future__ import annotations

import logging

import torch

from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_device_sm,
    get_int_env_var,
    is_cpu,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_npu,
)
from hcu_sglang.utils.detection import is_dcu

logger = logging.getLogger(__name__)

_is_dcu = is_dcu()
_is_hip = is_hip()
_is_cuda = is_cuda()
_is_npu = is_npu()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_device_sm = get_device_sm()
_is_gfx95_supported = is_gfx95_supported()

from sglang.srt.layers.moe.utils import is_sbo_enabled  # noqa: E402

_is_sbo_enabled = is_sbo_enabled()
_use_lightop_moe_sum_mul_add = get_bool_env_var("SGLANG_USE_LIGHTOP_MOE_SUM_MUL_ADD")
_use_fused_silu_mul_quant = get_bool_env_var("SGLANG_USE_FUSED_SILU_MUL_QUANT")
_use_aiter_gfx95 = _use_aiter and _is_gfx95_supported
_use_opt_cat_decode = get_bool_env_var("SGLANG_USE_OPT_CAT")
_use_fused_mla_cat = get_bool_env_var("SGLANG_USE_FUSED_MLA_CAT")
_use_fused_rmsnorm_rope = get_bool_env_var("SGLANG_USE_FUSED_RMSNORM_ROPE")
_use_fused_rms_quant = get_bool_env_var("SGLANG_USE_FUSED_RMS_QUANT")
_rms_quant_path = get_int_env_var("SGLANG_USE_RMS_QUANT_PATH")

# Optional lightop import (only when fused rmsnorm+rope enabled).
fused_rms_norm_rope_contiguous = None
if _use_fused_rmsnorm_rope:
    try:
        from lightop import fused_rms_norm_rope_contiguous as _fused_rms_norm_rope

        fused_rms_norm_rope_contiguous = torch._dynamo.disable(_fused_rms_norm_rope)
    except ImportError:
        logger.warning(
            "SGLANG_USE_FUSED_RMSNORM_ROPE=1 but lightop is not installed; "
            "fused path disabled."
        )
        _use_fused_rmsnorm_rope = False

FORWARD_ABSORB_CORE_ATTENTION_BACKENDS = [
    "fa3",
    "nsa",
    "flashinfer",
    "cutlass_mla",
    "trtllm_mla",
    "ascend",
]

dtype_mapping = {
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e4m3fnuz: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
    torch.float8_e5m2fnuz: "fp8_e5m2",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

# Names patched onto sglang.srt.models.deepseek_v2 after apply_hooks().
MODULE_PATCH_ATTRS = (
    "_is_dcu",
    "_is_hip",
    "_is_cuda",
    "_is_npu",
    "_is_cpu",
    "_is_cpu_amx_available",
    "_device_sm",
    "_is_gfx95_supported",
    "_is_sbo_enabled",
    "_use_aiter",
    "_use_aiter_gfx95",
    "_use_lightop_moe_sum_mul_add",
    "_use_fused_silu_mul_quant",
    "_use_opt_cat_decode",
    "_use_fused_mla_cat",
    "_use_fused_rmsnorm_rope",
    "_use_fused_rms_quant",
    "_rms_quant_path",
)
