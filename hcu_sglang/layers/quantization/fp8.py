"""HCU FP8 quantization adaptations."""

from __future__ import annotations

import torch


def around_fallback_scaled_mm(
    original,
    qinput,
    weight,
    x_scale,
    weight_scale,
    input_2d_shape,
    output_shape,
    bias,
    input_dtype,
):
    from hcu_sglang.utils.detection import is_hcu

    if not is_hcu():
        return original(qinput, weight, x_scale, weight_scale,
                        input_2d_shape, output_shape, bias, input_dtype)
    from hcu_sglang.layers.quantization.fp8_fallback import _apply_fallback_scaled_mm
    return _apply_fallback_scaled_mm(qinput, weight, x_scale, weight_scale,
                                     input_2d_shape, output_shape, bias, input_dtype)


def around_rms_norm_forward_hip(
    original, self, x, residual=None, post_residual_addition=None, quant_linear=None
):
    from hcu_sglang.utils.detection import is_hcu

    if is_hcu():
        from hcu_sglang.layers.norm_current import rms_forward_hip
        return rms_forward_hip(self, x, residual, post_residual_addition, quant_linear)
    return original(self, x, residual, post_residual_addition, quant_linear)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8_utils._apply_fallback_scaled_mm",
        around_fallback_scaled_mm,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.layernorm.RMSNorm.forward_hip",
        around_rms_norm_forward_hip,
        HookType.AROUND,
    )

    from hcu_sglang.utils.detection import is_hcu
    if is_hcu():
        from hcu_sglang.layers.norm_current import register_hooks as register_norm_hooks
        register_norm_hooks()

    from . import fp8_moe
    fp8_moe.register_hooks()
