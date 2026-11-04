from __future__ import annotations

import logging
import torch
from sglang.srt.layers.quantization import fp8_utils as N
from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.layers.quantization.fp8_scaled_mm import triton_scaled_mm

logger = logging.getLogger(__name__)
_is_hcu = is_hcu()
TORCH_DEVICE_IDENTITY = None
_HIPBLASLT_UNSUPPORTED_FP8_SHAPES: set[tuple] = set()


def _apply_fallback_scaled_mm(
    qinput,
    weight,
    x_scale,
    weight_scale,
    input_2d_shape,
    output_shape,
    bias,
    input_dtype,
):
    global TORCH_DEVICE_IDENTITY
    if TORCH_DEVICE_IDENTITY is None:
        TORCH_DEVICE_IDENTITY = torch.ones(1, dtype=torch.float32, device=weight.device)

    # hipBLASLt does not provide solutions for every channel-FP8 matrix shape
    # on BW1100 (for example MiniMax-M3 TP8 decode: M=17, K=6144, N=768).
    # Probe each runtime shape once, then keep it on the numerically equivalent
    # Triton rowwise/channelwise scaled GEMM instead of crashing every rank.
    hip_shape_key = (
        qinput.device.type,
        qinput.device.index,
        qinput.shape[0],
        weight.shape[1],
        qinput.shape[1],
        qinput.dtype,
        input_dtype,
    )
    use_triton_fallback = _is_hcu and (
        hip_shape_key in _HIPBLASLT_UNSUPPORTED_FP8_SHAPES
    )
    if not use_triton_fallback:
        try:
            output = torch._scaled_mm(
                qinput,
                weight,
                scale_a=TORCH_DEVICE_IDENTITY,
                scale_b=TORCH_DEVICE_IDENTITY,
                out_dtype=torch.float32,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if not _is_hcu or "blaslt" not in message:
                raise
            _HIPBLASLT_UNSUPPORTED_FP8_SHAPES.add(hip_shape_key)
            logger.warning(
                "hipBLASLt has no channel-FP8 solution for M=%s, N=%s, K=%s; "
                "using Triton scaled_mm for this shape.",
                qinput.shape[0],
                weight.shape[1],
                qinput.shape[1],
            )
            use_triton_fallback = True

    if use_triton_fallback:
        output = triton_scaled_mm(
            qinput,
            weight,
            x_scale,
            weight_scale,
            input_dtype,
            bias,
        )
        return N._process_scaled_mm_output(output, input_2d_shape, output_shape)

    output = N._process_scaled_mm_output(
        output, input_2d_shape, (input_2d_shape[0], output_shape[-1])
    )
    x_scale = torch.narrow(x_scale, 0, 0, input_2d_shape[0])

    output = output * x_scale * weight_scale.t()
    if bias is not None:
        output = output + bias
    return output.to(dtype=input_dtype).view(*output_shape)
