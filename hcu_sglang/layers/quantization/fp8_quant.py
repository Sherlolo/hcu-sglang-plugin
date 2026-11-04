# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# Copyright 2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

from typing import Optional
import torch
from lightop.quant import per_token_quant_fp8 as lightop_per_token_quant_fp8
from sglang.kernels.ops.quantization import fp8_kernel as native_quant
from hcu_sglang.utils.detection import is_hcu


def _native_dynamic_per_token_quant_fp8(output, input, scale):
    """Native PyTorch fallback for dynamic per-token FP8 quantization when AITER is disabled."""
    M, N = input.shape
    eps = 1e-12
    # Compute per-token scale
    absmax = input.abs().max(dim=1, keepdim=True).values
    absmax = torch.clamp(absmax, min=eps)
    scale_val = absmax / native_quant.fp8_max
    scale[: input.shape[0]].copy_(scale_val)
    # Quantize
    output_data = torch.clamp(input / scale_val, native_quant.fp8_min, native_quant.fp8_max).to(native_quant.fp8_dtype)
    output[: input.shape[0]].copy_(output_data)


def _native_dynamic_per_tensor_quant_fp8(output, input, scale):
    """Native PyTorch fallback for dynamic per-tensor FP8 quantization when AITER is disabled."""
    eps = 1e-12
    absmax = input.abs().max()
    absmax = torch.clamp(absmax, min=eps)
    scale_val = absmax / native_quant.fp8_max
    # Use copy_ instead of fill_ with .item() to avoid CPU-GPU sync
    scale.view(-1).copy_(scale_val.view(-1))
    # Quantize
    output_data = torch.clamp(input / scale_val, native_quant.fp8_min, native_quant.fp8_max).to(native_quant.fp8_dtype)
    output[: input.shape[0]].copy_(output_data)


def _native_static_quant_fp8(output, input, scale):
    """Native PyTorch fallback for static FP8 quantization when AITER is disabled."""
    # Use tensor directly instead of .item() to avoid CPU-GPU sync
    output_data = torch.clamp(input / scale, native_quant.fp8_min, native_quant.fp8_max).to(native_quant.fp8_dtype)
    output[: input.shape[0]].copy_(output_data)


def scaled_fp8_quant(
    input: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    num_token_padding: Optional[int] = None,
    use_per_token_if_dynamic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert input.ndim == 2, f"Expected 2D input tensor, got {input.ndim}D"
    shape = input.shape
    if num_token_padding:
        shape = (max(num_token_padding, input.shape[0]), shape[1])
    output = torch.empty(shape, device=input.device, dtype=native_quant.fp8_dtype)

    if scale is None:
        # Dynamic scaling
        if use_per_token_if_dynamic:
            # The LightOp path quantizes the unpadded input. Keep its output
            # and per-token scale shapes aligned with the real token count;
            # returning a padded scale with an unpadded qinput would give
            # downstream GEMM operands inconsistent M dimensions.
            output = torch.empty_like(input, device=input.device, dtype=native_quant.fp8_dtype)
            scale = torch.empty(
                (input.shape[0], 1),
                device=input.device,
                dtype=torch.float32,
            )
            if not input.is_contiguous():
                input = input.contiguous()
            output, scale = lightop_per_token_quant_fp8(
                input,
                dtype=output.dtype,
                out_q=output,
                out_scale=scale,
            )
        else:
            scale = torch.zeros(1, device=input.device, dtype=torch.float32)
            _native_dynamic_per_tensor_quant_fp8(output, input, scale)
    else:
        # Static scaling
        assert scale.numel() == 1, (
            f"Expected scalar scale, got numel={scale.numel()}"
        )
        _native_static_quant_fp8(output, input, scale)

    return output, scale


def around_scaled_fp8_quant(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return scaled_fp8_quant(*args, **kwargs)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register(
        "sglang.kernels.ops.quantization.fp8_kernel.scaled_fp8_quant",
        around_scaled_fp8_quant, HookType.AROUND,
    )
