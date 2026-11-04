# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations
import inspect
from typing import Optional, Tuple, Union
import torch
from sglang.srt.layers.quantization import fp8_utils as U
from sglang.srt.layers.quantization.compressed_tensors.schemes import compressed_tensors_w8a8_fp8 as Q
from sglang.srt.utils import get_bool_env_var
from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.layers.quantization import fp8_block as B

_is_hcu = is_hcu()
_use_lightop_channel_fp8 = get_bool_env_var("SGLANG_USE_LIGHTOP_CHANNEL_FP8") and _is_hcu
if _use_lightop_channel_fp8:
    from lightop.gemm_ops import hipblaslt_w8a8_channelwise_gemm, hipblaslt_w8a8_channelwise_gemm_kme
    from lightop.quant.fp8 import per_token_quant_fp8 as lightop_per_token_quant_fp8

def apply_fp8_lightop_channelwise_linear(
    input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run dynamic-token/channel-weight FP8 GEMM through LightOp on ROCm.

    LightOp's BW1100 channelwise kernels consume OCP E4M3 weights in ``[N, K]``
    layout. This deliberately differs from the native gfx94/FNUZ path, which
    stores transposed FNUZ weights for ``torch._scaled_mm``/Triton.
    """
    if not _use_lightop_channel_fp8:
        raise RuntimeError(
            "SGLANG_USE_LIGHTOP_CHANNEL_FP8=1 is required for this FP8 path"
        )

    if isinstance(input, tuple):
        qinput, x_scale = input[0], input[1]
        output_dtype = input[2] if len(input) > 2 else torch.bfloat16
        input_shape = qinput.shape
        qinput = qinput.view(-1, qinput.shape[-1])
        # A fused producer on gfx94 may emit FNUZ. Re-encode the quantized
        # values as OCP E4M3; its scale remains valid because conversion
        # preserves represented values.
        if qinput.dtype != torch.float8_e4m3fn:
            qinput = qinput.to(torch.float8_e4m3fn)
    else:
        input_shape = input.shape
        output_dtype = input.dtype
        input_2d = input.view(-1, input.shape[-1]).contiguous()
        qinput, x_scale = lightop_per_token_quant_fp8(
            input_2d, dtype=torch.float8_e4m3fn
        )

    m, k = qinput.shape
    n = weight.shape[0]
    if weight.shape[1] != k:
        raise ValueError(
            f"LightOp channel-FP8 expects weight [N,K], got {weight.shape} "
            f"for input [M,K]={qinput.shape}"
        )

    try:
        _, output = hipblaslt_w8a8_channelwise_gemm(
            qinput,
            weight,
            x_scale,
            weight_scale,
            m,
            n,
            k,
            "NT",
            output_dtype,
            bias,
        )
    except RuntimeError:
        # KME has a separate algorithm search/dispatch and covers some shapes
        # not present in the ordinary hipBLASLt table.
        _, output = hipblaslt_w8a8_channelwise_gemm_kme(
            qinput,
            weight,
            x_scale,
            weight_scale,
            m,
            n,
            k,
            "NT",
            output_dtype,
            bias,
        )

    output_shape = [*input_shape[:-1], n]
    return output.reshape(m, n).view(*output_shape)

def supports_fp8_prequantized_input(self) -> bool:
    """Whether this scheme can consume LightOp's per-token FP8 pair.

        The HCU DeepGEMM tuple path expects channelwise weights in transposed
        layout and dynamic per-token activation scales.  Keep the capability
        narrow so block/static schemes and AITER's shuffled weight layout stay
        on their existing quantization path.
        """

    return bool(
        _is_hcu
        and not Q._use_aiter
        and self.strategy == Q.QuantizationStrategy.CHANNEL
        and not self.is_static_input_scheme
        and self.weight_block_size is None
    )

def process_weights_after_loading(self, layer) -> None:
    if self.strategy == Q.QuantizationStrategy.TENSOR:
        max_w_scale, weight = Q.requantize_with_max_scale(
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            logical_widths=layer.logical_widths,
        )

        if Q.is_fp8_fnuz():
            input_scale = getattr(layer, "input_scale", None)

            weight, max_w_scale, input_scale = Q.normalize_e4m3fn_to_e4m3fnuz(
                weight=weight, weight_scale=max_w_scale, input_scale=input_scale
            )
            if input_scale is not None:
                layer.input_scale = Q.Parameter(input_scale, requires_grad=False)
        layer.weight = Q.Parameter(weight.t(), requires_grad=False)
        layer.weight_scale = Q.Parameter(max_w_scale, requires_grad=False)

    elif self.strategy == Q.QuantizationStrategy.CHANNEL:
        weight = layer.weight

        if _use_lightop_channel_fp8:
            # LightOp BW1100 kernels use OCP E4M3 in [N,K] layout. Keep
            # checkpoint weights intact instead of converting to FNUZ.
            weight_scale = layer.weight_scale.data
        elif Q.is_fp8_fnuz():
            input_scale = getattr(layer, "input_scale", None)

            weight, weight_scale, input_scale = Q.normalize_e4m3fn_to_e4m3fnuz(
                weight=weight,
                weight_scale=layer.weight_scale,
                input_scale=input_scale,
            )
            if input_scale is not None:
                layer.input_scale = Q.Parameter(input_scale, requires_grad=False)
        else:
            weight_scale = layer.weight_scale.data

        if _use_lightop_channel_fp8:
            layer.weight = Q.Parameter(weight.data.contiguous(), requires_grad=False)
        elif Q._use_aiter:
            # keep the weight as (N, K)
            layer.weight = Q.Parameter(
                Q.shuffle_weight(weight, (16, 16)), requires_grad=False
            )
        else:
            layer.weight = Q.Parameter(weight.t(), requires_grad=False)

        # required by torch.compile to be torch.nn.Parameter
        layer.weight_scale = Q.Parameter(weight_scale, requires_grad=False)

    elif self.strategy == Q.QuantizationStrategy.BLOCK:
        assert self.is_static_input_scheme is False
        if Q.is_fp8_fnuz():
            weight, weight_scale, _ = Q.normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.weight, weight_scale=layer.weight_scale
            )
            layer.weight = Q.Parameter(weight.data, requires_grad=False)
            layer.weight_scale = Q.Parameter(weight_scale.data, requires_grad=False)
            layer.weight_scale.format_ue8m0 = False
        else:
            layer.weight.requires_grad_(False)
            layer.weight_scale.requires_grad_(False)

        # On Blackwell, block-FP8 dispatches to DeepGEMM, which needs the
        # weight scales UE8M0-packed to match its UE8M0 activation scales.
        use_deepgemm_runner = (
            self.w8a8_block_fp8_linear
            is Q.deepgemm_w8a8_block_fp8_linear_with_fallback
        )
        Q.requant_block_scale_ue8m0_for_deepgemm(
            layer.weight,
            layer.weight_scale,
            self.weight_block_size,
            use_deepgemm_runner=use_deepgemm_runner,
            output_dtype=getattr(layer, "orig_dtype", None),
            weight_shape=layer.weight.shape,
        )

        if self.w8a8_block_fp8_linear is B.hipblaslt_w8a8_block_fp8_linear:
            layer.weight.data = layer.weight.data.T.contiguous()
            layer.weight_scale.data = layer.weight_scale.data.T.contiguous()

    else:
        raise ValueError(f"Unknown quantization strategy {self.strategy}")

    # INPUT SCALE
    if self.is_static_input_scheme and hasattr(layer, "input_scale"):
        layer.input_scale = Q.Parameter(layer.input_scale.max(), requires_grad=False)
    else:
        layer.input_scale = None

def apply_weights(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_quant_args: Optional[list[torch.Tensor]] = None,
    silu_quant_args: Optional[list[torch.Tensor]] = None,
) -> torch.Tensor:
    if _use_lightop_channel_fp8 and self.strategy == Q.QuantizationStrategy.CHANNEL:
        return apply_fp8_lightop_channelwise_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            bias=bias,
        )

    if isinstance(x, tuple):
        # Pre-quantized activation from a fused RMSNorm+FP8 quant kernel:
        # x = (fp8_input, per_tensor_input_scale[, orig_dtype]).
        # apply_fp8_linear detects the fp8 dtype and skips re-quantizing;
        # orig_dtype (when present) sets the GEMM output dtype.
        qx, x_scale = x[0], x[1]
        out_dtype = x[2] if len(x) > 2 else None
        if _is_hcu:
            # HCU's fused SwiGLU producer emits one dynamic scale per
            # token. Preserve the pair so apply_fp8_linear selects its
            # DeepGEMM prequantized-input path instead of the generic
            # scalar-scale path (which only accepts scale.numel() == 1).
            return Q.apply_fp8_linear(
                input=(qx, x_scale),
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                bias=bias,
                pre_quant_output_dtype=out_dtype,
            )
        return Q.apply_fp8_linear(
            input=qx,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=x_scale,
            bias=bias,
            use_per_token_if_dynamic=True,
            compressed_tensor_quant=True,
            pre_quant_output_dtype=out_dtype,
        )
    if self.weight_block_size is not None:
        return self.w8a8_block_fp8_linear(
            input=x,
            weight=layer.weight,
            block_size=self.weight_block_size,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
        )

    if Q._use_aiter and self.strategy == Q.QuantizationStrategy.CHANNEL:
        return Q.apply_fp8_ptpc_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            use_per_token_if_dynamic=True,
            compressed_tensor_quant=True,
        )
    else:
        return Q.apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=layer.input_scale,
            bias=bias,
            use_per_token_if_dynamic=True,
            compressed_tensor_quant=True,
        )

def linear_apply(
    self,
    layer: torch.nn.Module,
    x: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_quant_args: Optional[list[torch.Tensor]] = None,
    silu_quant_args: Optional[list[torch.Tensor]] = None,
):
    """
        Use the output of create_weights and the CompressedTensorsScheme
        associated with the layer to apply the forward pass with the
        layer input.  See LinearMethodBase for param details

        """

    scheme = layer.scheme
    if scheme is None:
        raise ValueError("A scheme must be defined for each layer")
    apply_kwargs = {
        "layer": layer,
        "x": x,
        "bias": bias,
    }

    supported_kwargs = getattr(scheme, "_apply_weights_supported_kwargs", None)
    if supported_kwargs is None:
        supported_kwargs = frozenset(
            inspect.signature(scheme.apply_weights).parameters
        )
        scheme._apply_weights_supported_kwargs = supported_kwargs

    if "input_quant_args" in supported_kwargs and input_quant_args is not None:
        apply_kwargs["input_quant_args"] = input_quant_args
    if "silu_quant_args" in supported_kwargs and silu_quant_args is not None:
        apply_kwargs["silu_quant_args"] = silu_quant_args

    return scheme.apply_weights(**apply_kwargs)

_SCHEME_APPLY_PARAMETERS = frozenset(inspect.signature(apply_weights).parameters) - {"self"}


def around_linear_apply(original, self, layer, x, bias=None, input_quant_args=None, silu_quant_args=None):
    if not is_hcu():
        return original(self, layer, x, bias)
    if type(layer.scheme) is Q.CompressedTensorsW8A8Fp8:
        layer.scheme._apply_weights_supported_kwargs = _SCHEME_APPLY_PARAMETERS
    return linear_apply(self, layer, x, bias, input_quant_args, silu_quant_args)


def around_process_weights(original, self, layer):
    if not is_hcu():
        return original(self, layer)
    return process_weights_after_loading(self, layer)


def around_apply_weights(original, self, layer, x, bias=None, input_quant_args=None, silu_quant_args=None):
    if not is_hcu():
        return original(self, layer, x, bias)
    return apply_weights(self, layer, x, bias, input_quant_args, silu_quant_args)


_PREQUANTIZED_CAPABILITY = property(supports_fp8_prequantized_input)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    if not is_hcu():
        return
    previous = getattr(U, "apply_fp8_lightop_channelwise_linear", apply_fp8_lightop_channelwise_linear)
    if previous is not apply_fp8_lightop_channelwise_linear:
        raise RuntimeError("Native LightOp channelwise FP8 helper already exists; re-review the source contract")
    U.apply_fp8_lightop_channelwise_linear = apply_fp8_lightop_channelwise_linear
    cls = Q.CompressedTensorsW8A8Fp8
    previous = cls.__dict__.get("supports_fp8_prequantized_input", _PREQUANTIZED_CAPABILITY)
    if previous is not _PREQUANTIZED_CAPABILITY:
        raise RuntimeError("Native compressed FP8 capability already exists; re-review the source contract")
    cls.supports_fp8_prequantized_input = _PREQUANTIZED_CAPABILITY
    HookRegistry.register(
        "sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8.CompressedTensorsW8A8Fp8.process_weights_after_loading",
        around_process_weights, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8.CompressedTensorsW8A8Fp8.apply_weights",
        around_apply_weights, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors.CompressedTensorsLinearMethod.apply",
        around_linear_apply, HookType.AROUND,
    )
