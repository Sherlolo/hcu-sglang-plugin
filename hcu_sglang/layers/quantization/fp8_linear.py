# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import deepgemm
from sglang.srt.layers.quantization import fp8_utils as N
from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.layers.quantization.fp8_fallback import _apply_fallback_scaled_mm
from hcu_sglang.layers.quantization.fp8_scaled_mm import triton_scaled_mm

_is_hcu = is_hcu()
_sglang_enable_torch_compile = N.get_bool_env_var("SGLANG_ENABLE_TORCH_COMPILE")


def apply_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: N.Optional[torch.Tensor] = None,
    input_scale_ub: N.Optional[torch.Tensor] = None,
    bias: N.Optional[torch.Tensor] = None,
    cutlass_fp8_supported: bool = N.cutlass_fp8_supported(),
    use_per_token_if_dynamic: bool = False,
    pad_output: N.Optional[bool] = None,
    compressed_tensor_quant: bool = False,
    pre_quant_output_dtype: N.Optional[torch.dtype] = None,
) -> torch.Tensor:
    # Note: we pad the input because torch._scaled_mm is more performant
    # for matrices with batch dimension > 16.
    # This could change in the future.
    # We also don't pad when using torch.compile,
    # as it breaks with dynamic shapes.
    if pad_output is None:
        pad_output = not cutlass_fp8_supported and not _sglang_enable_torch_compile
    output_padding = 17 if pad_output else None

    if _is_hcu and isinstance(input, tuple):
        # HCU: the caller already produced an (fp8 activation, scale) pair, so
        # skip quantization entirely and run the DeepGEMM FP8 GEMM directly.
        output_shape = [*input[0].shape[:-1], weight.shape[1]]
        qinput = input[0].view(-1, input[0].shape[-1])
        x_scale = input[1]
        if x_scale.dim() > 2:
            x_scale = x_scale.view(-1, x_scale.shape[-1])
        output = torch.empty((qinput.shape[0], weight.shape[1]), device=qinput.device, dtype=pre_quant_output_dtype or torch.bfloat16)
        deepgemm.fp8_gemm((qinput, x_scale), (weight, weight_scale), output)
        if bias is not None:
            output += bias

        return output.view(*output_shape)

    # View input as 2D matrix for fp8 methods
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[1]]

    # A pre-quantized fp8 activation (e.g. from a fused RMSNorm+quant kernel)
    # carries no original dtype: skip re-quant, reuse its supplied per-tensor
    # or per-token input scale, and emit ``pre_quant_output_dtype`` (the model's
    # activation dtype, propagated by the producer) or bf16 if omitted.
    input_prequantized = input_2d.dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    )
    if input_prequantized:
        output_dtype = pre_quant_output_dtype or torch.bfloat16
    else:
        output_dtype = input.dtype

    channelwise_cutlass = (
        cutlass_fp8_supported and weight_scale.numel() == weight.shape[1]
    )
    cutlass_compatible_b = weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0
    use_cutlass_channelwise_gemm = (
        channelwise_cutlass and cutlass_compatible_b and not N.use_triton_w8a8_fp8_kernel
    )
    # Consider a tuned Triton tile only where the shape would otherwise go to
    # CUTLASS (that is the path the offline sweep tuned against). On by default;
    # SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE=0 is the kill switch.
    use_tuned_triton_channelwise = (
        use_cutlass_channelwise_gemm and N.envs.SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE.get()
    )
    native_scalar_a_scale = use_cutlass_channelwise_gemm and (
        N.get_platform().is_sm90 or N.get_platform().is_sm100 or N.get_platform().is_sm120
    )

    if input_prequantized:
        assert input_scale is not None and input_scale.numel() in (
            1,
            input_2d.shape[0],
        )
        qinput = input_2d
        if (
            input_scale.numel() == 1
            and channelwise_cutlass
            and not native_scalar_a_scale
        ):
            # Unsupported CUTLASS epilogues require one A scale per row.
            x_scale = input_scale.repeat(input_2d.shape[0]).view(-1, 1)
        else:
            x_scale = input_scale
    elif compressed_tensor_quant:
        # Maybe apply padding to output, see comment in __init__
        num_token_padding = output_padding
        if channelwise_cutlass or (N._is_xpu and weight_scale.numel() == weight.shape[1]):
            # On XPU, sgl-kernel-xpu's native quant kernels require output_q
            # to exactly match input's shape; padded output isn't supported.
            num_token_padding = None
        # For static per-tensor activation scales when using inductor compiler,
        # use pure PyTorch ops instead of the opaque sgl_kernel quant kernel.
        # Inductor fuses these with surrounding ops (RMSNorm, residual add),
        # eliminating a separate kernel launch per linear layer.
        # weight_scale shape does not matter here -- it is only used in the
        # GEMM epilogue, not in the activation quant fusion. Only activates when
        # cuda_graph_config[prefill].tc_compiler=inductor; eager PCG and
        # decode both use the faster custom kernel.

        if (
            input_scale is not None
            and input_scale.numel() == 1
            and N.get_exec().graph.cuda_graph_config.prefill.tc_compiler == "inductor"
        ):
            qinput = (
                (input_2d * input_scale.reciprocal())
                .clamp(min=N.fp8_min, max=N.fp8_max)
                .to(N.fp8_dtype)
            )
            x_scale = input_scale
        else:
            qinput, x_scale = N.scaled_fp8_quant(
                input_2d,
                input_scale,
                num_token_padding=num_token_padding,
                use_per_token_if_dynamic=use_per_token_if_dynamic,
            )
        if (
            input_scale is not None
            and channelwise_cutlass
            and not native_scalar_a_scale
        ):
            x_scale = input_scale.repeat(input_2d.shape[0]).view(-1, 1)
    else:
        if input_scale is not None:
            assert input_scale.numel() == 1
            qinput, x_scale = N.static_quant_fp8(
                input_2d,
                input_scale,
                repeat_scale=channelwise_cutlass and not native_scalar_a_scale,
            )
        else:
            # default use per-token quantization if dynamic
            if N._is_cuda:
                qinput, x_scale = N.sglang_per_token_quant_fp8(input_2d)
            else:
                # TODO(kkhuang): temporarily enforce per-tensor activation scaling if weight is per-tensor scaling
                # final solution should be: 1. add support to per-tensor activation scaling.
                # 2. solve the torch.compile error from weight_scale.numel() == 1 and x_scale.numel() > 1 (below line#308)
                if N._is_hip and weight_scale.numel() == 1:
                    qinput, x_scale = N.scaled_fp8_quant(
                        input_2d,
                        input_scale,
                        use_per_token_if_dynamic=use_per_token_if_dynamic,
                    )
                else:
                    qinput, x_scale = N.per_token_group_quant_fp8(
                        input_2d, group_size=input_2d.shape[1]
                    )

    if _is_hcu:
        if x_scale.dim() > 2:
            x_scale = x_scale.view(-1, x_scale.shape[-1])
        output = torch.empty((qinput.shape[0], weight.shape[1]), device=input.device, dtype=output_dtype)
        deepgemm.fp8_gemm((qinput, x_scale), (weight, weight_scale), output)
        output = torch.narrow(output, 0, 0, input_2d.shape[0])
        if bias is not None:
            output += bias

        return output.view(*output_shape)

    if channelwise_cutlass:
        # A tuned config exists only for shapes where tuned Triton beat the
        # CUTLASS dispatch on this GPU; otherwise this is None and the backend
        # choice below is unchanged. weight is [K, N] here.
        tuned_config = (
            N.get_w8a8_channelwise_fp8_config(
                N=weight.shape[1], K=weight.shape[0], M=qinput.shape[0]
            )
            if use_tuned_triton_channelwise
            else None
        )
        if not use_cutlass_channelwise_gemm:
            # Massage the input to be 2D
            qinput = qinput.view(-1, qinput.shape[-1])
            output = triton_scaled_mm(
                qinput, weight, x_scale, weight_scale, output_dtype, bias
            )
        elif tuned_config is not None:
            qinput = qinput.view(-1, qinput.shape[-1])
            output = triton_scaled_mm(
                qinput,
                weight,
                x_scale,
                weight_scale,
                output_dtype,
                bias,
                block_size_m=tuned_config["BLOCK_SIZE_M"],
                block_size_n=tuned_config["BLOCK_SIZE_N"],
                block_size_k=tuned_config["BLOCK_SIZE_K"],
                use_heuristic=False,
                num_warps=tuned_config["num_warps"],
                num_stages=tuned_config["num_stages"],
            )
        else:
            output = N.fp8_scaled_mm(
                qinput,
                weight,
                x_scale,
                weight_scale,
                out_dtype=output_dtype,
                bias=bias,
            )
        return output.view(*output_shape)

    # torch.scaled_mm supports per tensor weights + activations only
    # so fallback to naive if per channel or per token
    per_tensor_weights = weight_scale.numel() == 1
    # When the number of token is 1,
    # per-token scale has shape (1, 1), per-tensor scale has shape (1) or ().
    per_tensor_activations = (x_scale.numel() == 1) and x_scale.dim() < 2

    if (
        use_per_token_if_dynamic
        and not per_tensor_weights
        and not per_tensor_activations
        and (N.USE_ROWWISE_TORCH_SCALED_MM or N._use_aiter)
    ):
        # into this sector means use dynamic per-token-per-channel quant
        # per-token scale quant for input matrix, every row(one token) have one scale factor
        # per-channel scale quant for weight matrix, every col(one channel) have one scale factor
        # Must agree with the load-time predicate that decides whether the
        # weight was pre-shuffled; an unshuffled weight through the aiter path
        # (or a shuffled one through torch._scaled_mm) silently returns garbage.
        if N.use_aiter_bpreshuffle_gemm(weight.shape[1]):
            # gemm_a8w8_bpreshuffle(XQ, WQ, x_scale, w_scale, dtype)
            # XQ -> input tensor, shape = (m, k)
            # WQ -> weight tensor, shape = (n, k), with preshuffe get better perf
            # x_scale -> input scale tensor, shape = (m, 1)
            # w_scale -> weight scale tensor, shape = (n ,1)
            # dtype -> output dtype
            gemm = (
                N.smallm_fp8_gemm
                if N.smallm_fp8_gemm_supported(qinput, weight.T, x_scale, output_dtype)
                else N.gemm_a8w8_bpreshuffle
            )
            output = gemm(
                XQ=qinput,
                WQ=weight.T,
                x_scale=x_scale,
                w_scale=weight_scale,
                dtype=output_dtype,
            )
            if bias is not None:
                output += bias
            return N._process_scaled_mm_output(output, input_2d.shape, output_shape)
        else:
            # For now validated on ROCm platform
            # fp8 rowwise scaling in torch._scaled_mm is introduced in
            # https://github.com/pytorch/pytorch/pull/144432 using hipBLASLt
            # and ROCm 6.3, which only exists in torch 2.7 and above.
            # For CUDA platform please validate if the
            # torch._scaled_mm support rowwise scaled GEMM
            # Fused GEMM_DQ Rowwise GEMM
            output = torch._scaled_mm(
                qinput,
                weight,
                out_dtype=output_dtype,
                scale_a=x_scale,
                scale_b=weight_scale.t(),
                bias=bias,
            )
            return N._process_scaled_mm_output(output, input_2d.shape, output_shape)

    if per_tensor_weights and per_tensor_activations:
        # Fused GEMM_DQ; _scaled_mm with torch.compile requires len(weight_scale.shape) == len(x_scale.shape)
        if weight_scale.ndim == 0 and x_scale.ndim == 1:
            weight_scale = weight_scale.unsqueeze(0)
        output = torch._scaled_mm(
            qinput,
            weight,
            out_dtype=output_dtype,
            scale_a=x_scale,
            scale_b=weight_scale,
            bias=bias,
        )
        return N._process_scaled_mm_output(output, input_2d.shape, output_shape)

    # Fallback for channelwise case, where we use unfused DQ
    # due to limitations with scaled_mm

    # Symmetric quantized GEMM by definition computes the following:
    #   C = (s_x * X) (s_w * W) + bias
    # This is equivalent to dequantizing the weights and activations
    # before applying a GEMM.
    #
    # In order to compute quantized operands, a quantized kernel
    # will rewrite the above like so:
    #   C = s_w * s_x * (X * W) + bias
    #
    # For the scaled_mm fallback case, we break this down, since it
    # does not support s_w being a vector.
    return _apply_fallback_scaled_mm(
        qinput,
        weight,
        x_scale,
        weight_scale,
        input_2d.shape,
        output_shape,
        bias,
        output_dtype,
    )


def around_apply_fp8_linear(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return apply_fp8_linear(*args, **kwargs)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8_utils.apply_fp8_linear",
        around_apply_fp8_linear, HookType.AROUND,
    )
