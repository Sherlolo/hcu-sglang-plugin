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
from typing import Callable, List, Optional
import torch
from torch.nn import Module
from lightop import gemm_ops as quant_ops
from lightop.gemm_ops import BlockSize
from lightop.quant import per_token_group_quant_fp8 as per_token_group_quant_fp8_hcu
from sglang.srt.layers.quantization import fp8_utils as N, fp8 as P
from hcu_sglang.utils.detection import is_hcu

_is_hcu = is_hcu()

def _dispatch_auto_backend() -> Callable:
    """Auto-select the best backend based on hardware capabilities."""
    # Priority order for auto selection:
    # 1. DeepGEMM (if enabled and available)
    # 2. FlashInfer TRTLLM (if Blackwell GPU and FlashInfer available)
    # 3. CUTLASS (if SM120 GPU and CUDA 12.8+)
    # 4. AITER (if AMD GPU with AITER enabled)
    # 5. NPU (Ascend)
    # 6. XPU (Intel GPU, PyTorch torch._scaled_mm)
    # 7. Triton (fallback)

    if N.deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
        return N.deepgemm_w8a8_block_fp8_linear_with_fallback
    elif N.get_platform().is_blackwell and N.is_flashinfer_available():
        return N.flashinfer_gemm_w8a8_block_fp8_linear_with_fallback
    elif N.get_platform().is_sm120:
        return N.cutlass_w8a8_block_fp8_linear_with_fallback
    elif N._use_aiter:
        return N.aiter_w8a8_block_fp8_linear
    elif _is_hcu:
        return hipblaslt_w8a8_block_fp8_linear
    elif N.is_npu_arch35():
        from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
            npu_w8a8_mxfp8_linear,
        )

        return npu_w8a8_mxfp8_linear
    elif N._is_xpu:
        return N.torch_w8a8_block_fp8_linear
    else:
        return N.triton_w8a8_block_fp8_linear

def hipblaslt_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[1]]
    q_input, input_scale = per_token_group_quant_fp8_hcu(
        input_2d, block_size[1], column_major_scales=False
    )
    enum_block_size = BlockSize.block_128x128

    # if hasattr(self, "block_size") and self.block_size[0] == 64:
    #     enum_block_size = BlockSize.block_64x64

    output = hipblaslt_w8a8_block_fp8_matmul(
        A=q_input,
        B=weight,
        As=input_scale,
        Bs=weight_scale,
        block_size=enum_block_size,
        output_dtype=input_2d.dtype,
    )
    if bias is not None:
        output += bias
    return output.to(dtype=input_2d.dtype).view(*output_shape)

def hipblaslt_w8a8_block_fp8_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    block_size: BlockSize,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    assert A.shape[1] == B.shape[0]
    m, k = A.shape
    _, n = B.shape
    _, d = quant_ops.hipblaslt_w8a8_blockwise_gemm(
        A, B, As, Bs, m, n, k, "NN", output_dtype, block_size, None
    )

    return d

def process_weights_after_loading_block_quant(self, layer: Module) -> None:
    use_deepgemm_runner = (
        getattr(self.w8a8_block_fp8_linear, "func", self.w8a8_block_fp8_linear)
        is P.deepgemm_w8a8_block_fp8_linear_with_fallback
    )
    if self.convert_mxfp8_to_block:
        from sglang.srt.layers.quantization.mxfp8_block_convert import (
            convert_mxfp8_weight_to_block_fp8,
            dequant_mxfp8_2d_to_bf16,
        )

        mx_weight, mx_scale = layer.weight.data, layer.weight_scale_inv.data
        qweight, scale = convert_mxfp8_weight_to_block_fp8(
            mx_weight, mx_scale, block=128
        )
        layer.weight = P.Parameter(qweight, requires_grad=False)
        if (
            P._use_aiter
            and P._is_gfx95_supported
            and self.w8a8_block_fp8_linear is P.aiter_w8a8_block_fp8_linear
        ):
            # rowwise-fp8 copy for the small-M path of aiter_w8a8_block_fp8_linear;
            # the later bpreshuffle is an in-place copy_, so these attrs survive
            weight_fp32 = dequant_mxfp8_2d_to_bf16(mx_weight, mx_scale).float()
            fp8_max = torch.finfo(torch.float8_e4m3fn).max
            row_scale = (
                weight_fp32.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                / fp8_max
            )
            layer.weight._ptpc_weight = P.shuffle_weight(
                (weight_fp32 / row_scale)
                .clamp(-fp8_max, fp8_max)
                .to(torch.float8_e4m3fn),
                (16, 16),
            )
            layer.weight._ptpc_scale = row_scale
        layer.weight_scale_inv = P.Parameter(scale, requires_grad=False)
        self.use_mxfp8 = False
        self.convert_mxfp8_to_block = False
        self.weight_block_size = [128, 128]
    elif self.use_mxfp8:
        # MXFP8 (e4m3fn + UE8M0) must NOT be fnuz-normalized; check before
        # the fnuz branch since is_fp8_fnuz() is also True on gfx942.
        if not self.is_checkpoint_fp8_serialized:
            self._quantize_mxfp8_weights(layer)
            return
        # MXFP8 scales are stored as UE8M0 uint8; no requantization here.
        # Keep parameter object to preserve weight_loader attrs for hot reload.
        layer.weight_scale_inv.requires_grad_(False)
        layer.weight_scale_inv.format_ue8m0 = True
        self._process_mxfp8_linear_weight_scale(layer)
        return
    elif P._is_npu and P.is_npu_arch35():
        from sglang.srt.hardware_backend.npu.quantization.w8a8_mxfp8 import (
            process_npu_arch35_mxfp8_linear_weights,
        )

        # UE8M0 checkpoints only need re-layout; plain block-FP8 ones
        # (fp32 block scales) get requantized inside. Either way the
        # layer ends up in the MXFP8 layout for npu_w8a8_mxfp8_linear.
        process_npu_arch35_mxfp8_linear_weights(
            layer,
            self.weight_block_size,
            scale_fmt=getattr(self.quant_config, "scale_fmt", None),
        )
        return
    # If ROCm, normalize the weights and scales to e4m3fnuz
    if P._is_fp8_fnuz:
        # activation_scheme: dynamic
        weight, weight_scale, _ = P.normalize_e4m3fn_to_e4m3fnuz(
            weight=layer.weight,
            weight_scale=layer.weight_scale_inv,
            input_scale=None,
        )
        layer.input_scale = None
    elif P._is_cpu:
        assert P._is_cpu_amx_available, (
            "Fp8LinearMethod on CPU requires that CPU has AMX support"
        )
        P._amx_process_weight_after_loading(layer, ["weight"])
        layer.weight_scale_inv = torch.nn.Parameter(
            layer.weight_scale_inv.data, requires_grad=False
        )
        return
    else:
        # Requantize block scales to UE8M0 when DeepGEMM is the active runner.
        P.requant_block_scale_ue8m0_for_deepgemm(
            layer.weight,
            layer.weight_scale_inv,
            getattr(self.quant_config, "weight_block_size", None),
            use_deepgemm_runner=use_deepgemm_runner,
            output_dtype=getattr(layer, "orig_dtype", None),
            weight_shape=layer.weight.shape,
        )
        weight, weight_scale = layer.weight.data, layer.weight_scale_inv.data

    from hcu_sglang.layers.quantization.fp8_block import (
        hipblaslt_w8a8_block_fp8_linear,
    )

    if self.w8a8_block_fp8_linear is hipblaslt_w8a8_block_fp8_linear:
        weight = weight.T.contiguous()
        weight_scale = weight_scale.T.contiguous()
    layer.weight.data = weight.data
    layer.weight_scale_inv.data = weight_scale.data
    if hasattr(layer, "_block_fp8_bf16_weight"):
        layer._block_fp8_bf16_weight = None
        layer._derived_weight_cache_error = None
    if (
        P._is_cuda
        and P.get_platform().is_sm90
        and P.envs.SGLANG_OPT_HOPPER_BLOCK_FP8_BF16.get()
        and not (
            use_deepgemm_runner
            and P._deepgemm_block_fp8_supported(
                weight.shape,
                self.weight_block_size,
                getattr(layer, "orig_dtype", None),
            )
        )
        and weight.is_cuda
        and weight.dtype == torch.float8_e4m3fn
        and self.weight_block_size == [32, 32]
        and getattr(self.quant_config, "scale_fmt", None) == "ue8m0"
        and getattr(layer, "orig_dtype", None) == torch.bfloat16
        and not self.use_marlin
        and not getattr(layer, "keep_plain_weight_layout", False)
    ):
        from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
        from sglang.srt.layers.quantization.fp8_utils import block_quant_dequant

        # Keep even the smallest/largest finite E4M3 value representable
        # as normal BF16, and reject mislabeled non-power-of-two scales.
        exact_bf16_scales = (
            (weight_scale >= 2.0**-117)
            & (weight_scale <= 2.0**119)
            & (torch.frexp(weight_scale)[0] == 0.5)
        ).all()
        if not is_batch_invariant_mode_enabled() and exact_bf16_scales.item():
            # Hopper group32 FP8 accumulates each 32-wide block separately.
            # Reuse a BF16 weight expansion for larger GEMMs, keeping the
            # original FP8 weights for decode and weight reloads.
            layer.register_buffer(
                "_block_fp8_bf16_weight",
                block_quant_dequant(weight, weight_scale, [32, 32], torch.bfloat16),
                persistent=False,
            )
            layer._derived_weight_cache_error = (
                "Online weight updates are not supported while Hopper FP8 "
                "BF16 weight caches are active: captured CUDA graphs retain "
                "these derived weights. Restart with "
                "SGLANG_OPT_HOPPER_BLOCK_FP8_BF16=0 to allow online updates."
            )
    if self.block_fp8_as_mxfp8:
        self._prepare_block_fp8_as_mxfp8(layer)

    # The preshuffle rewrites the weight into a layout only
    # aiter_w8a8_block_fp8_linear can read, so it is correct exactly when
    # this quant method is what consumes the weight. A layer whose weight is
    # read directly by the model (DeepSeek-V4 wo_a, whose absorb GEMM takes
    # .weight/.weight_scale_inv and runs its own batched kernel) sets
    # keep_plain_weight_layout and keeps the plain row-major layout.
    if (
        P._use_aiter_bpreshuffle_gfx95
        and self.w8a8_block_fp8_linear is P.aiter_w8a8_block_fp8_linear
        and not getattr(layer, "keep_plain_weight_layout", False)
    ):
        n, k = layer.weight.shape
        if not P.use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k):
            # TODO(1am9trash), to deal with case that this branch chance
            # drops as use_aiter_triton_gemm_w8a8_tuned_gfx950() expands
            t = P.shuffle_weight(layer.weight, (16, 16))
            layer.weight.copy_(t)
            del t
            # The shuffle is in place and preserves shape, dtype and
            # strides, so nothing downstream can tell it happened. Record
            # it so a consumer that needs the row-major layout can assert
            # instead of silently reading a permuted weight.
            layer.aiter_bpreshuffled = True
            layer.weight.is_shuffled = True
            self._maybe_prepare_dense_mxscale(layer)

    if (
        P.is_xpu()
        and self.w8a8_block_fp8_linear is P.torch_w8a8_block_fp8_linear
        and self.weight_block_size in ([1, 128], [128, 128])
        and layer.weight_scale_inv.ndim == 2
    ):
        # Keep the checkpoint's logical [N-blocks, K-blocks] shape, but use
        # transpose-contiguous storage. For [1, 128], scaled_mm transposes
        # scale_b internally; for [128, 128], the wrapper passes scale_b.t().
        # This avoids a per-forward contiguous/copy in either path.
        scale = layer.weight_scale_inv.data
        scale_b_is_contiguous = scale.t().is_contiguous()
        if not scale_b_is_contiguous:
            scale_reordered = torch.empty_strided(
                scale.shape,
                (1, scale.shape[0]),
                dtype=scale.dtype,
                device=scale.device,
            )
            scale_reordered.copy_(scale)
            with torch.no_grad():
                layer.weight_scale_inv.set_(scale_reordered)

def _dispatch_explicit_backend(backend: N.Fp8GemmRunnerBackend) -> Callable:
    """Dispatch based on explicitly selected backend."""
    if backend.is_flashinfer_trtllm():
        if not (N.get_platform().is_sm100 and N.is_flashinfer_available()):
            raise RuntimeError(
                "FlashInfer FP8 GEMM requested via --fp8-gemm-backend=flashinfer_trtllm, "
                "but FlashInfer is not available or not supported on this hardware. "
                "FlashInfer TRTLLM FP8 GEMM requires SM100/SM103 GPUs and FlashInfer."
            )
        return N.flashinfer_gemm_w8a8_block_fp8_linear_with_fallback

    elif backend.is_flashinfer_cutlass():
        if not (N.get_platform().is_blackwell and N.is_flashinfer_available()):
            raise RuntimeError(
                "FlashInfer FP8 GEMM requested via --fp8-gemm-backend=flashinfer_cutlass, "
                "but FlashInfer is not available or not supported on this hardware. "
                "FlashInfer CUTLASS FP8 GEMM requires Blackwell GPUs and FlashInfer."
            )
        return N.flashinfer_gemm_w8a8_block_fp8_linear_with_fallback

    elif backend.is_flashinfer_deepgemm():
        if not (N.get_platform().is_sm90 and N.is_flashinfer_available()):
            raise RuntimeError(
                "FlashInfer DeepGEMM with swapAB requested via --fp8-gemm-backend=flashinfer_deepgemm, "
                "but it's not available. This backend requires Hopper (SM90) GPUs and FlashInfer "
                "to be installed."
            )
        return N.flashinfer_deepgemm_w8a8_block_fp8_linear_with_fallback

    elif backend.is_cutlass():
        if not N.get_platform().is_sm120:
            raise RuntimeError(
                "--fp8-gemm-backend=cutlass is deprecated on this hardware. "
                "Please switch to DeepGEMM or FlashInfer TRTLLM on SM90/SM100."
            )
        return N.cutlass_w8a8_block_fp8_linear_with_fallback

    elif backend.is_aiter():
        if not N._use_aiter:
            aiter_platform = "HCU devices" if _is_hcu else "AMD GPUs"
            raise RuntimeError(
                "AITER backend requested via --fp8-gemm-backend=aiter, "
                f"but AITER is not available. AITER requires {aiter_platform} with "
                "SGLANG_USE_AITER=1 environment variable set."
            )
        return N.aiter_w8a8_block_fp8_linear

    elif backend.is_deep_gemm():
        if not N.deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            raise RuntimeError(
                "DeepGEMM backend requested via --fp8-gemm-backend=deep_gemm, "
                "but DeepGEMM is not available. This usually means the deep_gemm package "
                "is not installed or has been disabled via SGLANG_ENABLE_JIT_DEEPGEMM=0."
            )
        return N.deepgemm_w8a8_block_fp8_linear_with_fallback

    elif backend.is_triton():
        return N.triton_w8a8_block_fp8_linear

    else:
        raise ValueError(f"Unknown FP8 GEMM backend: {backend}")

def torch_fp8_block_gemm(
    a,
    b,
    scale_a,
    scale_b,
    block_size,
    out_dtype=torch.bfloat16,
):

    a_cp = a.clone().to(torch.float32)
    b_cp = b.clone().to(torch.float32)
    scale_a_cp = scale_a.clone().to(torch.float32)
    scale_b_cp = scale_b.clone().to(torch.float32)

    m, k = a.shape
    _, n = b.shape

    k_block = torch.arange(k, device=a.device) // block_size
    n_block = torch.arange(n, device=a.device) // block_size

    scale_a_full = scale_a_cp[:, k_block]
    scale_b_full = scale_b_cp[k_block][:, n_block]

    a_scaled = a_cp * scale_a_full
    b_scaled = b_cp * scale_b_full

    c = a_scaled @ b_scaled
    return c.to(out_dtype)

def around_explicit_backend(original, backend):
    if not is_hcu():
        return original(backend)
    return _dispatch_explicit_backend(backend)


def around_auto_backend(original):
    if not is_hcu():
        return original()
    return _dispatch_auto_backend()


def around_process_block_weights(original, self, layer):
    if not is_hcu() or self.w8a8_block_fp8_linear not in (hipblaslt_w8a8_block_fp8_linear, getattr(P, "aiter_w8a8_block_fp8_linear", None)):
        return original(self, layer)
    return process_weights_after_loading_block_quant(self, layer)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8_utils._dispatch_auto_backend",
        around_auto_backend, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8.Fp8LinearMethod.process_weights_after_loading_block_quant",
        around_process_block_weights, HookType.AROUND,
    )
    if is_hcu():
        previous = getattr(N, "torch_fp8_block_gemm", torch_fp8_block_gemm)
        if previous is not torch_fp8_block_gemm:
            raise RuntimeError("Native FP8 block reference already exists; re-review the source contract")
        N.torch_fp8_block_gemm = torch_fp8_block_gemm
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8_utils._dispatch_explicit_backend",
        around_explicit_backend, HookType.AROUND,
    )
