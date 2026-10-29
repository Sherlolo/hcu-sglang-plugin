"""DCU DeepSeek V2 fork methods (auto-extracted for plugin hooks)."""

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2023-2024 SGLang Team
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

# Adapted from:
# https://github.com/vllm-project/vllm/blob/fb6af8bc086328ca6659e72d11ffd4309ce4de22/vllm/model_executor/models/deepseek_v2.py
from __future__ import annotations

from hcu_sglang.models.deepseek.deepseek_v2 import constants as C
from sglang.srt import utils as U
import sglang.srt.models.deepseek_v2 as N0


def forward(
    self,
    x,
    forward_batch=None,
    gemm_output_zero_allocator: N0.BumpAllocator = None,
    input_quant_args: N0.Optional[N0.Tuple[N0.torch.Tensor, N0.torch.Tensor]] = None,
    gateup_pre_quant: N0.Optional[N0.Tuple[N0.torch.Tensor, N0.torch.Tensor]] = None,
):
    if self.is_replicated and x.shape[0] == 0:
        return x

    if (
        getattr(self, "_enable_nvfp4_gemm_swiglu_fusion", False)
        and self.swiglu_limit is None
        and not isinstance(x, tuple)
    ):
        from sglang.kernels.ops.quantization.nvfp4_gemm_swiglu_nvfp4_quant import (
            nvfp4_gemm_swiglu_nvfp4_quant,
        )
        from sglang.srt.layers.quantization.fp4_utils import fp4_quantize

        x_fp4, x_scale = fp4_quantize(
            x, self.gate_up_proj.input_scale_inv, enable_pdl=True
        )
        out_fp4, out_scale = nvfp4_gemm_swiglu_nvfp4_quant(
            x_fp4,
            x_scale,
            self.gate_up_proj.weight_swiglu_interleaved,
            self.gate_up_proj.weight_scale_swiglu_interleaved,
            self.gate_up_proj.alpha,
            self.down_proj.input_scale_inv,
            enable_pdl=True,
        )
        out, _ = self.down_proj((out_fp4, out_scale))
        return out

    if input_quant_args is not None:
        if gateup_pre_quant is not None:
            raise ValueError(
                "INT8 and FP8 prequantized MLP inputs are mutually exclusive"
            )
        gate_up, _ = self.gate_up_proj(x, input_quant_args=input_quant_args)
    elif gateup_pre_quant is not None:
        # SGLANG_OPT_MOE_QUANT_ONCE: reuse the caller's per-token-group-128
        # fp8 (q, scale) of x for the gate_up GEMM instead of re-quantizing
        # inside the fp8 linear method. q rows may be padded to a multiple
        # of 4; the caller slices the MLP output back.
        gate_up, _ = self.gate_up_proj(gateup_pre_quant)
    else:
        if (
            gemm_output_zero_allocator is not None
            and x.shape[0] <= 256
            and getattr(self.gate_up_proj, "weight", None) is not None
            and self.gate_up_proj.weight.dtype == N0.torch.uint8
        ):
            y = gemm_output_zero_allocator.allocate(
                x.shape[0] * self.gate_up_proj.output_size_per_partition
            ).view(x.shape[0], self.gate_up_proj.output_size_per_partition)
            x = (x, None, y)

        gate_up, _ = self.gate_up_proj(x)
    if (
        self.swiglu_limit is not None
        and gate_up.dim() == 2
        and gate_up.is_contiguous()
        and self.down_proj.supports_fused_silu_mul_fp8_quant_input()
    ):
        output, _ = self.down_proj(
            gate_up,
            use_fused_silu_mul_fp8_quant=True,
            swiglu_limit=float(self.swiglu_limit),
        )
        return output

    if (
        self.swiglu_limit is not None
        and gate_up.dim() == 2
        and gate_up.is_contiguous()
        and self.down_proj.supports_fused_silu_mul_quant_input()
    ):
        output, _ = self.down_proj(
            gate_up,
            use_fused_silu_mul_quant=True,
            swiglu_limit=float(self.swiglu_limit),
        )
        return output

    # Fast path: fused silu+clamp+fp8_quant+deepgemm when conditions met.
    # Only valid when down_proj does NOT need an all-reduce and its weights
    # are fp8 (uint8 storage with weight_scale_inv).
    if (
        self.swiglu_limit is not None
        and self.allow_fused_down
        and getattr(self.down_proj, "weight", None) is not None
        and self.down_proj.weight.dtype == N0.torch.uint8
        and hasattr(self.down_proj, "weight_scale_inv")
    ):
        M, N = gate_up.shape
        down_input_fp8 = gate_up.new_empty((M, N // 2), dtype=N0.torch.float8_e4m3fn)
        scale_block_size = 128
        down_input_scale = N0.create_per_token_group_quant_fp8_output_scale(
            x_shape=(M, N // 2),
            device=gate_up.device,
            group_size=scale_block_size,
            column_major_scales=N0.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_tma_aligned=N0.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_ue8m0=N0.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        N0.silu_and_mul_contig_post_quant(
            input=gate_up,
            output=down_input_fp8,
            output_scale=down_input_scale,
            quant_group_size=scale_block_size,
            scale_ue8m0=N0.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            transposed=N0.deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            swiglu_limit=float(self.swiglu_limit),
        )
        down_output = gate_up.new_empty(
            (M, self.down_proj.output_size), dtype=N0.torch.bfloat16
        )
        N0.deep_gemm_wrapper.gemm_nt_f8f8bf16(
            (down_input_fp8, down_input_scale),
            (self.down_proj.weight, self.down_proj.weight_scale_inv),
            down_output,
        )
        return down_output

    if self.use_fused_clamp_act_mul and not self._fused_clamp_fp8_checked:
        N0._hip_act.resolve_fused_clamp_route(self, gate_up.shape[-1] // 2)

    # SGLANG_HIP_SHARED_ACT_MXFP8: an MXFP8 down_proj takes fp8 + ue8m0 straight from the Triton
    # epilogue instead of aiter's bf16 output plus a separate quant launch.
    if (
        self.use_fused_clamp_act_mul
        and self.swiglu_limit is not None
        and N0._hip_shared_act_mxfp8
        and self._hip_act_fp8_grid
    ):
        x = N0._hip_act.silu_and_mul_clamp(self, gate_up)
    elif self.use_fused_clamp_act_mul and self.swiglu_limit is not None:
        from aiter.ops.triton.fusions.fused_clamp_act_mul import (
            fused_clamp_act_mul,
        )

        if self._fused_clamp_use_fp8:
            from aiter import dtypes

            _emit_bpre = N0.emit_transposed_bpreshuffle_scale(
                gate_up.shape[0], on_bpreshuffle_gfx95=N0._use_aiter_bpreshuffle_gfx95
            )
            x_fp8, x_scale = fused_clamp_act_mul(
                gate_up,
                swiglu_limit=self.swiglu_limit,
                activation="silu",
                dtype_quant=dtypes.fp8,
                transpose_scale=_emit_bpre,
            )
            if _emit_bpre:
                x_scale = N0.view_aiter_fused_rms_transposed_fp8_scale(x_scale)
            elif N0._use_aiter_bpreshuffle_gfx95:
                x_scale = N0.materialize_bpreshuffle_fp8_scale(x_scale)
            x = (x_fp8, x_scale)
        else:
            x = fused_clamp_act_mul(
                gate_up,
                swiglu_limit=self.swiglu_limit,
                activation="silu",
            )

    # Fallback: fused silu+clamp kernel (still faster than unfused)
    elif self.swiglu_limit is not None:
        if N0._is_hip and not C._is_dcu:
            x = N0._hip_act.silu_and_mul_clamp(self, gate_up)
        elif N0._is_npu:
            x = N0.torch.ops.npu.npu_clipped_swiglu(
                gate_up,
                alpha=1,
                limit=self.swiglu_limit,
                bias=0,
                interleaved=False,
            )
        else:
            M, N = gate_up.shape
            x = gate_up.new_empty((M, N // 2))
            N0.silu_and_mul_clamp(gate_up, x, float(self.swiglu_limit))
    else:
        x = self.act_fn(gate_up)
    x, _ = self.down_proj(x)
    return x


def __init__(
    self,
    hidden_size: int,
    intermediate_size: int,
    hidden_act: str,
    quant_config: N0.Optional[N0.QuantizationConfig] = None,
    reduce_results: bool = True,
    prefix: str = "",
    swiglu_limit: N0.Optional[float] = None,
    allow_fused_down: N0.Optional[bool] = None,
    *,
    parallel_group: N0.LinearParallelGroup = "tp",
) -> None:
    super(N0.DeepseekV2MLP, self).__init__()
    self.is_replicated = parallel_group == "replicated"
    self.swiglu_limit = swiglu_limit
    # The fused down path calls deepgemm directly and runs no all-reduce.
    self.allow_fused_down = (
        not reduce_results if allow_fused_down is None else allow_fused_down
    )

    self.gate_up_proj = N0.MergedColumnParallelLinear(
        hidden_size,
        [intermediate_size] * 2,
        bias=False,
        quant_config=quant_config,
        prefix=N0.add_prefix("gate_up_proj", prefix),
        parallel_group=parallel_group,
    )
    self.down_proj = N0.RowParallelLinear(
        intermediate_size,
        hidden_size,
        bias=False,
        quant_config=quant_config,
        reduce_results=reduce_results,
        prefix=N0.add_prefix("down_proj", prefix),
        parallel_group=parallel_group,
    )
    if not hasattr(self.gate_up_proj, "weight") and hasattr(
        self.gate_up_proj, "weight_packed"
    ):
        self.gate_up_proj.weight = self.gate_up_proj.weight_packed
    if not hasattr(self.down_proj, "weight") and hasattr(
        self.down_proj, "weight_packed"
    ):
        self.down_proj.weight = self.down_proj.weight_packed
    if hidden_act != "silu":
        raise ValueError(
            f"Unsupported activation: {hidden_act}. Only silu is supported for now."
        )
    self.act_fn = N0.SiluAndMul()
    self.use_fused_clamp_act_mul = (
        N0._is_hip and not C._is_dcu and N0.envs.SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL.get()
    )
    self._fused_clamp_fp8_checked = False
    self._fused_clamp_use_fp8 = False
