"""DCU DeepSeek V2 fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from hcu_sglang.models.deepseek.deepseek_v2 import constants as C
from hcu_sglang.models.deepseek.deepseek_v2.methods._runtime import *  # noqa: F403

def forward(
        self,
        x,
        forward_batch=None,
        gemm_output_zero_allocator: BumpAllocator = None,
        gateup_pre_quant: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        if (self.tp_size == 1) and x.shape[0] == 0:
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

        if gateup_pre_quant is not None:
            # SGLANG_OPT_MOE_QUANT_ONCE: reuse the caller's per-token-group-128
            # fp8 (q, scale) of x for the gate_up GEMM instead of re-quantizing
            # inside the fp8 linear method. q rows may be padded to a multiple
            # of 4; the caller slices the MLP output back.
            gate_up, _ = self.gate_up_proj(gateup_pre_quant)
        else:
            if (
                gemm_output_zero_allocator is not None
                and x.shape[0] <= 256
                and self.gate_up_proj.weight.dtype == torch.uint8
            ):
                y = gemm_output_zero_allocator.allocate(
                    x.shape[0] * self.gate_up_proj.output_size_per_partition
                ).view(x.shape[0], self.gate_up_proj.output_size_per_partition)
                x = (x, None, y)

            gate_up, _ = self.gate_up_proj(x)
        # Fast path: fused silu+clamp+fp8_quant+deepgemm when conditions met.
        # Only valid when down_proj does NOT need an all-reduce and its weights
        # are fp8 (uint8 storage with weight_scale_inv).
        if (
            self.swiglu_limit is not None
            and not self.down_proj.reduce_results
            and self.down_proj.weight.dtype == torch.uint8
            and hasattr(self.down_proj, "weight_scale_inv")
        ):
            M, N = gate_up.shape
            down_input_fp8 = gate_up.new_empty((M, N // 2), dtype=torch.float8_e4m3fn)
            scale_block_size = 128
            down_input_scale = create_per_token_group_quant_fp8_output_scale(
                x_shape=(M, N // 2),
                device=gate_up.device,
                group_size=scale_block_size,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
            silu_and_mul_contig_post_quant(
                input=gate_up,
                output=down_input_fp8,
                output_scale=down_input_scale,
                quant_group_size=scale_block_size,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                transposed=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                swiglu_limit=float(self.swiglu_limit),
            )
            down_output = gate_up.new_empty(
                (M, self.down_proj.output_size), dtype=torch.bfloat16
            )
            deep_gemm_wrapper.gemm_nt_f8f8bf16(
                (down_input_fp8, down_input_scale),
                (self.down_proj.weight, self.down_proj.weight_scale_inv),
                down_output,
            )
            return down_output

        if self.use_fused_clamp_act_mul and self.swiglu_limit is not None:
            from aiter.ops.triton.fusions.fused_clamp_act_mul import (
                fused_clamp_act_mul,
            )

            if not self._fused_clamp_fp8_checked:
                from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

                qm = getattr(self.down_proj, "quant_method", None)
                self._fused_clamp_use_fp8 = (
                    isinstance(qm, Fp8LinearMethod) and qm.block_quant
                )
                self._fused_clamp_fp8_checked = True

            if self._fused_clamp_use_fp8:
                from aiter import dtypes

                x_fp8, x_scale = fused_clamp_act_mul(
                    gate_up,
                    swiglu_limit=self.swiglu_limit,
                    activation="silu",
                    dtype_quant=dtypes.fp8,
                    transpose_scale=False,
                )
                if _use_aiter_bpreshuffle_gfx95:
                    x_scale = materialize_bpreshuffle_fp8_scale(x_scale)
                x = (x_fp8, x_scale)
            else:
                x = fused_clamp_act_mul(
                    gate_up,
                    swiglu_limit=self.swiglu_limit,
                    activation="silu",
                )

        # Fallback: fused silu+clamp kernel (still faster than unfused)
        elif self.swiglu_limit is not None:
            if C._is_npu:
                _g, _u = gate_up.chunk(2, dim=-1)
                _lim = float(self.swiglu_limit)
                gate_up = torch.cat(
                    [_g.clamp(max=_lim), _u.clamp(min=-_lim, max=_lim)], dim=-1
                )
                x = self.act_fn(gate_up)
            else:
                M, N = gate_up.shape
                x = gate_up.new_empty((M, N // 2))
                silu_and_mul_clamp(gate_up, x, float(self.swiglu_limit))
        else:
            x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
