"""DCU DeepSeek V4 fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from hcu_sglang.models.deepseek.deepseek_v4 import constants as C
from hcu_sglang.models.deepseek.deepseek_v4.methods._runtime import *  # noqa: F403

def hc_pre(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: Optional[nn.Module] = None,
        forward_batch: Optional[ForwardBatch] = None,
    ):
        """If *norm* is given and the TileLang path is active, the returned
        hidden_states are already post-norm (the norm is fused into the kernel)."""

        @compile_in_capture_mode
        def hc_pre_torch_impl(x, hc_fn):
            x_flat = x.flatten(1).float()
            rsqrt = torch.rsqrt(
                x_flat.square().mean(-1, keepdim=True) + self.rms_norm_eps
            )
            mixes = (F.linear(x_flat, hc_fn) * rsqrt).unsqueeze(1)
            return x_flat, mixes

        shape, dtype = x.size(), x.dtype

        if _is_npu:
            return _get_mhc_ops().npu_hc_pre(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                hc_mult=self.hc_mult,
                hc_sinkhorn_iters=self.hc_sinkhorn_iters,
                rms_norm_eps=self.rms_norm_eps,
                hc_eps=self.hc_eps,
                forward_batch=forward_batch,
            )

        if x.shape[0] == 0:
            y = torch.empty((0, shape[-1]), dtype=dtype, device=x.device)
            post = torch.empty((0, self.hc_mult), dtype=torch.float32, device=x.device)
            comb = torch.empty(
                (0, self.hc_mult, self.hc_mult), dtype=torch.float32, device=x.device
            )
            return y, post, comb, False

        if envs.SGLANG_OPT_USE_FLASHINFER_MHC.get():
            y, post, comb = _flashinfer_hc_pre(
                x,
                hc_fn,
                hc_scale,
                hc_base,
                rms_eps=self.rms_norm_eps,
                hc_eps=self.hc_eps,
                sinkhorn_iters=self.hc_sinkhorn_iters,
            )
            return y, post, comb, False

        if envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get():
            from sglang.kernels.ops.layernorm.mhc import mhc_pre

            norm_kwargs = {}
            if norm is not None:
                norm_kwargs["norm_weight"] = norm.weight.data
                norm_kwargs["norm_eps"] = norm.variance_epsilon

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=_MHC_POST_MULT_VALUE,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
                **norm_kwargs,
            )
            # The HCU compatibility path inside mhc_pre dispatches to AITER's
            # pre_big_fuse_tilelang, which computes MHC pre but does not fuse
            # the decoder RMSNorm.  Keep the validated v0.5.15.post1_dev
            # contract so the caller still applies input_layernorm on HCU.
            norm_fused = norm is not None and not (
                C._is_dcu and C._use_aiter_tilelang_mhc
            )
            return y, post.squeeze(-1), comb, norm_fused

        if _is_hip and envs.SGLANG_OPT_USE_AITER_MHC_PRE.get():
            from aiter.ops.mhc import mhc_pre

            post, comb, y = mhc_pre(
                residual=x,
                fn=hc_fn,
                hc_scale=hc_scale,
                hc_base=hc_base,
                rms_eps=self.rms_norm_eps,
                hc_pre_eps=self.hc_eps,
                hc_sinkhorn_eps=self.hc_eps,
                hc_post_mult_value=_MHC_POST_MULT_VALUE,
                sinkhorn_repeat=self.hc_sinkhorn_iters,
            )
            return y, post.squeeze(-1), comb, False

        if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
            from sglang.srt.layers.deep_gemm_wrapper.entrypoint import (
                tf32_hc_prenorm_gemm,
            )

            x_flat = x.flatten(1).bfloat16()

            m, k = x_flat.shape
            mix_hc = hc_fn.size(0)
            d_out = torch.empty((m, mix_hc), dtype=torch.float, device=x.device)
            s_out = torch.empty((m,), dtype=torch.float, device=x.device)
            tf32_hc_prenorm_gemm(
                x_flat, hc_fn.float().contiguous(), d_out, s_out, num_splits=None
            )
            rsqrt = torch.rsqrt(s_out / k + self.rms_norm_eps)
            mixes = (d_out * rsqrt.unsqueeze(1)).unsqueeze(1)
        else:
            x_flat, mixes = hc_pre_torch_impl(x, hc_fn)

        pre, post, comb = _get_mhc_ops().hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        # y is the post-norm activation fed into the MoE. Allocate it in the
        # symmetric memory pool so the downstream all-reduce uses the low-latency
        # NCCL symmetric path: the Triton inplace MoE runner writes the expert
        # output back into this buffer, so a symmetric input yields a symmetric
        # all-reduce input. Gated by is_allocation_symmetric() (mirrors the
        # TileLang path in _mhc_pre_impl / mhc_fused_post_pre).
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            y = (pre.squeeze(1).unsqueeze(-1) * x_flat.view(shape)).sum(dim=1).to(dtype)
        return y, post.squeeze(1), comb.squeeze(1), False

def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ):

        if x.shape[0] == 0:
            return torch.empty(
                (0, self.hc_mult, x.shape[-1]), dtype=x.dtype, device=x.device
            )

        if _is_npu:
            return torch.ops.custom.npu_hc_post(x, residual, post, comb)

        if envs.SGLANG_OPT_USE_FLASHINFER_MHC.get():
            from flashinfer.mhc import mhc_post

            return mhc_post(x, residual, post, comb)

        if envs.SGLANG_OPT_USE_TILELANG_MHC_POST.get():
            # Keep the HCU implementation validated on v0.5.15.post1_dev.
            # Generic ROCm/CUDA platforms use the upstream kernel below.
            if C._is_dcu and C._use_aiter_tilelang_mhc:
                from aiter.ops.tilelang import mhc_post_fwd

                return mhc_post_fwd(x, residual, post, comb)

            from sglang.kernels.ops.layernorm.mhc import mhc_post

            return mhc_post(x, residual, post, comb)

        elif _is_hip and envs.SGLANG_OPT_USE_AITER_MHC_POST.get():
            from aiter.ops.mhc import mhc_post

            result = torch.empty_like(residual)
            mhc_post(result, x, residual, post, comb)
            return result

        assert residual.shape == (x.shape[0], self.hc_mult, x.shape[-1])
        assert post.shape == (x.shape[0], self.hc_mult)
        assert comb.shape == (x.shape[0], self.hc_mult, self.hc_mult)

        @compile_in_capture_mode
        def hc_post_torch_impl(x, residual, post, comb):
            return (
                post.unsqueeze(-1) * x.unsqueeze(1)
                + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
            ).type_as(x)

        return hc_post_torch_impl(x, residual, post, comb)

def forward(
        self,
        positions: torch.tensor,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        forward_batch: ForwardBatch,
        input_ids_global: torch.Tensor,
        prev_residual: Optional[torch.Tensor] = None,
        prev_post: Optional[torch.Tensor] = None,
        prev_comb: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        use_fused = self.use_fused_mhc_post_pre

        if prev_residual is not None and use_fused:
            residual, post, comb, hidden_states = _get_mhc_ops().mhc_fused_post_pre(
                hidden_states,
                prev_residual,
                prev_post,
                prev_comb,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.rms_norm_eps,
                self.hc_eps,
                self.hc_eps,
                _MHC_POST_MULT_VALUE,
                self.hc_sinkhorn_iters,
                norm_weight=(
                    self._input_layernorm_weight_bf16
                    if self._input_layernorm_weight_bf16 is not None
                    else self.input_layernorm.weight.data
                ),
                norm_eps=self.input_layernorm.variance_epsilon,
            )
            x_quant = None
        else:
            residual = hidden_states
            hidden_states, post, comb, norm_fused = self.hc_pre(
                hidden_states,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                norm=self.input_layernorm,
                forward_batch=forward_batch,
            )
            if not norm_fused:
                if _use_aiter and _is_gfx95_supported:
                    x_quant, hidden_states = _fused_rmsnorm_fp8_quant(
                        hidden_states,
                        self.input_layernorm.weight,
                        self.rms_norm_eps,
                    )
                else:
                    hidden_states = self.input_layernorm(hidden_states)
                    x_quant = None
            else:
                x_quant = None

        with self.self_attn.maybe_use_decode_attn_tp(forward_batch):
            hidden_states = self.self_attn(
                x=hidden_states,
                positions=positions,
                forward_batch=forward_batch,
                x_quant=x_quant,
            )

        if use_fused:
            fused_mhc = try_fused_hc_post_pre(
                hidden_states,
                residual,
                post,
                comb,
                self.hc_ffn_fn.T,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                self.hc_mult,
                self.rms_norm_eps,
                self.hc_eps,
                _MHC_POST_MULT_VALUE,
                self.hc_sinkhorn_iters,
                _is_gfx95_supported,
            )
            if fused_mhc is not None:
                residual, hidden_states, post, comb, norm_fused = fused_mhc
            else:
                residual, post, comb, hidden_states = _get_mhc_ops().mhc_fused_post_pre(
                    hidden_states,
                    residual,
                    post.unsqueeze(-1) if post.ndim == 2 else post,
                    comb,
                    self.hc_ffn_fn,
                    self.hc_ffn_scale,
                    self.hc_ffn_base,
                    self.rms_norm_eps,
                    self.hc_eps,
                    self.hc_eps,
                    _MHC_POST_MULT_VALUE,
                    self.hc_sinkhorn_iters,
                    norm_weight=(
                        self._post_attention_layernorm_weight_bf16
                        if self._post_attention_layernorm_weight_bf16 is not None
                        else self.post_attention_layernorm.weight.data
                    ),
                    norm_eps=self.post_attention_layernorm.variance_epsilon,
                )
                norm_fused = True
        else:
            hidden_states = self.hc_post(hidden_states, residual, post, comb)
            residual = hidden_states
            hidden_states, post, comb, norm_fused = self.hc_pre(
                hidden_states,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                norm=self.post_attention_layernorm,
                forward_batch=forward_batch,
            )
            if not norm_fused:
                hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self._run_moe_ffn_dp_sync(
            hidden_states,
            forward_batch,
            input_ids=input_ids,
            input_ids_global=input_ids_global,
        )

        if not use_fused:
            hidden_states = self.hc_post(hidden_states, residual, post, comb)
            return hidden_states, None, None, None

        # Return the deferred FFN hc_post state; the next layer consumes it with
        # cross-layer fusion, and the final layer is completed in DeepseekV4Model.
        return hidden_states, residual, post, comb
