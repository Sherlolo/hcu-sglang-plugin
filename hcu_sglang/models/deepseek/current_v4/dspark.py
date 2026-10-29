"""DSpark attention bound to the current HCU MQA base."""
from __future__ import annotations
import sglang.srt.models.deepseek_v4_dspark as upstream
globals().update({name:value for name,value in vars(upstream).items() if not name.startswith('__')})
from .attention import MqaAttentionBase, _is_cuda
from hcu_sglang.mem_cache.hcu_v4_pool import DeepSeekV4TokenToKVPool

class DSparkAttention(MqaAttentionBase):
    def __init__(
        self,
        config: DeepSeekV4Config,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_streams: Optional[List[torch.cuda.Stream]] = None,
    ) -> None:
        super().__init__(
            config,
            layer_id,
            quant_config,
            prefix,
            compress_ratio=0,
            fuse_wqa_wkv=False,
            wo_a_fp8=False,
            wo_a_keeps_quant_config=False,
            wo_b_reduce_results=True,
            rope_original_seq_len=0,
        )
        assert self.compress_ratio == 0, (
            "DSpark draft attention requires compress_ratio == 0."
        )
        self.window_size = int(
            getattr(config, "sliding_window", None) or config.window_size
        )

        self.attn = RadixAttention(
            self.n_local_heads,
            self.head_dim,
            self.softmax_scale,
            num_kv_heads=1,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

        self._use_fast_kernel = envs.SGLANG_DSPARK_FAST_KERNEL.get()
        self.alt_streams = alt_streams
        self._multi_stream_bs_limit = 128 if get_platform().is_blackwell else 64
        if _is_npu:
            self.register_buffer(
                "_q_post_norm_weight",
                torch.ones(self.head_dim),
                persistent=False,
            )

    def kv_proj_only(self, x: torch.Tensor) -> torch.Tensor:
        kv, _ = self.wkv(x)
        return kv

    def _store_block_kv(
        self,
        *,
        kv: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        attn_backend,
        pool: DeepSeekV4TokenToKVPool,
    ) -> None:
        if is_unified_kv_triton():
            # unified_kv: SWA K lives in the shared bf16 ring (swa_kv_pool is
            # None). Use the unified ring write target -- get_unified_swa_loc
            # recomputes it from live positions for multi-step draft decode.
            pool.set_unified_key_buffer_radix_fused_norm_rope(
                layer_id=self.layer_id,
                swa_loc=attn_backend.get_unified_swa_loc(forward_batch),
                kv=kv,
                kv_weight=self.kv_norm.weight.data,
                eps=self.eps,
                freqs_cis=self.freqs_cis,
                positions=positions,
            )
            return
        pool.set_swa_key_buffer_radix_fused_norm_rope(
            layer_id=self.layer_id,
            swa_loc=attn_backend.get_swa_out_cache_loc(forward_batch),
            kv=kv,
            kv_weight=self.kv_norm.weight.data,
            eps=self.eps,
            freqs_cis=self.freqs_cis,
            positions=positions,
        )

    def _compute_q(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        q_out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q, _ = self.wq_a(x)
        q = self.q_norm(q)
        q, _ = self.wq_b(q)
        q = q.view(-1, self.n_local_heads, self.head_dim)
        if not self.q_head_norm:
            if q_out is not None and q_out.dtype == torch.float8_e4m3fn:
                fused_q_norm_rope(q, q_out, None, self.freqs_cis, positions)
                return q_out
            if self._use_fast_kernel and not _is_npu:
                fused_rope_inplace(
                    q[..., -self.rope_head_dim :],
                    None,
                    self.freqs_cis,
                    positions=positions,
                )
            else:
                apply_rotary_emb(
                    q[..., -self.rope_head_dim :], self.freqs_cis[positions]
                )
            if q_out is None:
                return q
            q_out.copy_(q)
            return q_out
        if self._use_fast_kernel:
            if q_out is None:
                q_out = torch.empty_like(q)
            fused_q_norm_rope(q, q_out, self.eps, self.freqs_cis, positions)
            return q_out
        else:
            if _is_npu:
                import torch_npu

                from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import (
                    Dsv4NpuRoPE,
                )

                q = torch_npu.npu_rms_norm(q, self._q_post_norm_weight, self.eps)[0]
                cos4, sin4 = Dsv4NpuRoPE.for_freqs(self.freqs_cis).get_cos_sin(
                    positions,
                    q.dtype,
                    view_4d=True,
                    inverse=False,
                    allow_build=True,
                    cache_dtype=torch.float32,
                )
                Dsv4NpuRoPE.apply_rotary_mul_inplace(
                    q,
                    None,
                    cos4,
                    sin4,
                    qk_nope_dim=q.shape[-1] - self.rope_head_dim,
                )
            else:
                q = q * torch.rsqrt(
                    q.float().square().mean(-1, keepdim=True) + self.eps
                ).to(q.dtype)
                apply_rotary_emb(
                    q[..., -self.rope_head_dim :], self.freqs_cis[positions]
                )
            if q_out is not None:
                q_out.copy_(q)
                return q_out
            return q

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        defer_all_reduce: bool = False,
    ) -> torch.Tensor:
        # Idle ranks have no draft KV rows or attention metadata, but still
        # participate in the shared MoE forward later in the draft stage.
        if forward_batch.forward_mode.is_idle():
            return torch.zeros_like(hidden_states)

        from sglang.srt.model_executor.forward_context import get_attn_backend

        pool = _resolve_dspark_pool()
        attn_backend = get_attn_backend()
        rd = self.rope_head_dim

        enable_multi_stream = (
            self.alt_streams is not None
            and get_is_capture_mode()
            and hidden_states.shape[0] <= self._multi_stream_bs_limit
        )

        q_padded: Optional[torch.Tensor] = None
        q_out: Optional[torch.Tensor] = None
        if (
            pool.uniform_fp8
            and not self.q_head_norm
            and self.n_local_heads in (8, 16, 32, 64, 128)
        ):
            q_out = torch.empty(
                (hidden_states.shape[0], self.n_local_heads, self.head_dim),
                dtype=torch.float8_e4m3fn,
                device=hidden_states.device,
            )
        elif self.n_local_heads < _PAD_NUM_HEADS:
            q_padded = hidden_states.new_empty(
                hidden_states.shape[0], _PAD_NUM_HEADS, self.head_dim
            )
            q_out = q_padded[:, : self.n_local_heads, :]

        if enable_multi_stream:
            current_stream = torch.cuda.current_stream()
            stream_kv = self.alt_streams[0]
            stream_kv.wait_stream(current_stream)
            with torch.cuda.stream(stream_kv):
                kv = self.kv_proj_only(hidden_states)
                self._store_block_kv(
                    kv=kv,
                    positions=positions,
                    forward_batch=forward_batch,
                    attn_backend=attn_backend,
                    pool=pool,
                )
            q = self._compute_q(hidden_states, positions, q_out=q_out)
            current_stream.wait_stream(stream_kv)
        else:
            kv = self.kv_proj_only(hidden_states)
            self._store_block_kv(
                kv=kv,
                positions=positions,
                forward_batch=forward_batch,
                attn_backend=attn_backend,
                pool=pool,
            )
            q = self._compute_q(hidden_states, positions, q_out=q_out)

        if q_padded is not None:
            q = q_padded
        attn_sink = self._local_attn_sink()

        o = attn_backend.forward(
            q=q,
            k=kv,
            v=kv,
            layer=self.attn,
            forward_batch=forward_batch,
            compress_ratio=0,
            attn_sink=attn_sink,
            save_kv_cache=False,
        )

        if o.shape[1] != self.n_local_heads:
            o = o[:, : self.n_local_heads, :]

        if self._use_fast_kernel:
            fused_rope_inplace(
                o[..., -rd:], None, self.freqs_cis, positions=positions, inverse=True
            )
        elif _is_npu:
            from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE

            cos4, sin4 = Dsv4NpuRoPE.for_freqs(self.freqs_cis).get_cos_sin(
                positions,
                o.dtype,
                view_4d=True,
                inverse=True,
                allow_build=True,
                cache_dtype=torch.float32,
            )
            Dsv4NpuRoPE.apply_rotary_mul_inplace(
                o,
                None,
                cos4,
                sin4,
                qk_nope_dim=o.shape[-1] - rd,
            )
        else:
            apply_rotary_emb(o[..., -rd:], self.freqs_cis[positions], inverse=True)

        o = o.view(
            o.shape[0],
            self.n_local_groups,
            o.shape[1] * o.shape[2] // self.n_local_groups,
        )
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        if self._use_fast_kernel:
            o = _apply_wo_a_bf16_matmul(
                o,
                wo_a,
                is_decode=forward_batch.forward_mode.is_decode(),
                is_target_verify=forward_batch.forward_mode.is_target_verify(),
                fast_path=self.is_dsv41,
            )
        else:
            o = torch.einsum("bgd,grd->bgr", o.float(), wo_a.float()).to(q.dtype)
        value = o.reshape(o.shape[0], o.shape[1] * o.shape[2])
        out, _ = self.wo_b(value, skip_all_reduce=defer_all_reduce)
        return (
            mhc.AttnOutput(out)
            if _is_cuda and defer_all_reduce and self.attn_tp_size > 1
            else out
        )

    @staticmethod
    def accepts_mxfp8_swizzled_input():
        return False

