"""DCU DeepSeek V2 fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from hcu_sglang.models.deepseek.deepseek_v2 import constants as C
from hcu_sglang.models.deepseek.deepseek_v2.methods._runtime import *  # noqa: F403

def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        layer_id: int = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        skip_rope: bool = False,
        is_nextn: bool = False,
        dsa_enable_prefill_cp: bool = False,
        mla_enable_prefill_cp: bool = False,
    ) -> None:
        super(DeepseekV2AttentionMLA, self).__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.quant_config = quant_config
        self.is_nextn = is_nextn
        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size
        self.use_dsa = is_deepseek_dsa(config)
        self.dsa_enable_prefill_cp = dsa_enable_prefill_cp
        self.mla_enable_prefill_cp = mla_enable_prefill_cp
        if self.dsa_enable_prefill_cp:
            assert self.use_dsa, "CP currently only supports deepseek v3.2 model"
        # cp reuses the attn_tp comm group but needs to duplicate the weights;
        # store cp_size whenever either CP flavor is active so rebuild_cp_kv_cache
        # and the FA3 MLA wrapper can reach it on the dense MLA path too.
        if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp:
            self.cp_size = get_parallel().attn_cp_size
        self.num_heads = num_heads
        assert num_heads % attn_tp_size == 0
        self.num_local_heads = num_heads // attn_tp_size
        self.scaling = self.qk_head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.kv_cache_dtype = get_model().kv_cache_dtype

        # NOTE modification to rope_scaling must be done early enough, b/c e.g. Indexer needs it
        if rope_scaling:
            rope_scaling["rope_type"] = "deepseek_yarn"

        # For tensor parallel attention
        if self.q_lora_rank is not None:
            self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("fused_qkv_a_proj_with_mqa", prefix),
            )
            self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = ColumnParallelLinear(
                q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=self._get_q_b_proj_quant_config(quant_config),
                prefix=add_prefix("q_b_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("q_proj", prefix),
                tp_rank=attn_tp_rank,
                tp_size=attn_tp_size,
            )
            self.kv_a_proj_with_mqa = ReplicatedLinear(
                self.hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("kv_a_proj_with_mqa", prefix),
            )

        self.skip_topk = None
        self.next_skip_topk = None
        if self.use_dsa:
            is_neox_style = not getattr(config, "indexer_rope_interleave", False)
            self.indexer = Indexer(
                hidden_size=hidden_size,
                index_n_heads=get_dsa_index_n_heads(config),
                index_head_dim=get_dsa_index_head_dim(config),
                rope_head_dim=qk_rope_head_dim,
                index_topk=get_dsa_index_topk(config),
                q_lora_rank=q_lora_rank,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                scale_fmt="ue8m0",
                block_size=128,
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
                prefix=add_prefix("indexer", prefix),
                quant_config=quant_config,
                layer_id=layer_id,
                alt_stream=alt_stream,
                config=config,
            )
            # Refer: https://arxiv.org/abs/2603.12201 for more details.
            # skip_topk: when True, this layer will skip computation and reuse previous layer's topk indices.
            # next_skip_topk: when True, the next layer will skip computation and reuse this layer's topk indices.
            if is_nextn:
                self.skip_topk = True
                self.next_skip_topk = True
            else:
                index_cli_factor = getattr(config, "cli_factor", 1)
                if index_cli_factor > 1:
                    self.skip_topk = layer_id % index_cli_factor != 0
                    self.next_skip_topk = (layer_id + 1) % index_cli_factor != 0
                else:
                    self.skip_topk = dsa_layer_skips_topk(config, layer_id)
                    self.next_skip_topk = dsa_layer_skips_topk(config, layer_id + 1)

        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("kv_b_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        # O projection.
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)

        if not skip_rope:
            is_neox_style = not getattr(config, "rope_interleave", True)
            self.rotary_emb = get_rope_wrapper(
                qk_rope_head_dim,
                rotary_dim=qk_rope_head_dim,
                max_position=max_position_embeddings,
                base=rope_theta,
                rope_scaling=rope_scaling,
                is_neox_style=is_neox_style,
                device=get_device().device,
            )

            if rope_scaling and rope_scaling.get("apply_yarn_scaling", True):
                self.scaling = compute_mla_mscale_scaling(rope_scaling, self.scaling)
        else:
            self.rotary_emb = None
        self.use_deepseek_yarn_rope = rope_scaling is not None

        self.attn_mqa = RadixAttention(
            self.num_local_heads,
            self.kv_lora_rank + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=1,
            layer_id=layer_id,
            v_head_dim=self.kv_lora_rank,
            quant_config=quant_config,
            prefix=add_prefix("attn_mqa", prefix),
        )
        # use num_local_heads * dcp_world_size because q_nope, q_rope is all gathered from dcp ranks
        if get_parallel().dcp_enabled:
            self.attn_mqa_for_dcp_decode = RadixAttention(
                self.num_local_heads * get_parallel().attn_dcp_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                self.scaling,
                num_kv_heads=1,
                layer_id=layer_id,
                v_head_dim=self.kv_lora_rank,
                quant_config=quant_config,
                prefix=add_prefix("attn_mqa", prefix),
            )

        self.attn_mha = RadixAttention(
            self.num_local_heads,
            self.qk_nope_head_dim + self.qk_rope_head_dim,
            self.scaling,
            num_kv_heads=self.num_local_heads,
            layer_id=layer_id,
            v_head_dim=self.v_head_dim,
            quant_config=quant_config,
            prefix=add_prefix("attn_mha", prefix),
        )

        self.alt_stream = alt_stream
        self.attn_mha.kv_b_proj = None

        self.w_kc = None
        self.w_vc = None
        self.w_scale = 1.0

        # Full-head Q/absorb weights for --dcp-replicate-q-proj, gathered once
        # pre-CUDA-graph-capture by the model runner; None unless replicate is on.
        self.w_kc_qrep = None
        self.q_b_proj_qrep_weight = None

        self.w_scale_k = None
        self.w_scale_v = None
        self.use_deep_gemm_bmm = False

        self.current_attention_backend = (
            None  # Attention backend used by current forward batch
        )

        self.has_fused_proj = hasattr(self, "fused_qkv_a_proj_with_mqa")
        self.is_packed_weight = (
            self.has_fused_proj
            and hasattr(self.fused_qkv_a_proj_with_mqa.quant_method, "quant_config")
            and self.fused_qkv_a_proj_with_mqa.quant_method.quant_config.get_name()
            in {"awq", "awq_marlin", "moe_wna16"}
        )
        self._use_min_latency_fused_a_gemm: bool | None = None
        self.fused_a_gemm_backend = "auto"

        self.has_q_b_proj = hasattr(self, "q_b_proj")
        q_b_proj_verified_shapes = {(2048, 2048), (4096, 2048)}
        self._q_b_proj_verified_shape = self.has_q_b_proj and (
            tuple(self.q_b_proj.weight.shape) in q_b_proj_verified_shapes
        )
        self._use_min_latency_q_b_gemm: bool | None = None

        self.init_mha_forward()
        self.init_mla_forward()
        self.init_mla_fused_rope_rocm_forward()
        self.init_mla_fused_rope_cpu_forward()

def prepare_qkv_latent(
        self, hidden_states: torch.Tensor, forward_batch: ForwardBatch
    ):
        assert self.q_lora_rank is not None
        if self._use_min_latency_fused_a_gemm is None:
            self._use_min_latency_fused_a_gemm = (
                self.has_fused_proj
                and not get_exec().deterministic.enable_deterministic_inference
                and not self.is_packed_weight
                and fused_a_gemm_weight_eligible(self.fused_qkv_a_proj_with_mqa)
            )
        if self._use_min_latency_fused_a_gemm:
            return linear_with_fused_a_gemm(
                self.fused_qkv_a_proj_with_mqa,
                hidden_states,
                backend=self.fused_a_gemm_backend,
            )
        return self.fused_qkv_a_proj_with_mqa(hidden_states)[0]

def forward_absorb_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.model_executor.runner import get_is_capture_mode

        # Q8KV8 q-prep/indexer overlap handshake (see the fork site below):
        # True between the alt-stream fork and its consumption in the born
        # block; also suppresses the duplicate split/rope on that path.
        self._q8kv8_qprep_overlap_pending = False

        fuse_bmm_attention = (
            self.q_lora_rank is not None
            and self._can_fuse_bmm_into_attention(forward_batch)
        )
        # --dcp-replicate-q-proj: project full-head Q locally from pre-gathered
        # weights and skip the per-layer Q all-gather (bf16 decode absorb only).
        q_replicate_active = (
            get_parallel().dcp_replicate_q_proj
            and is_dcp_mla_decode_phase(forward_batch)
            and not self.use_deep_gemm_bmm
            and self.w_kc_qrep is not None
            and self.q_b_proj_qrep_weight is not None
        )
        if q_replicate_active:
            # force standard absorb so the full-head w_kc bmm runs
            fuse_bmm_attention = False
        q_lora = None
        topk_indices = None
        q_nope = None
        q_pe = None
        k_pe = None
        fusion_plan: Optional[MlaBmmFusionPlan] = None
        if self.q_lora_rank is not None:
            q, latent_cache = (
                get_attn_tp_context()
                .fetch_qkv_latent()
                .split(
                    [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
                    dim=-1,
                )
            )
            k_nope = latent_cache[..., : self.kv_lora_rank]

            # overlap qk norm
            if self.alt_stream is not None and get_is_capture_mode():
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                q = self.q_a_layernorm(q)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = self.kv_a_layernorm(k_nope)
                current_stream.wait_stream(self.alt_stream)
            else:
                q = self.q_a_layernorm(q)
                k_nope = self.kv_a_layernorm(k_nope)

            # q_lora needed by indexer
            if self.use_dsa:
                if q_lora is None:
                    q_lora = q

            # overlap q_b_proj and indexer during decode
            if (
                self.alt_stream is not None
                and get_is_capture_mode()
                and forward_batch.forward_mode.is_decode_or_idle()
                and q_lora is not None
                and not q_replicate_active
            ):
                current_stream = torch.cuda.current_stream()
                self.alt_stream.wait_stream(current_stream)
                with torch.cuda.stream(self.alt_stream):
                    k_nope = k_nope.unsqueeze(1)
                    q = self.q_b_proj_forward(q)
                if self.should_run_indexer(prev_topk_indices):
                    topk_indices = self.indexer(
                        x=hidden_states,
                        q_lora=q_lora,
                        positions=positions,
                        forward_batch=forward_batch,
                        layer_id=self.layer_id,
                    )
                else:
                    # skip_topk reuses prev layer's indices; mirror into this
                    # layer's slot so the captured buffer matches what's used.
                    topk_indices = maybe_capture_indexer_topk(
                        self.layer_id, prev_topk_indices
                    )
                current_stream.wait_stream(self.alt_stream)
            else:
                k_nope = k_nope.unsqueeze(1)
                if q_replicate_active:
                    # full-head Q from the gathered weight (skips Q all-gather)
                    q = torch.nn.functional.linear(q, self.q_b_proj_qrep_weight).view(
                        -1,
                        self.num_local_heads * get_parallel().attn_dcp_size,
                        self.qk_head_dim,
                    )
                else:
                    q = self.q_b_proj_forward(q)

                # Hoist these above the DSA indexer split op so the indexer
                # and the composite bmm+attention split op are adjacent in FX.
                if fuse_bmm_attention:
                    q_nope, q_pe, k_pe = self._split_q_nope_pe(q, latent_cache)
                    fusion_plan = self._make_mla_bmm_fusion_plan(q, q_nope)

                # Q8KV8 q-prep/indexer overlap (opt-in): the born-fp8 q-prep
                # chain (split -> rope -> fused absorbed-bmm+cast, ~173us)
                # and the indexer chain both fork from the q_a_layernorm
                # output and never touch each other's tensors, so the q-prep
                # can run on alt_stream underneath the indexer.  The fork
                # must be enqueued BEFORE the indexer (a later wait_stream
                # would serialize behind it).  The born predicate itself
                # guarantees eager-only and the plain-rope branch (all fused
                # /skip-rope variants make it return None), so applying rope
                # here is exactly what the skipped block below would do.
                if (
                    _ENABLE_DSA_Q8KV8_QPREP_OVERLAP
                    and _ENABLE_DSA_Q8KV8_BORN_FP8_Q
                    and fusion_plan is None
                    and self.alt_stream is not None
                    and q_lora is not None
                    and self.rotary_emb is not None
                ):
                    _born_backend_early = self._q8kv8_born_fp8_q_backend(
                        forward_batch, llama_4_scaling
                    )
                    if _born_backend_early is not None:
                        q_nope, q_pe, k_pe = self._split_q_nope_pe(q, latent_cache)
                        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
                        _q_fp8 = _born_backend_early.q8kv8_acquire_born_q_buffer(
                            q_nope.shape[0],
                            self.num_local_heads,
                            self.kv_lora_rank + self.qk_rope_head_dim,
                            q_nope.device,
                        )
                        self.alt_stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(self.alt_stream):
                            absorbed_bmm_concat_cast_q_fp8(
                                _q_fp8,
                                q_nope,
                                self.w_kc,
                                q_pe,
                                self.num_local_heads,
                            )
                        self._q8kv8_qprep_overlap_pending = True

                if q_lora is not None:
                    if self.should_run_indexer(prev_topk_indices):
                        topk_indices = self.indexer(
                            x=hidden_states,
                            q_lora=q_lora,
                            positions=positions,
                            forward_batch=forward_batch,
                            layer_id=self.layer_id,
                        )
                    else:
                        topk_indices = maybe_capture_indexer_topk(
                            self.layer_id, prev_topk_indices
                        )
        else:
            if q_replicate_active:
                q = torch.nn.functional.linear(
                    hidden_states, self.q_b_proj_qrep_weight
                ).view(
                    -1,
                    self.num_local_heads * get_parallel().attn_dcp_size,
                    self.qk_head_dim,
                )
            else:
                q = self.q_proj(hidden_states)[0].view(
                    -1, self.num_local_heads, self.qk_head_dim
                )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
            k_nope = latent_cache[..., : self.kv_lora_rank]
            k_nope = self.kv_a_layernorm(k_nope).unsqueeze(1)

        if q_nope is None:
            q_nope, q_pe, k_pe = self._split_q_nope_pe(q, latent_cache)

        _kvb_q = None
        born_q_backend = None
        if (
            _ENABLE_DSA_Q8KV8_BORN_FP8_Q
            and fusion_plan is None
            and q_nope.dtype == torch.bfloat16
        ):
            born_q_backend = self._q8kv8_born_fp8_q_backend(
                forward_batch, llama_4_scaling
            )
        if q_replicate_active:
            # full-head absorb with the pre-gathered w_kc (q_nope already full-head)
            q_nope_out = (
                torch.bmm(q_nope.transpose(0, 1), self.w_kc_qrep)
                .transpose(0, 1)
                .contiguous()
            )
        elif fusion_plan is not None:
            # The composite split op fills q_nope_out_buf and attention reads
            # this transposed alias directly.
            q_nope_out = fusion_plan.q_nope_out_view
        elif born_q_backend is not None:
            # Born-fp8 q: skip the bf16 absorbed bmm entirely; the fused
            # bmm+concat+cast kernel (launched after rope below) writes the
            # fp8 q directly into the q8kv8 backend buffer.
            q_nope_out = None
        else:
            if _SGLANG_EXPERIMENTAL_LORA_OPTI:
                # Fork the kv_b q-correction A-step onto the LoRA side stream to overlap the bmm.
                from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                    kv_b_lora_q_prepare,
                )

                _kvb_q = kv_b_lora_q_prepare(self, q_nope)

            if self.use_deep_gemm_bmm:
                (
                    q_nope_val,
                    q_nope_scale,
                    masked_m,
                    expected_m,
                    aligned_m,
                ) = per_token_group_quant_mla_deep_gemm_masked_fp8(
                    q_nope.transpose(0, 1)
                )
                q_nope_out = q_nope.new_empty(
                    (self.num_local_heads, aligned_m, self.kv_lora_rank)
                )
                deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                    (q_nope_val, q_nope_scale),
                    (self.w_kc, self.w_scale_k),
                    q_nope_out,
                    masked_m,
                    expected_m,
                )
                q_nope_out = q_nope_out[:, :expected_m, :]
            elif self.w_kc.dtype == torch.float8_e4m3fn:
                if C._is_cpu:
                    q_nope_out = torch.bmm(
                        q_nope.to(torch.bfloat16).transpose(0, 1),
                        self.w_kc.to(torch.bfloat16) * self.w_scale,
                    )
                else:
                    # fix bmm_fp8 error under cublas12.9 caused by bumpallocator, detail in pr#11612
                    q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                        q_nope.transpose(0, 1),
                        (
                            torch.zeros((1,), dtype=torch.float32, device=q_nope.device)
                            if _is_cublas_ge_129
                            else zero_allocator.allocate(1)
                        ),
                    )
                    q_nope_out = bmm_fp8(
                        q_nope_val,
                        self.w_kc,
                        q_nope_scale,
                        self.w_scale,
                        torch.bfloat16,
                    )
            else:
                q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)

            q_nope_out = q_nope_out.transpose(0, 1)
            if _SGLANG_EXPERIMENTAL_LORA_OPTI:
                from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                    kv_b_lora_q_apply,
                )

                q_nope_out = kv_b_lora_q_apply(self, q_nope, q_nope_out, _kvb_q)
            elif is_kv_b_lora_active(self):
                q_nope_out = apply_kv_b_lora_q_correction(self, q_nope, q_nope_out)

        fuse_rope_for_trtllm_mla = self._fuse_rope_for_trtllm_mla(forward_batch)
        if (
            self.rotary_emb is not None
            and not fuse_rope_for_trtllm_mla
            # Already applied at the q-prep/indexer overlap fork.
            and not self._q8kv8_qprep_overlap_pending
        ):
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

        if born_q_backend is not None:
            # Born-fp8 q (SGLANG_ENABLE_DSA_Q8KV8_BORN_FP8_Q): one fused
            # kernel replaces bmm -> bf16 q_nope_out ->
            # concat_and_cast_q_fp8_pad.  q_nope is the pre-absorb bf16 view
            # (rope only touched the disjoint q_pe columns) and q_pe carries
            # the post-rope values.  The stash is consumed by
            # _forward_flashmla_sparse_q8kv8; q_nope_out becomes a
            # NaN-poisoned shape-only sentinel.
            num_tokens = q_nope.shape[0]
            if self._q8kv8_qprep_overlap_pending:
                # q_fp8 was produced on alt_stream at the fork above; join so
                # everything downstream (incl. the next layer's fork, which
                # reuses the single born-q slot) orders after it.
                torch.cuda.current_stream().wait_stream(self.alt_stream)
                self._q8kv8_qprep_overlap_pending = False
            else:
                q_fp8 = born_q_backend.q8kv8_acquire_born_q_buffer(
                    num_tokens,
                    self.num_local_heads,
                    self.kv_lora_rank + self.qk_rope_head_dim,
                    q_nope.device,
                )
                absorbed_bmm_concat_cast_q_fp8(
                    q_fp8, q_nope, self.w_kc, q_pe, self.num_local_heads
                )
            born_q_backend.q8kv8_stash_born_q(num_tokens, self.attn_mqa.layer_id)
            q_nope_out = born_q_backend.q8kv8_born_q_sentinel(
                num_tokens, self.num_local_heads, self.kv_lora_rank, q_nope.device
            )

        dsa_prefill_cp = dsa_use_prefill_cp(forward_batch)
        mla_prefill_cp = mla_use_prefill_cp(forward_batch)
        defer_kv_gather_until_after_rope = should_defer_dsa_cp_kv_gather(
            dsa_prefill_cp=dsa_prefill_cp,
            fuse_rope_for_trtllm_mla=fuse_rope_for_trtllm_mla,
        )
        if dsa_prefill_cp and not defer_kv_gather_until_after_rope:
            from sglang.srt.layers.attention.dsa_backend import materialize_full_kv_cp

            k_nope, k_pe = materialize_full_kv_cp(
                self,
                forward_batch,
                latent_cache,
                k_nope,
                k_pe,
            )
        elif mla_prefill_cp and not is_cp_v2_active(forward_batch):
            # CP-v1 gathers the latent here; CP-v2 gathers it in the attention
            # backend via the strategy (materialize_full_mla_kv).
            k_nope, k_pe = self.rebuild_cp_kv_cache(
                latent_cache,
                forward_batch,
                k_nope,
                k_pe,
            )

        # all_gather q_pe, q_nope_out,take tp8 as an example， q_pe [B, H, ROPE_DIM], q_nope_out [B, H, NOPE_DIM] gathered to [B, H * dcp_world_size, ROPE_DIM] [B, H * dcp_world_size, NOPE_DIM] for decode batch, and all gather k_pe, k_nope for extend batch.
        if get_parallel().dcp_enabled:
            if is_dcp_mla_decode_phase(forward_batch):
                if not q_replicate_active:
                    q_nope_out, q_pe = all_gather_q_for_mla_decode(
                        q_nope_out=q_nope_out,
                        q_pe=q_pe,
                    )
            elif forward_batch.forward_mode.is_extend():
                # for extend, gather kv
                all_gather_kv_cache_for_mla_extend(
                    get_token_to_kv_pool(),
                    self.attn_mqa,
                    forward_batch.extend_prefix_lens_cpu,
                    forward_batch.attn_dcp_metadata.dcp_local_prefix_kv_indices,
                    forward_batch.attn_dcp_metadata.dcp_extend_prefix_lens_sum,
                    forward_batch.attn_dcp_metadata.dcp_kv_buffer,
                    self.kv_lora_rank,
                    k_nope,
                    k_pe,
                )
            else:
                logger.warning(
                    f"not supported forward_mode {forward_batch.forward_mode}"
                )

        return (
            q_pe,
            k_pe,
            q_nope_out,
            k_nope,
            forward_batch,
            zero_allocator,
            positions,
            topk_indices,
            llama_4_scaling,
            fusion_plan,
        )

def forward_absorb_core(
        self: DeepseekV2AttentionMLA,
        q_pe,
        k_pe,
        q_nope_out,
        k_nope,
        forward_batch,
        zero_allocator,
        positions,
        topk_indices,
        llama_4_scaling,
        fusion_plan: Optional[MlaBmmFusionPlan] = None,
    ):
        save_kv_cache = True

        if self.current_attention_backend in C.FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
            extra_args = {}
            if self._fuse_rope_for_trtllm_mla(forward_batch):
                extra_args = {
                    "cos_sin_cache": self.rotary_emb.cos_sin_cache,
                    "is_neox": self.rotary_emb.is_neox_style,
                    "llama_4_scaling": llama_4_scaling,
                }
            if fusion_plan is not None:
                bmm_attention_fn = (
                    bcg_mla_bmm_then_unified_attention
                    if is_in_breakable_cuda_graph()
                    else mla_bmm_then_unified_attention
                )
                bmm_attention_fn(
                    fusion_plan.q_nope_t,
                    self.w_kc,
                    fusion_plan.q_nope_out_buf,
                    q_nope_out,
                    k_nope,
                    fusion_plan.attn_output_buf,
                    save_kv_cache,
                    self.layer_id,
                    q_pe,
                    k_pe,
                    cos_sin_cache=extra_args.get("cos_sin_cache"),
                    is_neox=extra_args.get("is_neox"),
                    llama_4_scaling=extra_args.get("llama_4_scaling"),
                    topk_indices=topk_indices,
                )
                attn_output = fusion_plan.attn_output_buf
            elif is_dcp_mla_decode_phase(forward_batch):
                # set return_lse=True to correct attn_output
                attn_output, lse = self.attn_mqa_for_dcp_decode(
                    q_nope_out,
                    k_nope,
                    k_nope,
                    forward_batch,
                    q_rope=q_pe,
                    k_rope=k_pe,
                    **extra_args,
                    **(
                        dict(topk_indices=topk_indices)
                        if topk_indices is not None
                        else {}
                    ),
                )
            else:
                attn_output = self.attn_mqa(
                    q_nope_out,
                    k_nope,
                    k_nope,
                    forward_batch,
                    q_rope=q_pe,
                    k_rope=k_pe,
                    **extra_args,
                    **(
                        dict(topk_indices=topk_indices)
                        if topk_indices is not None
                        else {}
                    ),
                )
        else:
            q = torch.cat([q_nope_out, q_pe], dim=-1)
            k = torch.cat([k_nope, k_pe], dim=-1)

            # Apply llama 4 scaling if provided
            if llama_4_scaling is not None:
                q *= llama_4_scaling

            attn_output = self.attn_mqa(
                q,
                k,
                k_nope,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **(dict(topk_indices=topk_indices) if topk_indices is not None else {}),
            )

        # correct attn_output with respect to lse from other ranks
        if is_dcp_mla_decode_phase(forward_batch):
            attn_output = attn_output.view(
                -1,
                self.num_local_heads * get_parallel().attn_dcp_size,
                self.kv_lora_rank,
            )
            if get_in_autotune_dummy_run():
                # The synthetic FlashInfer MoE autotune pass discards model
                # outputs. Avoid an unnecessary cross-node MNNVL exchange of
                # zero attention partials.
                attn_output = _select_local_dcp_heads_for_autotune(
                    attn_output, self.num_local_heads
                )
            else:
                dcp_comm_backend = get_parallel().dcp_comm_backend
                is_lse_base_on_e = is_mla_dcp_lse_base_on_e(
                    self.current_attention_backend
                )
                if dcp_comm_backend in ("a2a", "fi_a2a"):
                    # A2A exchange of head partials + LSE, then local Triton combine.
                    attn_output = dcp_a2a_lse_reduce(
                        attn_output.contiguous(),
                        lse.contiguous(),
                        get_parallel().dcp_group,
                        is_lse_base_on_e=is_lse_base_on_e,
                        comm_backend=dcp_comm_backend,
                    )
                else:
                    attn_output = cp_lse_ag_out_rs_mla(
                        attn_output,
                        lse,
                        get_parallel().dcp_group,
                        is_lse_base_on_e=is_lse_base_on_e,
                    )
                    attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        _kvb_v = None
        if _SGLANG_EXPERIMENTAL_LORA_OPTI:
            # Fork the kv_b v-correction A-step onto the LoRA side stream to overlap the bmm.
            from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                kv_b_lora_v_prepare,
            )

            _kvb_v = kv_b_lora_v_prepare(self, attn_output)

        if self.use_deep_gemm_bmm:
            (
                attn_output_val,
                attn_output_scale,
                masked_m,
                expected_m,
                aligned_m,
            ) = per_token_group_quant_mla_deep_gemm_masked_fp8(
                attn_output.transpose(0, 1)
            )
            attn_bmm_output = attn_output.new_empty(
                (self.num_local_heads, aligned_m, self.v_head_dim)
            )
            deep_gemm_wrapper.grouped_gemm_nt_f8f8bf16_masked(
                (attn_output_val, attn_output_scale),
                (self.w_vc, self.w_scale_v),
                attn_bmm_output,
                masked_m,
                expected_m,
            )
            attn_bmm_output = (
                attn_bmm_output[:, :expected_m, :].transpose(0, 1).flatten(1, 2)
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            if C._is_cpu:
                attn_bmm_output = torch.bmm(
                    attn_output.to(torch.bfloat16).transpose(0, 1),
                    self.w_vc.to(torch.bfloat16) * self.w_scale,
                )
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
            else:
                attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                    attn_output.transpose(0, 1),
                    (
                        torch.zeros(
                            (1,), dtype=torch.float32, device=attn_output.device
                        )
                        if _is_cublas_ge_129
                        else zero_allocator.allocate(1)
                    ),
                )
                attn_bmm_output = bmm_fp8(
                    attn_output_val,
                    self.w_vc,
                    attn_output_scale,
                    self.w_scale,
                    torch.bfloat16,
                )
                attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        elif _is_musa:
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1), self.w_vc
            )
            attn_bmm_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        else:
            if is_in_tc_piecewise_cuda_graph():
                # torch dynamo requires out= op was called where output tensor was non-contiguous
                attn_bmm_output = (
                    torch.bmm(attn_output.transpose(0, 1), self.w_vc)
                    .transpose(0, 1)
                    .flatten(1, 2)
                )
            else:
                attn_bmm_output = torch.empty(
                    (attn_output.shape[0], self.num_local_heads * self.v_head_dim),
                    dtype=attn_output.dtype,
                    device=attn_output.device,
                )
                torch.bmm(
                    attn_output.transpose(0, 1),
                    self.w_vc,
                    out=attn_bmm_output.view(
                        -1, self.num_local_heads, self.v_head_dim
                    ).transpose(0, 1),
                )
        if _SGLANG_EXPERIMENTAL_LORA_OPTI:
            from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import (
                kv_b_lora_v_apply,
            )

            attn_bmm_output = kv_b_lora_v_apply(
                self, attn_output, attn_bmm_output, _kvb_v
            )
        elif is_kv_b_lora_active(self):
            attn_bmm_output = apply_kv_b_lora_v_correction(
                self, attn_output, attn_bmm_output
            )
        output, _ = self.o_proj(attn_bmm_output)

        if self.next_skip_topk is None:
            return output

        # Return topk_indices for the next layer when enabling index cache
        if not self.next_skip_topk:
            return output, None
        else:
            return output, topk_indices

def _set_mla_kv_buffer(
        self: DeepseekV2AttentionMLA,
        latent_cache: torch.Tensor,
        kv_a: torch.Tensor,
        k_pe: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        if C._is_cuda:
            # Save latent cache
            get_token_to_kv_pool().set_mla_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
            )
        elif C._is_npu:
            # To reduce a time-costing split operation
            get_token_to_kv_pool().set_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, kv_a.unsqueeze(1), k_pe
            )
        else:
            latent_cache[:, :, : self.kv_lora_rank] = kv_a.unsqueeze(1)
            latent_cache[:, :, self.kv_lora_rank :] = k_pe.clone()

            # Save latent cache
            get_token_to_kv_pool().set_kv_buffer(
                self.attn_mha, forward_batch.out_cache_loc, latent_cache, None
            )

def _get_mla_kv_buffer(
        self: DeepseekV2AttentionMLA,
        kv_indices: torch.Tensor,
        dst_dtype: torch.dtype,
        forward_batch: ForwardBatch,
    ):
        if C._is_cuda:
            kv_indices = filter_dcp_local_kv_indices(kv_indices=kv_indices)
            kv_a, k_pe = get_token_to_kv_pool().get_mla_kv_buffer(
                self.attn_mha, kv_indices, dst_dtype
            )
            kv_a = kv_a.squeeze(1)
        else:
            latent_cache_buf = get_token_to_kv_pool().get_key_buffer(
                self.attn_mha.layer_id
            )
            latent_cache = latent_cache_buf[kv_indices].contiguous().to(dst_dtype)

            kv_a, k_pe = latent_cache.split(
                [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )
            kv_a = kv_a.squeeze(1).contiguous()
        return kv_a, k_pe

def _get_mla_kv_buffer_from_fp8_for_dsa(
        self: DeepseekV2AttentionMLA,
        forward_batch: ForwardBatch,
    ):
        """
        Dequantize FP8 KV cache to BF16 for MLA attention (DSA-specific format).

        Returns: (kv_a, k_pe) both in BF16
        """
        backend = get_attn_backend()
        if isinstance(backend, TboAttnBackend):  # if enable tbo, get primary backend
            backend = backend.primary
        kv_indices = backend.forward_metadata.page_table_1_flattened
        assert (
            kv_indices is not None
        ), "page_table_1_flattened should have been generated for FP8 MHA path"

        if C._use_aiter_gfx95:
            # ROCm (gfx950) stores the FP8 MLA KV in the raw
            # (kv_lora_rank + qk_rope_head_dim) layout, not the scaled 656-byte
            # layout that dequantize_k_cache_paged expects (it asserts dim==656).
            # Dequantize the raw layout via the pool's HIP-aware path instead —
            # the same routine _get_mla_kv_buffer uses for the BF16 MHA path.
            # Without this, a chunked-prefill split (extend_prefix_lens != 0) that
            # reads cached prefix KV crashes with "576 != 656".
            kv_indices = filter_dcp_local_kv_indices(kv_indices=kv_indices)
            kv_a, k_pe = get_token_to_kv_pool().get_mla_kv_buffer(
                self.attn_mha, kv_indices, torch.bfloat16
            )
            kv_a = kv_a.squeeze(1).contiguous()
            return kv_a, k_pe

        kv_cache_fp8 = get_token_to_kv_pool().get_key_buffer(self.attn_mha.layer_id)

        kv_latent_bf16 = dequantize_k_cache_paged(kv_cache_fp8, kv_indices)

        kv_a = kv_latent_bf16[:, :, : self.kv_lora_rank].squeeze(1).contiguous()
        k_pe = kv_latent_bf16[:, :, self.kv_lora_rank :]

        return kv_a, k_pe

def _concat_and_cast_mha_k(
        self: DeepseekV2AttentionMLA,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        # Temporary for DeepSeek V3/R1 only, but can generalize if needed
        k_shape = (k_nope.shape[0], self.num_local_heads, self.qk_head_dim)
        if (
            (C._is_cuda or _is_musa)
            and (self.num_local_heads == 128)
            and (self.qk_nope_head_dim == 128)
            and (self.qk_rope_head_dim == 64)
        ):
            k = k_nope.new_empty(*k_shape)
            concat_mla_k(k=k, k_nope=k_nope, k_rope=k_pe)
        elif (
            C._is_cuda
            and next_power_of_2(self.num_local_heads) == self.num_local_heads
            and next_power_of_2(self.qk_nope_head_dim) == self.qk_nope_head_dim
            and next_power_of_2(self.qk_rope_head_dim) == self.qk_rope_head_dim
        ):
            # fa3 mha support fp8 inputs
            if (
                self.current_attention_backend == "fa3"
                and self.kv_cache_dtype != "auto"
            ):
                attn_dtype = get_token_to_kv_pool().dtype
            else:
                attn_dtype = k_nope.dtype
            k = k_nope.new_empty(*k_shape, dtype=attn_dtype)
            concat_and_cast_mha_k_triton(k, k_nope, k_pe)
        else:
            k = k_nope.new_empty(*k_shape)
            k[..., : self.qk_nope_head_dim] = k_nope
            k[..., self.qk_nope_head_dim :] = k_pe
        return k

def _chunked_prefix_attn_mha(
        self: DeepseekV2AttentionMLA,
        q: torch.Tensor,
        accum_output: torch.Tensor,
        accum_lse: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # kv_b_proj needs BF16 input, but legacy q.dtype was BF16 by accident.
        backend = resolve_attn_backend(forward_batch)
        pack_fn = getattr(backend, "pack_prefix_chunk_kv", None)
        kv_a_dtype = torch.bfloat16 if pack_fn is not None else q.dtype

        assert forward_batch.num_prefix_chunks is not None
        for i in range(forward_batch.num_prefix_chunks):
            forward_batch.set_prefix_chunk_idx(i)

            kv_indices = forward_batch.prefix_chunk_kv_indices[i]
            # Fetch latent cache from memory pool with precomputed chunked kv indices
            kv_a_normed, k_pe = self._get_mla_kv_buffer(
                kv_indices, kv_a_dtype, forward_batch
            )
            kv_a_normed, k_pe = all_gather_kv_cache_for_mha_chunk_extend(
                kv_a_normed,
                k_pe,
                forward_batch.prefix_chunk_seq_lens_cpu[i],
                forward_batch.prefix_chunk_starts_cpu[i],
            )
            kv = self.kv_b_proj(kv_a_normed)[0]
            kv = kv.view(
                -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
            )
            v = kv[..., self.qk_nope_head_dim :]
            k_nope = kv[..., : self.qk_nope_head_dim]

            if pack_fn is not None:
                k, v = pack_fn(k_nope, k_pe, v)
            else:
                k = torch.empty(
                    (
                        k_nope.shape[0],
                        self.num_local_heads,
                        self.qk_nope_head_dim + self.qk_rope_head_dim,
                    ),
                    dtype=v.dtype,
                    device=v.device,
                )
                k[..., : self.qk_nope_head_dim] = k_nope
                k[..., self.qk_nope_head_dim :] = k_pe

            output, lse = self.attn_mha(
                q,
                k,
                v,
                forward_batch,
                save_kv_cache=False,
                # Prefix K/V is independent of the suffix query length. Under
                # FullCG this is the fixed captured chunk extent; per-request
                # active lengths remain encoded in the backend metadata.
                key_value_num_tokens=k.shape[0],
            )
            tmp_output = torch.empty_like(accum_output)
            tmp_lse = torch.empty_like(accum_lse)
            merge_state_v2(output, lse, accum_output, accum_lse, tmp_output, tmp_lse)
            accum_output, accum_lse = tmp_output, tmp_lse
            del kv, k, v, output, lse, tmp_output, tmp_lse

        return accum_output

def forward_normal_chunked_kv_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        # In normal mha, the k and v tensors will become overly large when the prefix length is long.
        # To avoid this, we split the kv cache into chunks and process them one after another.
        # Since mha is compute friendly, the for loop induced here will not introduce significant overhead.
        # The top comments in https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/mla/common.py
        # will be helpful for understanding the purpose of this function.

        # First do normal mha forward to get output for extended part
        return self.forward_normal_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

def forward_normal_chunked_kv_core(
        self: DeepseekV2AttentionMLA,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        has_extend_prefix = forward_batch.extend_prefix_lens_cpu is not None and any(
            forward_batch.extend_prefix_lens_cpu
        )
        # Only initialize the info once
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.prepare_chunked_prefix_cache_info(q.device)
            if hasattr(get_attn_backend(), "init_mha_chunk_metadata"):
                get_attn_backend().init_mha_chunk_metadata(forward_batch)

        forward_batch.mha_return_lse = has_extend_prefix
        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        attn_output = self.attn_mha(q, k, v, forward_batch, save_kv_cache=False)

        # Do mha attention with chunked prefix cache if there are any sequence with prefix
        if has_extend_prefix:
            attn_output, lse = attn_output
            forward_batch.set_attn_attend_prefix_cache(True)
            attn_output = self._chunked_prefix_attn_mha(
                q=q,
                accum_output=attn_output,
                accum_lse=lse,
                forward_batch=forward_batch,
            )

        attn_output = attn_output.reshape(-1, self.num_local_heads * self.v_head_dim)
        output, _ = self.o_proj(attn_output)
        return output

def forward_normal_one_shot_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        forward_batch.mha_one_shot = True
        return self.forward_normal_prepare(
            positions, hidden_states, forward_batch, zero_allocator
        )

def forward_normal_one_shot_core(
        self: DeepseekV2AttentionMLA,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        has_extend_prefix = any(forward_batch.extend_prefix_lens_cpu)
        # Only initialize the info once
        if has_extend_prefix and forward_batch.num_prefix_chunks is None:
            forward_batch.num_prefix_chunks = 0
            if hasattr(get_attn_backend(), "init_mha_chunk_metadata"):
                get_attn_backend().init_mha_chunk_metadata(forward_batch)
        forward_batch.mha_return_lse = False
        # Do mha for extended part without prefix
        forward_batch.set_attn_attend_prefix_cache(False)
        return self.forward_normal_core(q, k, v, forward_batch)

def forward_absorb_fused_mla_rope_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        enable_rope_fusion = (
            os.getenv("SGLANG_FUSED_MLA_ENABLE_ROPE_FUSION", "1") == "1"
        )
        # NOTE: hidden_states can be a tuple for some quantization paths.
        # For shape/device/dtype, use the first tensor; still pass the original
        # hidden_states through linear ops which may accept tuple inputs.
        hidden_states_tensor = (
            hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
        )

        q_len = hidden_states_tensor.shape[0]
        q_input = hidden_states_tensor.new_empty(
            q_len, self.num_local_heads, self.kv_lora_rank + self.qk_rope_head_dim
        )
        if self.q_lora_rank is not None:
            q, latent_cache = self.fused_qkv_a_proj_with_mqa(hidden_states)[0].split(
                [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
            )
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
            latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if C._is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            q_nope_out = torch.bmm(
                q_nope.to(torch.bfloat16).transpose(0, 1),
                self.w_kc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_kc.dtype == torch.float8_e4m3fn:
            q_nope_val, q_nope_scale = per_tensor_quant_mla_fp8(
                q_nope.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
            )
            q_nope_out = bmm_fp8(
                q_nope_val, self.w_kc, q_nope_scale, self.w_scale, torch.bfloat16
            )
        else:
            q_nope_out = torch.bmm(q_nope.transpose(0, 1), self.w_kc)
        q_input[..., : self.kv_lora_rank] = q_nope_out.transpose(0, 1)
        v_input = latent_cache[..., : self.kv_lora_rank]
        v_input = self.kv_a_layernorm(v_input.contiguous()).unsqueeze(1)
        k_input = latent_cache.unsqueeze(1)
        k_input[..., : self.kv_lora_rank] = v_input

        if not enable_rope_fusion:
            k_pe = k_input[..., self.kv_lora_rank :]
            q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
            q_input[..., self.kv_lora_rank :] = q_pe
            k_input[..., self.kv_lora_rank :] = k_pe
            k_pe_output = None
        else:
            k_pe_output = torch.empty_like(k_input[..., self.kv_lora_rank :])

        q_input[..., self.kv_lora_rank :] = q_pe

        # attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        # Use Fused ROPE with use_rope=OFF.
        attn_output = torch.empty(
            (q_len, self.num_local_heads, self.kv_lora_rank),
            dtype=q.dtype,
            device=q.device,
        )
        attn_logits, _, kv_indptr, kv_indices, _, _, _ = (
            get_attn_backend().forward_metadata
        )
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        num_kv_split = get_attn_backend().num_kv_splits
        sm_scale = self.attn_mqa.scaling
        if attn_logits is None:
            attn_logits = torch.empty(
                (
                    forward_batch.batch_size,
                    self.num_local_heads,
                    num_kv_split,
                    self.kv_lora_rank + 1,
                ),
                dtype=torch.float32,
                device=q.device,
            )

        # save current latent cache.
        get_token_to_kv_pool().set_kv_buffer(
            self.attn_mqa, forward_batch.out_cache_loc, k_input, None
        )
        key_cache_buf = get_token_to_kv_pool().get_key_buffer(self.attn_mqa.layer_id)
        val_cache_buf = key_cache_buf[..., : self.kv_lora_rank]

        return (
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            enable_rope_fusion,
            k_input,
            forward_batch,
            zero_allocator,
        )

def forward_absorb_fused_mla_rope_core(
        self: DeepseekV2AttentionMLA,
        q_input,
        key_cache_buf,
        val_cache_buf,
        attn_output,
        kv_indptr,
        kv_indices,
        k_pe_output,
        cos_sin_cache,
        positions,
        attn_logits,
        num_kv_split,
        sm_scale,
        enable_rope_fusion,
        k_input,
        forward_batch,
        zero_allocator,
    ):
        decode_attention_fwd_grouped_rope(
            q_input,
            key_cache_buf,
            val_cache_buf,
            attn_output,
            kv_indptr,
            kv_indices,
            k_pe_output,
            self.kv_lora_rank,
            self.rotary_emb.rotary_dim,
            cos_sin_cache,
            positions,
            attn_logits,
            num_kv_split,
            sm_scale,
            logit_cap=self.attn_mqa.logit_cap,
            use_rope=enable_rope_fusion,
            is_neox_style=self.rotary_emb.is_neox_style,
        )

        if enable_rope_fusion:
            k_input[..., self.kv_lora_rank :] = k_pe_output
            get_token_to_kv_pool().set_kv_buffer(
                self.attn_mqa, forward_batch.out_cache_loc, k_input, None
            )

        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        if C._is_hip:
            # TODO(haishaw): add bmm_fp8 to ROCm
            attn_bmm_output = torch.bmm(
                attn_output.to(torch.bfloat16).transpose(0, 1),
                self.w_vc.to(torch.bfloat16) * self.w_scale,
            )
        elif self.w_vc.dtype == torch.float8_e4m3fn:
            attn_output_val, attn_output_scale = per_tensor_quant_mla_fp8(
                attn_output.transpose(0, 1),
                zero_allocator.allocate(1),
                dtype=torch.float8_e4m3fn,
            )
            attn_bmm_output = bmm_fp8(
                attn_output_val,
                self.w_vc,
                attn_output_scale,
                self.w_scale,
                torch.bfloat16,
            )
        else:
            attn_bmm_output = torch.bmm(attn_output.transpose(0, 1), self.w_vc)
        attn_output = attn_bmm_output.transpose(0, 1).flatten(1, 2)
        output, _ = self.o_proj(attn_output)

        return output

def forward_absorb_fused_mla_rope_cpu_prepare(
        self: DeepseekV2AttentionMLA,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        zero_allocator: BumpAllocator,
    ):
        assert self.q_lora_rank is not None and use_intel_amx_backend(
            self
        ), "forward_absorb_fused_mla_rope_cpu_prepare requires q_lora_rank is not None and use_intel_amx_backend"

        q_input, k_input, v_input = (
            torch.ops.sgl_kernel.qkv_proj_with_rope_fused_weight(
                hidden_states,
                self.fused_qkv_a_proj_with_mqa.weight,
                self.q_b_proj.weight,
                self.w_kc,
                self.q_a_layernorm.weight,
                self.kv_a_layernorm.weight,
                positions,
                self.rotary_emb.cos_sin_cache,
                self.kv_a_layernorm.variance_epsilon,
                self.qkv_proj_with_rope_is_int8,
                self.qkv_proj_with_rope_is_fp8,
                (
                    self.fused_qkv_a_proj_with_mqa.weight_scale
                    if self.qkv_proj_with_rope_is_int8
                    else (
                        self.fused_qkv_a_proj_with_mqa.weight_scale_inv
                        if self.qkv_proj_with_rope_is_fp8
                        else None
                    )
                ),
                (
                    self.q_b_proj.weight_scale
                    if self.qkv_proj_with_rope_is_int8
                    else (
                        self.q_b_proj.weight_scale_inv
                        if self.qkv_proj_with_rope_is_fp8
                        else None
                    )
                ),
                self.w_scale if self.qkv_proj_with_rope_is_fp8 else None,
                True,  # is_vnni
                self.weight_block_size,
                self.q_lora_rank,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            )
        )
        return (q_input, k_input, v_input, forward_batch, zero_allocator)

def forward_absorb_fused_mla_rope_cpu_core(
        self: DeepseekV2AttentionMLA,
        q_input,
        k_input,
        v_input,
        forward_batch,
        zero_allocator,
    ):
        assert self.q_lora_rank is not None and use_intel_amx_backend(
            self
        ), "forward_absorb_fused_mla_rope_cpu_core requires q_lora_rank is not None and use_intel_amx_backend"

        attn_output = self.attn_mqa(q_input, k_input, v_input, forward_batch)
        attn_output = attn_output.view(-1, self.num_local_heads, self.kv_lora_rank)

        # [Note] Align shapes of bmm inputs.
        # Shapes of inputs:
        #   q_nope: [M, B, K]
        #   original self.w_kc: [B, K, N]
        #   current self.w_kc (which has been converted in PackWeightMethod): [B, N, K]

        # Shapes of inputs to sgl_kernel.cpu.bmm:
        #   out: [B, M, N]
        #   mat1: [B, M, K]
        #   mat2: [B, N, K]
        B = self.w_vc.size(0)
        N = self.w_vc.size(1)
        M = attn_output.size(0)
        output = torch.empty([M, int(B * N)], dtype=attn_output.dtype)
        attn_bmm_output = output.view([M, B, N]).transpose_(0, 1)
        torch.ops.sgl_kernel.bmm_cpu(
            attn_bmm_output,
            attn_output.transpose(0, 1),
            self.w_vc,
            True,  # is_vnni
            self.w_scale if self.qkv_proj_with_rope_is_fp8 else None,  # scale
        )
        attn_output = output
        output, _ = self.o_proj(attn_output)

        return output
