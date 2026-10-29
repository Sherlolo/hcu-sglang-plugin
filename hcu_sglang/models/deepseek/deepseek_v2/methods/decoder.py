"""DCU DeepSeek V2 fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from hcu_sglang.models.deepseek.deepseek_v2 import constants as C
from hcu_sglang.models.deepseek.deepseek_v2.methods._runtime import *  # noqa: F403

def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        moe_quant_config_override: Optional[QuantizationConfig] = None,
        is_nextn: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        dsa_enable_prefill_cp: bool = False,
        mla_enable_prefill_cp: bool = False,
    ) -> None:
        super(DeepseekV2DecoderLayer, self).__init__()
        self.hidden_size = config.hidden_size
        self.config = config
        if hasattr(config, "rope_parameters"):
            rope_theta = config.rope_parameters["rope_theta"]
            assert rope_theta is not None, f"rope_theta not found in config: {config}"
            rope_type = config.rope_parameters.get("rope_type")
            rope_scaling = config.rope_parameters if rope_type != "default" else None
        else:
            rope_theta = config.rope_theta
            rope_scaling = config.rope_scaling
        max_position_embeddings = config.max_position_embeddings
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            get_spec().speculative_algorithm
        )
        self.dsa_enable_prefill_cp = dsa_enable_prefill_cp
        self.mla_enable_prefill_cp = mla_enable_prefill_cp
        self.layer_id = layer_id
        self.is_nextn = is_nextn
        self.self_attn = DeepseekV2AttentionMLA(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            qk_nope_head_dim=config.qk_nope_head_dim,
            qk_rope_head_dim=config.qk_rope_head_dim,
            v_head_dim=config.v_head_dim,
            q_lora_rank=(
                config.q_lora_rank if hasattr(config, "q_lora_rank") else None
            ),
            kv_lora_rank=config.kv_lora_rank,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            layer_id=layer_id,
            reduce_results=False,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
            is_nextn=is_nextn,
            dsa_enable_prefill_cp=dsa_enable_prefill_cp,
            mla_enable_prefill_cp=mla_enable_prefill_cp,
        )
        if not hasattr(config, "q_lora_rank") and envs.SGLANG_USE_AG_AFTER_QLORA.get():
            raise ValueError(
                "SGLANG_USE_AG_AFTER_QLORA only supports the model with q_lora_rank"
            )

        self.is_layer_sparse = self._is_layer_sparse(layer_id, is_nextn=is_nextn)
        is_previous_layer_sparse = self._is_layer_sparse(layer_id - 1, is_nextn=False)
        is_next_layer_sparse = self._is_layer_sparse(layer_id + 1, is_nextn=False)

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=1 if is_nextn else config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = DeepseekV2MoE(
                config=config,
                quant_config=moe_quant_config_override or quant_config,
                prefix=add_prefix("mlp", prefix),
                layer_id=self.layer_id,
                alt_stream=alt_stream,
                is_nextn=is_nextn,
                dsa_enable_prefill_cp=dsa_enable_prefill_cp,
                mla_enable_prefill_cp=mla_enable_prefill_cp,
            )
        else:
            if enable_moe_dense_fully_dp():
                mlp_tp_rank, mlp_tp_size = 0, 1
            else:
                mlp_tp_rank, mlp_tp_size = None, None
            self.mlp = DeepseekV2MLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
                tp_rank=mlp_tp_rank,
                tp_size=mlp_tp_size,
                swiglu_limit=getattr(config, "swiglu_limit", None),
            )

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self._gfx95_quant_format = self._detect_gfx95_quant_format()

        if self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp:
            # DSACPLayerCommunicator is flavor-agnostic; its internal gates
            # read both dsa_use_prefill_cp and mla_use_prefill_cp. The rename
            # to CPLayerCommunicator is deferred to a cleanup PR.
            self.layer_communicator = DSACPLayerCommunicator(
                layer_scatter_modes=self.layer_scatter_modes,
                input_layernorm=self.input_layernorm,
                post_attention_layernorm=self.post_attention_layernorm,
                allow_reduce_scatter=True,
                is_last_layer=(
                    is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
                ),
                qkv_latent_func=self.self_attn.prepare_qkv_latent,
            )
        else:
            self.layer_communicator = LayerCommunicator(
                layer_scatter_modes=self.layer_scatter_modes,
                input_layernorm=self.input_layernorm,
                post_attention_layernorm=self.post_attention_layernorm,
                allow_reduce_scatter=True,
                is_last_layer=(
                    is_nextn or (self.layer_id == self.config.num_hidden_layers - 1)
                ),
                qkv_latent_func=self.self_attn.prepare_qkv_latent,
            )

def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        zero_allocator: BumpAllocator,
        gemm_output_zero_allocator: BumpAllocator = None,
        llama_4_scaling: Optional[torch.Tensor] = None,
        prev_topk_indices: Optional[torch.Tensor] = None,
        captured_last_layer_outputs: Optional[AuxHiddenStateAccumulator] = None,
        next_full_attention_layer_id: Optional[int] = None,
    ) -> torch.Tensor:
        hidden_states_orig = hidden_states
        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
                quant_format=self._resolve_gfx95_quant_format(),
            )
        )

        with self.self_attn.maybe_use_decode_attn_tp(forward_batch):
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                zero_allocator=zero_allocator,
                llama_4_scaling=llama_4_scaling,
                layer_scatter_modes=self.layer_scatter_modes,
                prev_topk_indices=prev_topk_indices,
            )
        if isinstance(hidden_states, tuple):
            hidden_states, topk_indices = hidden_states
        else:
            topk_indices = None
        get_attn_tp_context().clear_attn_inputs()

        maybe_prefetch_next_full_attention_kv(
            forward_batch, next_full_attention_layer_id
        )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        fuse_mlp_allreduce = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        mlp_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        if isinstance(self.mlp, DeepseekV2MLP):
            gemm_output_zero_allocator = None

        if (
            isinstance(self.mlp, DeepseekV2MoE)
            and not self.mlp.experts.moe_runner_config.inplace
            and not torch.compiler.is_compiling()
        ):
            from sglang.srt.layers.moe.moe_runner.base import moe_output_buffer_ctx

            _mlp_ctx = moe_output_buffer_ctx(hidden_states_orig)
        else:
            _mlp_ctx = nullcontext()

        with get_forward().scoped(
            fuse_mlp_allreduce=fuse_mlp_allreduce,
            mlp_reduce_scatter=mlp_reduce_scatter,
        ):
            with _mlp_ctx:
                hidden_states = self.mlp(
                    hidden_states,
                    forward_batch,
                    gemm_output_zero_allocator,
                )

        if (
            not (self.dsa_enable_prefill_cp or self.mla_enable_prefill_cp)
            and fuse_mlp_allreduce
        ):
            hidden_states._sglang_needs_allreduce_fusion = True

        if not fuse_mlp_allreduce:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual, topk_indices
