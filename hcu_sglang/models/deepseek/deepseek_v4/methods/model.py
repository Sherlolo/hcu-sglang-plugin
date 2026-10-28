"""DCU DeepSeek V4 fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from hcu_sglang.models.deepseek.deepseek_v4 import constants as C
from hcu_sglang.models.deepseek.deepseek_v4.methods._runtime import *  # noqa: F403

def __init__(
        self,
        config: DeepSeekV4Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super(DeepseekV4Model, self).__init__()
        self.pp_group = get_pp_group()
        self.hidden_size = config.hidden_size
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                enable_tp=not is_dp_attention_enabled(),
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.rms_norm_eps = config.rms_norm_eps
        # HCU retains the five-stream topology required by its dedicated DSV4
        # CP/MegaMoE path; generic HIP keeps the official opt-in behavior.
        use_stream_pool = (
            C._is_dcu
            or _is_cuda
            or (
                _is_hip
                and not C._is_dcu
                and (
                    envs.SGLANG_ROCM_USE_MULTI_STREAM.get()
                    or envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get()
                )
            )
            or (_is_npu and envs.SGLANG_NPU_USE_MULTI_STREAM.get())
        )
        device_module = torch.get_device_module()
        num_alt_streams = 5 if (_is_cuda or C._is_dcu) else 2
        self.alt_streams = (
            [device_module.Stream() for _ in range(num_alt_streams)]
            if use_stream_pool
            else None
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: DeepseekV4DecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_streams=self.alt_streams,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
        self.gemm_output_zero_allocator_size = 0
        self.hc_eps = config.hc_eps
        self.hc_mult = hc_mult = config.hc_mult
        self.norm_eps = config.rms_norm_eps
        if self.pp_group.is_last_rank:
            (
                self.hc_head_fn,
                self.hc_head_base,
                self.hc_head_scale,
            ) = make_hc_head_params(hc_mult, config.hidden_size)

        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        self.use_fused_mhc_post_pre = _is_fused_mhc_post_pre_enabled()
        if self.dsa_enable_prefill_cp:
            self.cp_size = get_parallel().attn_cp_size

        self.dspark_layers_to_capture: Optional[List[int]] = None

def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor],
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        cp_v2_active = is_cp_v2_active(forward_batch)
        use_prefill_cp = dsa_use_prefill_cp(forward_batch)
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            # Unflatten 2D PP IPC tensor back to 3D mHC shape.
            if hidden_states.ndim == 2:
                hidden_states = hidden_states.view(
                    hidden_states.shape[0], self.hc_mult, self.hidden_size
                )

        if get_parallel().attn_dp_size > 1 and get_moe_a2a_backend().is_none():
            input_ids_global = torch.empty(
                (get_global_dp_buffer_len(), 1),
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            # Token ids are replicated within an attention-TP group. Use replicate
            # gather here to avoid summing duplicated ids when attention_tp_size > 1.
            # Clone because the MAX_LEN gather may zero its local input in place.
            dp_gather_replicate(
                input_ids_global, input_ids[:, None].clone(), forward_batch
            )
            input_ids_global = input_ids_global.squeeze(-1)
        else:
            input_ids_global = input_ids

        if use_prefill_cp:
            if cp_v2_active:
                input_ids = cp_round_robin_input_ids_v2(input_ids, forward_batch)
            else:
                if self.pp_group.is_first_rank:
                    hidden_states = cp_split_and_rebuild_data(
                        forward_batch, hidden_states
                    )
                positions = cp_split_and_rebuild_position(forward_batch, positions)
                input_ids = cp_round_robin_input_ids(input_ids)
            input_ids_global = input_ids

        # Reset Compressor's per-step freqs_cis cache from any previous step.
        for _attr in ("freqs_cis_c4", "freqs_cis_c128"):
            if hasattr(forward_batch, _attr):
                delattr(forward_batch, _attr)
        capture_dspark = self.dspark_layers_to_capture is not None
        dspark_aux_hidden_states: List[torch.Tensor] = []
        # DSpark aux capture needs the per-layer eager loop (TBO's overlapped
        # execution cannot expose per-layer completed hidden states), so skip
        # TBO when capturing -- a perf-only downgrade, not a correctness one.
        if self._can_run_tbo(forward_batch) and not capture_dspark:
            # Two-batch-overlap prefill (EP / mori). Cross-layer mHC fusion is
            # disabled here (each layer self-contained), so no trailing hc_post.
            hidden_states = self._forward_layers_tbo(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        else:
            use_fused = self.use_fused_mhc_post_pre
            prev_residual, prev_post, prev_comb = None, None, None
            last_layer = None
            for i in range(self.start_layer, self.end_layer):
                layer = self.layers[i]
                last_layer = layer
                ctx = (
                    nullcontext()
                    if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:
                    hidden_states, prev_residual, prev_post, prev_comb = layer(
                        positions=positions,
                        hidden_states=hidden_states,
                        forward_batch=forward_batch,
                        input_ids=input_ids,
                        input_ids_global=input_ids_global,
                        prev_residual=prev_residual,
                        prev_post=prev_post,
                        prev_comb=prev_comb,
                    )
                if capture_dspark and i in self.dspark_layers_to_capture:
                    if use_fused:
                        completed = layer.hc_post(
                            hidden_states, prev_residual, prev_post, prev_comb
                        )
                    else:
                        completed = hidden_states
                    dspark_aux_hidden_states.append(completed.mean(dim=1))
            if use_fused and last_layer is not None:
                hidden_states = last_layer.hc_post(
                    hidden_states, prev_residual, prev_post, prev_comb
                )

        # CP all-gather only on the last PP rank; PP IPC carries CP-split tensors.
        if self.pp_group.is_last_rank and use_prefill_cp and not cp_v2_active:
            stream = torch.cuda.current_stream()
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.cp_size,
                forward_batch,
                stream,
            )
            # Gather DSpark aux tensors on the same CP token split.
            if capture_dspark:
                dspark_aux_hidden_states = [
                    cp_all_gather_rerange_output(
                        aux, self.cp_size, forward_batch, stream
                    )
                    for aux in dspark_aux_hidden_states
                ]

        if not self.pp_group.is_last_rank:
            # Flatten 3D mHC tensor for PP IPC.
            return PPProxyTensors({"hidden_states": hidden_states.flatten(1)})

        pre_hc_head = hidden_states.flatten(1)

        hidden_states = self.hc_head(
            hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base
        )
        hidden_states = self.norm(hidden_states)

        if capture_dspark:
            return (hidden_states, pre_hc_head), dspark_aux_hidden_states

        return hidden_states, pre_hc_head
