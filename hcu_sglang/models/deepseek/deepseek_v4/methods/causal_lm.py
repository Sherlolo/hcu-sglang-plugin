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
        super(DeepseekV4ForCausalLM, self).__init__()
        # DeepseekV4 enables, by default, the CK w8a8-block GEMM (MLA proj) and the
        # batched/contiguous-load rope kernels (faster on gfx95; .
        # Module-level toggles default OFF; flipped True here for DSV4
        if _is_hip:
            from sglang.kernels.ops.attention.deepseek_v4_rope import set_batched_rope
            from sglang.srt.layers.quantization.fp8_utils import set_force_ck_w8a8

            set_force_ck_w8a8(True)
            set_batched_rope(True)
        self.config = config
        self.tp_size = get_parallel().tp_size
        self.quant_config = quant_config
        self.determine_num_fused_shared_experts()
        self.model = DeepseekV4Model(
            config, quant_config, prefix=add_prefix("model", prefix)
        )
        self.pp_group = get_pp_group()
        if self.pp_group.is_last_rank:
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                    use_attn_tp_group=get_parallel().enable_dp_lm_head,
                )
        else:
            self.lm_head = PPMissingLayer()
        self.logits_processor = LogitsProcessor(config)
        self.capture_aux_hidden_states = False
        get_attn_tp_context().init_context(config.q_lora_rank, is_dsa=True)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: self.model.layers[layer_id].mlp.get_moe_weights()
                for layer_id in range(self.model.start_layer, self.model.end_layer)
                if isinstance(
                    self.model.layers[layer_id].mlp, deepseek_v2.DeepseekV2MoE
                )
            }
        )

        # Expose start_layer/end_layer for model_runner PP support
        self.start_layer = self.model.start_layer
        self.end_layer = self.model.end_layer

        self.dsa_enable_prefill_cp = is_dsa_enable_prefill_cp()
        if self.dsa_enable_prefill_cp:
            self.cp_rank = get_parallel().attn_cp_rank
            self.cp_size = get_parallel().attn_cp_size

        # update_weights_from_disk/_tensor/_distributed re-enter load_weights
        # mid-serving (RL refit sends many partial batches); the prewarm and
        # its barrier must only run on the first (startup) load.
        self._mhc_prewarmed_at_load = False

def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        if self.dsa_enable_prefill_cp:
            if can_dsa_cp_split(len(input_ids), self.cp_size, True, forward_batch):
                forward_batch.attn_cp_metadata = prepare_context_parallel_metadata(
                    len(input_ids),
                    self.cp_rank,
                    self.cp_size,
                    forward_batch.seq_lens_cpu.tolist(),
                    extend_seqs_len=forward_batch.extend_seq_lens_cpu,
                )
                if is_dsa_prefill_cp_round_robin_split():
                    attn_backend = get_attn_backend()
                    metadata = attn_backend.forward_metadata
                    core_meta = metadata.core_attn_metadata
                    core_meta.apply_cp_reindex()
                    core_meta.init_flashmla_related(is_prefill=True)
                    if metadata.indexer_metadata is not None:
                        metadata.indexer_metadata = (
                            attn_backend.init_forward_metadata_indexer(core_meta)
                        )

        with get_attn_tp_context().maybe_input_scattered(forward_batch):
            hidden_states = self.model.forward(
                input_ids, positions, forward_batch, input_embeds, pp_proxy_tensors
            )
        if not self.pp_group.is_last_rank:
            return hidden_states

        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        hidden_states, pre_hc_head = hidden_states

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
            hidden_states_before_norm=(
                None if aux_hidden_states is not None else pre_hc_head
            ),
        )

def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]], is_nextn=False):
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()

        if is_nextn:
            if hasattr(self.config, "num_nextn_predict_layers"):
                num_nextn_layers = self.config.num_nextn_predict_layers
                assert num_nextn_layers == 1, "Only 1 nextn layer is supported"
                nextn_layer_id = (
                    0
                    if self.config.num_hidden_layers == 1
                    else self.config.num_hidden_layers
                )
            else:
                raise ValueError("num_nextn_predict_layers is not in the config")

        if not envs.SGLANG_OPT_FP8_WO_A_GEMM.get():
            weights = _dequant_fp8_wo_a_streaming(weights)

        stacked_params_mapping = DEEPSEEK_V4_STACKED_PARAMS_MAPPING

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts + self.num_fused_shared_experts,
        )

        if is_wint4afp8_or_wint4a16_config(self.quant_config):
            expert_params_mapping += FusedMoE.make_expert_input_scale_params_mapping(
                num_experts=self.config.n_routed_experts
            )

        cache_compressor_weight = {}
        COMPRESSOR_PART = ".compressor.w"

        fuse_wqa_wkv = envs.SGLANG_OPT_FUSE_WQA_WKV.get()
        cache_wqkv_a_weight: dict[str, dict[str, torch.Tensor]] = {}

        def auto_weight_loader(module):
            return getattr(module, "weight_loader", default_weight_loader)

        def maybe_remap_compressed_tensors_scale_name(param_name: str) -> str:
            # Compressed-tensors checkpoints can expose *_scale_inv while the
            # materialized layer parameter is named *_scale. The generic
            # DeepSeek-V4 config alias does not expose this mapping itself.
            candidate_names = []
            if param_name.endswith(".weight_scale_inv"):
                candidate_names.append(
                    param_name.replace(".weight_scale_inv", ".weight_scale")
                )
            if param_name.endswith("_weight_scale_inv"):
                candidate_names.append(
                    param_name.replace("_weight_scale_inv", "_weight_scale")
                )
            if (
                _FP8_WO_A_GEMM
                and param_name.endswith(".scale")
                and not param_name.endswith(".weight_scale")
            ):
                base = param_name[: -len(".scale")]
                candidate_names.extend(
                    [base + ".weight_scale_inv", base + ".weight_scale"]
                )

            for candidate_name in candidate_names:
                if candidate_name in params_dict:
                    return candidate_name
            return param_name

        if is_nextn:
            nextn_layer_prefix = f"model.layers.{nextn_layer_id}"
            nextn_spec_weight_names_out_of_layer = [
                "shared_head.norm",
                "shared_head.head",
                "embed_tokens",
                ".e_proj",
                "h_proj",
                "enorm",
                "hnorm",
                "hc_head_base",
                "hc_head_fn",
                "hc_head_scale",
            ]

        if self.num_fused_shared_experts > 0:
            assert self.num_fused_shared_experts == 1
            log_info_on_rank0(logger, "Shared experts fusion optimization enabled.")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            weight_names = []
            for name, loaded_weight in weights:
                if (
                    _FP8_WO_A_GEMM
                    and name.endswith(".wo_a.weight")
                    and loaded_weight.dtype != torch.float8_e4m3fn
                ):
                    raise ValueError(
                        f"SGLANG_OPT_FP8_WO_A_GEMM is enabled but {name} has "
                        f"dtype {loaded_weight.dtype}, expected "
                        "torch.float8_e4m3fn. This checkpoint does not provide "
                        "a supported fp8-quantized wo_a; rerun with "
                        "SGLANG_OPT_FP8_WO_A_GEMM=0."
                    )
                try:
                    use_async_loading = should_async_load(loaded_weight)

                    name = self.remap_weight_name_to_dpsk_hf_format(
                        name,
                        is_nextn=is_nextn,
                        num_hidden_layers=self.config.num_hidden_layers,
                    )

                    layer_id = get_layer_id(name)
                    if (
                        layer_id is not None
                        and hasattr(self.model, "start_layer")
                        and (
                            layer_id < self.model.start_layer
                            or layer_id >= self.model.end_layer
                        )
                    ):
                        continue
                    if (
                        self.num_fused_shared_experts > 0
                        and "mlp.shared_experts" in name
                    ):
                        name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts}",
                        )

                    weight_names.append(name)

                    if not is_nextn:
                        if hasattr(self.config, "num_nextn_predict_layers"):
                            num_nextn_layers = self.config.num_nextn_predict_layers
                            if num_nextn_layers > 0 and name.startswith("model.layers"):
                                name_list = name.split(".")
                                if (
                                    len(name_list) >= 3
                                    and int(name_list[2])
                                    >= self.config.num_hidden_layers
                                ):
                                    continue

                            if name.startswith("mtp"):
                                continue
                    else:
                        if "shared_head.head" in name or "embed_tokens" in name:
                            continue

                        if not name.startswith(nextn_layer_prefix):
                            continue

                        in_decoder = True
                        for weight_name in nextn_spec_weight_names_out_of_layer:
                            if weight_name in name:
                                in_decoder = False
                                name = name.replace(nextn_layer_prefix, "model")
                                break

                        if in_decoder:
                            name = name.replace(nextn_layer_prefix, "model.decoder")

                    if "rotary_emb.inv_freq" in name:
                        continue
                    for param_name, weight_name, shard_id in stacked_params_mapping:
                        if weight_name not in name:
                            continue
                        if _is_npu:
                            name = name.replace("weight_packed", "weight")
                        if ("mlp.experts." in name) and name not in params_dict:
                            continue
                        name = name.replace(weight_name, param_name)
                        if name not in params_dict:
                            name = maybe_remap_compressed_tensors_scale_name(name)
                        if name.endswith(".bias") and name not in params_dict:
                            continue
                        if name not in params_dict and name.startswith("mtp"):
                            break
                        param = params_dict[name]
                        weight_loader = param.weight_loader
                        maybe_executor_submit(
                            executor=executor,
                            futures=futures,
                            use_async=use_async_loading,
                            func=weight_loader,
                            func_args=(param, loaded_weight, shard_id),
                        )
                        loaded_params.add(name)
                        break
                    else:
                        skip_unmaterialized_expert_param = False
                        for mapping in expert_params_mapping:
                            param_name, weight_name, expert_id, shard_id = mapping
                            if weight_name not in name:
                                continue
                            if _is_npu:
                                name = name.replace("weight_packed", "weight")
                            resolved_name = name.replace(weight_name, param_name)
                            if resolved_name not in params_dict:
                                resolved_name = maybe_remap_compressed_tensors_scale_name(
                                    resolved_name
                                )
                            if resolved_name not in params_dict:
                                skip_unmaterialized_expert_param = True
                                continue
                            param = params_dict[resolved_name]
                            weight_loader = param.weight_loader
                            maybe_executor_submit(
                                executor=executor,
                                futures=futures,
                                use_async=use_async_loading,
                                func=weight_loader,
                                func_args=(
                                    param,
                                    loaded_weight,
                                    resolved_name,
                                ),
                                func_kwargs={
                                    "shard_id": shard_id,
                                    "expert_id": expert_id,
                                },
                            )
                            loaded_params.add(resolved_name)
                            break
                        else:
                            if skip_unmaterialized_expert_param:
                                continue
                            if name.endswith(".bias") and name not in params_dict:
                                continue
                            if (
                                ".embed_tokens." in name
                                and not self.pp_group.is_first_rank
                            ):
                                continue
                            if (
                                name == "model.norm.weight"
                                and not self.pp_group.is_last_rank
                            ):
                                continue
                            if (
                                name.startswith("model.hc_head_")
                                or name == "lm_head.weight"
                            ) and not self.pp_group.is_last_rank:
                                continue
                            elif COMPRESSOR_PART in name:
                                is_kv = name.endswith(".wkv.weight")
                                is_wgate = name.endswith(".wgate.weight")
                                assert is_kv != is_wgate
                                key = name.rsplit(".", 2)[0]
                                assert key.endswith(".compressor")
                                if key not in cache_compressor_weight:
                                    cache_compressor_weight[key] = (
                                        is_kv,
                                        _clone_if_runai_streamed_tensor(loaded_weight),
                                    )
                                else:
                                    assert key in cache_compressor_weight
                                    cached_is_kv, cached_weight = (
                                        cache_compressor_weight[key]
                                    )
                                    assert cached_is_kv != is_kv
                                    kv = loaded_weight if is_kv else cached_weight
                                    wgate = loaded_weight if is_wgate else cached_weight
                                    fused_weight = torch.cat([kv, wgate], dim=0)
                                    param_name = key + ".wkv_gate.weight"
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_compressor_weight.pop(key)
                            elif fuse_wqa_wkv and (
                                name.endswith(".wq_a.weight")
                                or name.endswith(".wq_a.weight_scale_inv")
                                or name.endswith(".wq_a.weight_scale")
                                or name.endswith(".wkv.weight")
                                or name.endswith(".wkv.weight_scale_inv")
                                or name.endswith(".wkv.weight_scale")
                            ):
                                is_q = ".wq_a." in name
                                param_name = name.replace(
                                    ".wq_a." if is_q else ".wkv.", ".wqkv_a."
                                )
                                cache_key = param_name
                                bucket = cache_wqkv_a_weight.setdefault(cache_key, {})
                                shard_key = "q" if is_q else "kv"
                                assert (
                                    shard_key not in bucket
                                ), f"duplicate shard {shard_key} for {param_name}"
                                bucket[shard_key] = _clone_if_runai_streamed_tensor(
                                    loaded_weight
                                )
                                if len(bucket) == 2:
                                    fused_weight = torch.cat(
                                        [bucket["q"], bucket["kv"]], dim=0
                                    )
                                    param_name = maybe_remap_compressed_tensors_scale_name(
                                        param_name
                                    )
                                    param = params_dict[param_name]
                                    weight_loader = auto_weight_loader(param)
                                    maybe_executor_submit(
                                        executor=executor,
                                        futures=futures,
                                        use_async=use_async_loading,
                                        func=weight_loader,
                                        func_args=(param, fused_weight),
                                    )
                                    loaded_params.add(param_name)
                                    cache_wqkv_a_weight.pop(cache_key)
                            else:
                                if (
                                    "k_scale" in name or "v_scale" in name
                                ) and name not in params_dict:
                                    for scale in ["k_scale", "v_scale"]:
                                        if scale in name:
                                            name = name.replace(
                                                f"{scale[0]}_proj", "attn_mqa"
                                            )
                                            break
                                if name not in params_dict:
                                    name = maybe_remap_compressed_tensors_scale_name(name)
                                if name not in params_dict:
                                    if not name.startswith("mtp"):
                                        logger.warning(
                                            f"{name} not found in params_dict."
                                        )
                                    continue
                                param = params_dict[name]

                                weight_loader = auto_weight_loader(param)
                                maybe_executor_submit(
                                    executor=executor,
                                    futures=futures,
                                    use_async=use_async_loading,
                                    func=weight_loader,
                                    func_args=(param, loaded_weight),
                                )
                                loaded_params.add(name)
                except Exception as e:
                    context = f"name={name!r} loaded_weight.shape={loaded_weight.shape}"
                    if hasattr(e, "add_note"):
                        e.add_note(context)
                    raise

            for future in concurrent.futures.as_completed(futures):
                future.result()

        assert len(cache_compressor_weight) == 0
        assert len(cache_wqkv_a_weight) == 0, cache_wqkv_a_weight.keys()
        unloaded_params = params_dict.keys() - loaded_params

        skipped_checking_patterns = [
            "attn_mqa.k_scale",
            "attn_mqa.v_scale",
            "blockscale_swizzled",
        ]
        if not self.pp_group.is_first_rank:
            skipped_checking_patterns.append("embed_tokens")
        if not self.pp_group.is_last_rank:
            skipped_checking_patterns.append("model.norm.")
            skipped_checking_patterns.extend(["lm_head", "hc_head_"])
        if is_nextn:
            skipped_checking_patterns.extend(["lm_head", "embed_tokens"])
        unloaded_params = {
            p
            for p in unloaded_params
            if all(
                skipped_checking_pattern not in p
                for skipped_checking_pattern in skipped_checking_patterns
            )
        }
        if unloaded_params:
            logger.warning(
                f"Some weights are not initialized from checkpoints: {unloaded_params}"
            )

        self.post_load_weights(is_nextn=is_nextn, weight_names=weight_names)

        if not is_nextn:
            self._prewarm_mhc_kernels()

def post_load_weights(self, is_nextn=False, weight_names=None):
        if _FP8_WO_A_GEMM:
            self._setup_fp8_wo_a_scales(is_nextn)

        if is_nextn:
            return
        for layer_id in range(self.model.start_layer, self.model.end_layer):
            layer = self.model.layers[layer_id]
            self_attn = layer.self_attn
            if (
                self_attn.compress_ratio in (4, 128)
                and not self_attn.compressor.ape_converted
            ):
                self_attn.compressor.apply_ape_hotfix()
            if (
                self_attn.compress_ratio == 4
                and not self_attn.indexer.compressor.ape_converted
            ):
                self_attn.indexer.compressor.apply_ape_hotfix()
            layer.refresh_mhc_norm_weight_cache()
