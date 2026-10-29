# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

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

from __future__ import annotations

import sglang.srt.models.deepseek_nextn as N
import sglang.srt.layers.attention.dsa.utils as D
import hcu_sglang.models.deepseek.current_v4.hcu_backend.hcu_cp_utils as C
import hcu_sglang.models.deepseek.current_v4.hcu_backend.compat as H
import sglang.srt.layers.moe.utils as M
import sglang.srt.layers.cp.utils as P


def enable_cp_v2() -> bool:
    from hcu_sglang.utils.detection import is_dcu
    from sglang.srt.utils import get_bool_env_var, is_musa, is_npu

    if is_dcu():
        return get_bool_env_var("SGLANG_ENABLE_CP_V2")
    return not (is_npu() or is_musa())


def _gather_dsa_topk_indices_for_cp(
    topk_indices: N.torch.Tensor,
    local_num_tokens: int,
    cp_size: int,
    forward_batch: N.ForwardBatch,
    stream,
) -> N.torch.Tensor:
    if (
        H.is_dsa_prefill_cp_round_robin_split()
        and topk_indices.shape[0] < local_num_tokens
    ):
        pad_rows = local_num_tokens - topk_indices.shape[0]
        topk_indices = N.torch.cat(
            [
                topk_indices,
                topk_indices.new_full((pad_rows, topk_indices.shape[1]), -1),
            ],
            dim=0,
        )
    return C.cp_all_gather_rerange_output(
        topk_indices,
        cp_size,
        forward_batch,
        stream,
    )

def __init__(
    self,
    config: N.PretrainedConfig,
    quant_config: N.Optional[N.QuantizationConfig] = None,
    prefix: str = "",
) -> None:
    super(N.DeepseekModelNextN, self).__init__()
    if N.enable_nextn_moe_bf16_cast_to_fp8(quant_config):
        # refer to real DeepSeek V3 quant config
        moe_quant_config_override = N.Fp8Config(
            is_checkpoint_fp8_serialized=True,
            weight_block_size=[128, 128],
        )
    else:
        moe_quant_config_override = None

    if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
        N.logger.debug(
            "Overriding DeepseekV3ForCausalLMNextN quant config for modelopt_fp4 Deepseek model."
        )
        quant_config = None

    self.vocab_size = config.vocab_size

    self.embed_tokens = N.VocabParallelEmbedding(
        config.vocab_size,
        config.hidden_size,
        prefix=N.add_prefix("embed_tokens", prefix),
        **N.get_embedding_tp_kwargs(),
    )

    self.enorm = N.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self.hnorm = N.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    if quant_config is not None and quant_config.get_name() == "quark":
        self.eh_proj = N.ReplicatedLinear(
            2 * config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=N.add_prefix("eh_proj", prefix),
        )
    else:
        self.eh_proj = N.nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=False
        )

    self.rot_weight = None
    if N._is_npu:
        rot_weight_path = N.get_model().model_path + "/rot.safetensors"
        if N.os.path.isfile(rot_weight_path):
            self.rot_weight = N.load_file(rot_weight_path)
            self.rot_weight = self.rot_weight["rot.weight"].npu()

    self.alt_stream = (
        N.torch.cuda.Stream()
        if N._is_cuda or N.envs.SGLANG_NPU_USE_MULTI_STREAM.get() or M.is_sbo_enabled()
        else None
    )

    layer_name = "decoder"
    if N._is_npu and (
        N.get_spec().speculative_draft_model_path == N.get_model().model_path
    ):
        layer_name = "layers." + str(config.num_hidden_layers)

    self.quant_config = quant_config
    with N.layer_stack():
        self.decoder = N.DeepseekV2DecoderLayer(
            config,
            0,
            quant_config=quant_config,
            moe_quant_config_override=moe_quant_config_override,
            is_nextn=True,
            prefix=N.add_prefix(layer_name, prefix),
            alt_stream=self.alt_stream,
            skip_rope=config.qk_rope_head_dim == 0,
        )

    self.shared_head = N.nn.Module()
    self.shared_head.norm = N.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

def forward(
    self,
    input_ids: N.torch.Tensor,
    positions: N.torch.Tensor,
    forward_batch: N.ForwardBatch,
    input_embeds: N.torch.Tensor = None,
) -> N.torch.Tensor:
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    pool = get_token_to_kv_pool()
    full_pool = getattr(pool, "full_kv_pool", pool)
    if getattr(full_pool, "is_hcu_glm5_next_pool", False):
        from sglang.srt.layers.communicator_glm5_next_cp import (
            maybe_prefetch_full_attention_kv,
        )

        maybe_prefetch_full_attention_kv(forward_batch, self.decoder.layer_id)

    exit_stack = N.ExitStack()
    if (
        N._is_npu
        and self.quant_config is None
        and N.get_model().quantization is not None
    ):
        # ascend mtp unquant
        exit_stack.enter_context(N.envs.SGLANG_DEEPEP_BF16_DISPATCH.override(True))
        exit_stack.enter_context(
            N.envs.DEEP_NORMAL_MODE_USE_INT8_QUANT.override(False)
        )

    try:
        zero_allocator = N.BumpAllocator(
            buffer_size=2,
            dtype=N.torch.float32,
            device=(
                input_embeds.device
                if input_embeds is not None
                else input_ids.device
            ),
        )

        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids, forward_batch)
        hidden_states = input_embeds

        if hidden_states.shape[0] > 0:
            previous_hidden_states = forward_batch.spec_info.hidden_states
            if self.rot_weight is not None:
                previous_hidden_states = N.torch.matmul(
                    previous_hidden_states, self.rot_weight
                )
            if N._is_cuda:
                eh_input = N.fused_eh_norm(
                    hidden_states,
                    previous_hidden_states,
                    self.enorm.weight,
                    self.hnorm.weight,
                    self.enorm.variance_epsilon,
                )
            else:
                eh_input = N.torch.cat(
                    (
                        self.enorm(hidden_states),
                        self.hnorm(previous_hidden_states),
                    ),
                    dim=-1,
                )
            if isinstance(self.eh_proj, N.ReplicatedLinear):
                hidden_states, _ = self.eh_proj(eh_input)
            else:
                hidden_states = self.eh_proj(eh_input)

        # Protected platforms retain their model-side token split.
        use_platform_cp = not enable_cp_v2() and (
            D.dsa_use_prefill_cp(forward_batch) or C.mla_use_prefill_cp(forward_batch)
        )
        if use_platform_cp:
            hidden_states = C.cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = C.cp_split_and_rebuild_position(forward_batch, positions)
        # NextN owns a separate KV/page-plan namespace even when its
        # temporary Main-KV tensor aliases the target's LayerSplit scratch.
        # Configure it before the decoder/indexer can read historical KV.
        if N.get_parallel().enable_dsa_cache_layer_split:
            from sglang.srt.layers.attention.dsa.utils import (
                maybe_prefetch_full_attention_kv as maybe_prefetch_dsa_full_kv,
            )

            maybe_prefetch_dsa_full_kv(forward_batch, 0)
        N.residual_batch.start(forward_batch)
        index_topk_share = N.IndexTopKShareState.from_mtp_carry(forward_batch)
        with N.get_global_expert_distribution_recorder().disable_this_region():
            (hidden_states, topk_indices) = self.decoder(
                positions,
                hidden_states,
                forward_batch,
                zero_allocator,
                prev_topk_indices=index_topk_share.topk_indices,
            )
        hidden_states = N.residual_batch.complete_output(hidden_states, forward_batch)
        if not forward_batch.forward_mode.is_idle():
            hidden_states = N.residual_batch.final_norm(
                hidden_states, forward_batch, self.shared_head.norm
            )

            if use_platform_cp:
                local_num_tokens = hidden_states.shape[0]
                hidden_states = C.cp_all_gather_rerange_output(
                    hidden_states,
                    N.get_parallel().attn_cp_size,
                    forward_batch,
                    N.torch.cuda.current_stream(),
                )
                if index_topk_share.should_publish and topk_indices is not None:
                    topk_indices = _gather_dsa_topk_indices_for_cp(
                        topk_indices,
                        local_num_tokens,
                        N.get_parallel().attn_cp_size,
                        forward_batch,
                        N.torch.cuda.current_stream(),
                    )
        index_topk_share.update(topk_indices)
        index_topk_share.publish()
    finally:
        exit_stack.close()

    return hidden_states
