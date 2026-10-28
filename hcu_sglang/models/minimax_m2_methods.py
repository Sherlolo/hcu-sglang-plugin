# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""MiniMax boundary-stream methods extracted from pinned HCU 5f67ce01c3.

Only method names/calls are rewritten. Runtime globals come from the pinned
community module and explicit plugin-owned detection; no fork imports.
"""
from __future__ import annotations
import os
from sglang.srt.models import minimax_m2 as _upstream

globals().update({k: v for k, v in vars(_upstream).items() if not k.startswith("__")})
from sglang.srt.runtime_context import get_server_args
from hcu_sglang.utils.detection import is_dcu

_is_hcu = is_dcu()
_use_fused_rms_quant = _is_hcu and os.environ.get("SGLANG_USE_FUSED_RMS_QUANT", "0").lower() in {"1", "true", "yes", "on"}


def decoder_forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        captured_last_layer_outputs: Optional[AuxHiddenStateAccumulator] = None,
    ) -> torch.Tensor:
        if _is_hcu and get_server_args().minimax_opt:
            stream = residual_batch.stream_of(forward_batch)
            hidden_states, residual = stream.export(hidden_states)
            hidden_states, residual = decoder_forward_opt(self,
                positions, hidden_states, forward_batch, residual
            )
            stream.write(residual)
            return hidden_states

        if _use_fused_rms_quant:
            stream = residual_batch.stream_of(forward_batch)
            hidden_states, residual = stream.export(hidden_states)
            pre_norm = hidden_states.clone() if residual is None else residual
            hidden_states = self.attn_boundary.prepare(
                hidden_states, forward_batch, hcu_skip_layernorm=True
            )
            if residual is not None and residual.shape[0] != hidden_states.shape[0]:
                raise ValueError("HCU fused RMS quant requires matching token rows")
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                rms_weight=self.input_layernorm.weight,
                residual=residual,
            )
            stream.write(pre_norm)
            hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
            hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)
            with self.ffn_boundary.exit(forward_batch) as ffn_exit:
                hidden_states = self.block_sparse_moe(hidden_states, forward_batch)
            return ffn_exit.finish(hidden_states)

        # Self Attention
        hidden_states = self.attn_boundary.prepare(
            hidden_states,
            forward_batch,
            capture_gathered=captured_last_layer_outputs,
        )
        if not forward_batch.forward_mode.is_idle():
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # Fully Connected (MLP or MoE)

        hidden_states = self.attn_boundary.finish(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.prepare(hidden_states, forward_batch)

        hidden_states = self.block_sparse_moe(hidden_states, forward_batch)
        hidden_states = self.ffn_boundary.finish(hidden_states, forward_batch)

        return hidden_states

def decoder_forward_opt(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if hidden_states.shape[0] == 0:
            return hidden_states, hidden_states

        if _use_fused_rms_quant:
            pre_norm = hidden_states.clone() if residual is None else residual
            if residual is not None:
                assert residual.shape[0] == hidden_states.shape[0], (
                    f"Fused RMS quant requires hidden_states ({hidden_states.shape[0]}) "
                    f"and residual ({residual.shape[0]}) to have the same token count."
                )
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
                rms_weight=self.input_layernorm.weight,
                residual=residual,
            )
            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, pre_norm
            )
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(
                    hidden_states, residual, None
                )

            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

            hidden_states, residual = self.post_attention_layernorm(
                hidden_states, residual
            )
        hidden_states = self.block_sparse_moe(hidden_states, forward_batch)
        return hidden_states, residual

def model_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors, Tuple[torch.Tensor, list[torch.Tensor]]]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.get_input_embeddings(input_ids)
            else:
                hidden_states = input_embeds
            residual_batch.start(forward_batch)
        else:
            assert pp_proxy_tensors is not None
            hidden_states = self.layers[self.start_layer].attn_boundary.from_pp(
                pp_proxy_tensors, forward_batch
            )

        use_minimax_sp = _is_hcu and get_server_args().minimax_opt
        if use_minimax_sp:
            attn_tp_group = get_parallel().attn_tp_group
            attn_tp_size = attn_tp_group.world_size
            attn_tp_rank = attn_tp_group.rank_in_group
            num_tokens = hidden_states.shape[0]
            tokens_per_rank = num_tokens // attn_tp_size
            start = attn_tp_rank * tokens_per_rank
            hidden_states = hidden_states[start : start + tokens_per_rank]
            positions = positions[start : start + tokens_per_rank]

        aux_hidden_states = AuxHiddenStateList()
        if forward_batch.can_run_tbo and not use_minimax_sp:
            hidden_states = model_forward_stages(
                layers=self.layers,
                enable_tbo=True,
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
            )
        else:
            for i in range(self.start_layer, self.end_layer):
                ctx = (
                    nullcontext()
                    if check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:
                    layer = self.layers[i]
                    hidden_states = layer(
                        positions=positions,
                        forward_batch=forward_batch,
                        hidden_states=hidden_states,
                        captured_last_layer_outputs=aux_hidden_states
                        if i in self.layers_to_capture
                        else None,
                    )

        if not self.pp_group.is_last_rank:
            return residual_batch.to_pp(hidden_states, forward_batch)
        hidden_states = residual_batch.final_norm(
            hidden_states, forward_batch, self.norm, skip_empty=True
        )

        if use_minimax_sp:
            attn_tp_group = get_parallel().attn_tp_group
            attn_tp_size = attn_tp_group.world_size
            hidden_states_gathered = torch.empty(
                hidden_states.shape[0] * attn_tp_size,
                hidden_states.shape[1],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            attn_tp_group.all_gather_into_tensor(hidden_states_gathered, hidden_states)
            hidden_states = hidden_states_gathered

        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states
