"""MiniMax M2 HCU adaptations required by the LightOp FP8 serving path."""

from __future__ import annotations

import os
from typing import Any, Callable

import torch
import torch.nn.functional as F

from hcu_sglang.utils.detection import is_dcu

_FP32GEMM_GATE: Any = None


def _fused_rms_quant_enabled() -> bool:
    return os.environ.get("SGLANG_USE_FUSED_RMS_QUANT", "0").lower() in {"1", "true", "yes", "on"}


def _minimax_opt_enabled() -> bool:
    from sglang.srt.runtime_context import get_server_args

    return bool(getattr(get_server_args(), "minimax_opt", False))


def gate_fp32gemm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    global _FP32GEMM_GATE
    if _FP32GEMM_GATE is None:
        if os.environ.get("SGLANG_USE_FP32GEMM_GATE_CUSTOM", "0") == "1":
            try:
                import fp32gemm

                _FP32GEMM_GATE = fp32gemm
            except ImportError:
                _FP32GEMM_GATE = False
        else:
            _FP32GEMM_GATE = False

    if _FP32GEMM_GATE is not False and hidden_states.dtype == torch.bfloat16:
        return _FP32GEMM_GATE.gemm_abt(hidden_states, weight)
    return F.linear(hidden_states.float(), weight, bias=None)


def around_forward_normal(
    original: Callable[..., torch.Tensor], model, hidden_states: torch.Tensor
) -> torch.Tensor:
    if not is_dcu():
        return original(model, hidden_states)

    num_tokens, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    if hidden_states.shape[0] > 0:
        router_logits = gate_fp32gemm(hidden_states, model.gate.weight)
        topk_output = model.topk(hidden_states, router_logits)
    else:
        topk_output = model.topk.empty_topk_output(hidden_states.device)

    final_hidden_states = model.experts(hidden_states, topk_output)
    # The current stage boundary owns the TP sum of this partial MoE output.
    # Reducing here would make its declared output get summed twice.
    return final_hidden_states.view(num_tokens, hidden_dim)


def forward_normal_opt(model, hidden_states: torch.Tensor) -> torch.Tensor:
    from lightop.quant import per_token_quant_int8
    from sglang.srt.distributed import get_tp_group
    from hcu_sglang.layers.moe.minimax_m2 import (
        fused_experts_impl_int8_marlin_minimax_m2,
    )

    local_tokens, hidden_dim = hidden_states.shape
    num_tokens = local_tokens * model.tp_size
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = gate_fp32gemm(hidden_states, model.gate.weight)
    topk_output = model.topk(hidden_states, router_logits)
    topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

    tp_group = get_tp_group()
    i_q, i_s = per_token_quant_int8(hidden_states)
    i_q = tp_group.all_gather(i_q, dim=0)
    i_s = tp_group.all_gather(i_s, dim=0)
    topk_weights = tp_group.all_gather(topk_weights, dim=0)
    topk_ids = tp_group.all_gather(topk_ids, dim=0)

    final_hidden_states = fused_experts_impl_int8_marlin_minimax_m2(
        num_tokens,
        hidden_dim,
        hidden_states.device,
        hidden_states.dtype,
        model.experts.w13_weight,
        model.experts.w2_weight,
        topk_weights,
        topk_ids,
        model.experts.num_experts,
        model.experts.w13_weight_scale,
        model.experts.w2_weight_scale,
        model.experts.w13_input_scale,
        model.experts.w2_input_scale,
        i_q,
        i_s,
    )
    if model.tp_size > 1:
        chunks = final_hidden_states.chunk(model.tp_size)
        reduced = torch.empty(
            local_tokens,
            hidden_dim,
            device=final_hidden_states.device,
            dtype=final_hidden_states.dtype,
        )
        tp_group.reduce_scatter(reduced, list(chunks))
        return reduced
    return final_hidden_states.view(num_tokens, hidden_dim)


def around_moe_forward(original, model, hidden_states, forward_batch=None):
    if not is_dcu() or not _minimax_opt_enabled():
        return original(model, hidden_states, forward_batch)

    from sglang.srt.layers.moe import get_moe_a2a_backend

    backend = get_moe_a2a_backend()
    if not backend.is_deepep() and not backend.is_ascend_fuseep():
        return forward_normal_opt(model, hidden_states)
    return model.forward_deepep(hidden_states, forward_batch)


def around_forward_deepep(
    original: Callable[..., torch.Tensor], model, hidden_states, forward_batch
) -> torch.Tensor:
    if not is_dcu():
        return original(model, hidden_states, forward_batch)

    from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo

    if hidden_states.shape[0] > 0:
        router_logits = gate_fp32gemm(hidden_states, model.gate.weight)
        topk_output = model.topk(
            hidden_states,
            router_logits,
            num_token_non_padded=forward_batch.num_token_non_padded,
            expert_location_dispatch_info=ExpertLocationDispatchInfo.init_new(
                layer_id=model.layer_id,
            ),
        )
    else:
        topk_output = model.topk.empty_topk_output(device=hidden_states.device)
    return model.experts(
        hidden_states=hidden_states,
        topk_output=topk_output,
    )


def around_op_gate(original: Callable[..., Any], model, state) -> Any:
    if not is_dcu():
        return original(model, state)

    from sglang.srt.utils import is_non_idle_and_non_empty

    if is_non_idle_and_non_empty(
        state.forward_batch.forward_mode,
        state.hidden_states_mlp_input,
    ):
        state.router_logits = gate_fp32gemm(
            state.hidden_states_mlp_input,
            model.gate.weight,
        )
    else:
        state.router_logits = None


def around_rmsnorm_init(original, norm, hidden_size, num_heads, eps=1e-6):
    if not is_dcu() or not _minimax_opt_enabled():
        result = original(norm, hidden_size, num_heads, eps)
        norm.minimax_opt = False
        return result

    from torch import nn
    from sglang.srt.runtime_context import get_parallel

    nn.Module.__init__(norm)
    norm.minimax_opt = True
    norm.attn_tp_size = get_parallel().attn_tp_size
    norm.attn_tp_rank = get_parallel().attn_tp_rank
    norm.num_heads = num_heads
    norm.num_head_replicas = 1
    norm.head_dim = hidden_size // num_heads
    norm.weight = nn.Parameter(torch.ones(norm.num_heads * norm.head_dim))
    norm.weight.weight_loader = norm.weight_loader
    norm.variance_epsilon = eps


def around_rmsnorm_weight_loader(original, norm, param, loaded_weight):
    if not getattr(norm, "minimax_opt", False):
        return original(norm, param, loaded_weight)
    param.data.copy_(loaded_weight)


def around_qk_norm_forward(original, qk_norm, q, k):
    if not is_dcu() or not getattr(qk_norm._q_norm, "minimax_opt", False):
        return original(qk_norm, q, k)

    from lightop.rmsnorm import fused_rms_norm_contiguous

    q = q.contiguous()
    k = k.contiguous()
    q = fused_rms_norm_contiguous(
        torch.empty_like(q),
        q,
        qk_norm._q_norm.weight,
        qk_norm._eps,
    )
    k = fused_rms_norm_contiguous(
        torch.empty_like(k),
        k,
        qk_norm._k_norm.weight,
        qk_norm._eps,
    )
    return q, k


def around_attention_init(
    original, attention, config, layer_id=0, quant_config=None, prefix=""
):
    if not is_dcu() or not _minimax_opt_enabled():
        result = original(attention, config, layer_id, quant_config, prefix)
        attention.rms_norm_eps = config.rms_norm_eps
        return result

    from torch import nn
    from sglang.srt.layers.linear import QKVParallelLinear, ReplicatedLinear
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.layers.rotary_embedding import get_rope
    from sglang.srt.models.minimax_m2 import (
        MiniMaxM2QKRMSNorm,
        MiniMaxM2RMSNormTP,
        get_rope_config,
    )
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.utils import add_prefix

    nn.Module.__init__(attention)
    attention.hidden_size = config.hidden_size
    attention.rms_norm_eps = config.rms_norm_eps
    attn_tp_size = get_parallel().attn_tp_size
    attention.total_num_heads = config.num_attention_heads
    if attention.total_num_heads % attn_tp_size:
        raise ValueError("MiniMax attention heads must be divisible by attention TP")
    attention.num_heads = attention.total_num_heads // attn_tp_size
    attention.total_num_kv_heads = config.num_key_value_heads
    if attention.total_num_kv_heads >= attn_tp_size:
        if attention.total_num_kv_heads % attn_tp_size:
            raise ValueError("MiniMax KV heads must be divisible by attention TP")
    elif attn_tp_size % attention.total_num_kv_heads:
        raise ValueError("Attention TP must be divisible by replicated KV heads")
    attention.num_kv_heads = max(1, attention.total_num_kv_heads // attn_tp_size)
    attention.head_dim = getattr(
        config,
        "head_dim",
        attention.hidden_size // attention.total_num_heads,
    )
    attention.q_size = attention.num_heads * attention.head_dim
    attention.kv_size = attention.num_kv_heads * attention.head_dim
    attention.q_size_full = attention.total_num_heads * attention.head_dim
    attention.kv_size_full = attention.total_num_kv_heads * attention.head_dim
    attention.scaling = attention.head_dim**-0.5
    attention.rope_theta, attention.rope_scaling = get_rope_config(config)
    attention.max_position_embeddings = getattr(
        config,
        "max_position_embeddings",
        8192,
    )
    attention.rotary_dim = getattr(config, "rotary_dim", attention.head_dim)
    attention.use_qk_norm = getattr(config, "use_qk_norm", False)
    attention.qk_norm_type = getattr(config, "qk_norm_type", "per_layer")
    attention.qkv_proj = QKVParallelLinear(
        attention.hidden_size,
        attention.head_dim,
        attention.total_num_heads,
        attention.total_num_kv_heads,
        bias=False,
        quant_config=quant_config,
        parallel_group="replicated",
        prefix=add_prefix("qkv_proj", prefix),
    )
    attention.qkv_proj.eps = config.rms_norm_eps
    attention.o_proj = ReplicatedLinear(
        attention.total_num_heads * attention.head_dim,
        attention.hidden_size,
        bias=False,
        quant_config=quant_config,
        prefix=add_prefix("o_proj", prefix),
    )
    attention.rotary_emb = get_rope(
        attention.head_dim,
        rotary_dim=attention.rotary_dim,
        max_position=attention.max_position_embeddings,
        base=attention.rope_theta,
        rope_scaling=attention.rope_scaling,
    )
    if attention.use_qk_norm:
        if attention.qk_norm_type != "per_layer":
            raise ValueError(f"Unsupported qk_norm_type: {attention.qk_norm_type}")
        attention.q_norm = MiniMaxM2RMSNormTP(
            attention.q_size_full,
            num_heads=attention.total_num_heads,
            eps=config.rms_norm_eps,
        )
        attention.k_norm = MiniMaxM2RMSNormTP(
            attention.kv_size_full,
            num_heads=attention.total_num_kv_heads,
            eps=config.rms_norm_eps,
        )
        attention.qk_norm_impl = MiniMaxM2QKRMSNorm(
            attention.q_norm,
            attention.k_norm,
        )
    attention.attn = RadixAttention(
        attention.num_heads,
        attention.head_dim,
        attention.scaling,
        num_kv_heads=attention.num_kv_heads,
        layer_id=layer_id,
        quant_config=quant_config,
        prefix=add_prefix("attn", prefix),
    )


def attention_forward_prepare_opt(
    attention,
    positions,
    hidden_states,
    forward_batch,
    rms_weight=None,
    residual=None,
):
    if hidden_states.shape[0] == 0:
        return hidden_states, forward_batch, None
    from sglang.srt.runtime_context import get_parallel

    attn_tp_group = get_parallel().attn_tp_group
    attn_tp_size = attn_tp_group.world_size
    tokens_per_rank = hidden_states.shape[0]
    if rms_weight is None:
        qkv, _ = attention.qkv_proj(hidden_states)
    else:
        qkv, _ = attention.qkv_proj(
            hidden_states,
            rms_weight=rms_weight,
            residual=residual,
        )
    q, k, v = qkv.split(
        [attention.q_size_full, attention.kv_size_full, attention.kv_size_full],
        dim=-1,
    )
    if attention.use_qk_norm:
        q, k = attention.qk_norm_impl.forward(q.contiguous(), k.contiguous())
    q, k = attention.rotary_emb.forward_native(positions, q, k)
    q = q.view(tokens_per_rank, attn_tp_size, attention.q_size)
    if attention.total_num_kv_heads >= attn_tp_size:
        k = k.view(tokens_per_rank, attn_tp_size, attention.kv_size)
        v = v.view(tokens_per_rank, attn_tp_size, attention.kv_size)
    else:
        kv_replicas = attn_tp_size // attention.total_num_kv_heads
        k = k.view(tokens_per_rank, attention.total_num_kv_heads, attention.head_dim)
        v = v.view(tokens_per_rank, attention.total_num_kv_heads, attention.head_dim)
        k = k.repeat_interleave(kv_replicas, dim=1)
        v = v.repeat_interleave(kv_replicas, dim=1)
    qkv_by_head_rank = torch.cat([q, k, v], dim=-1).permute(1, 0, 2).contiguous()
    qkv_restored = torch.empty_like(qkv_by_head_rank)
    attn_tp_group.all_to_all_single(qkv_restored, qkv_by_head_rank)
    qkv_restored = qkv_restored.view(
        tokens_per_rank * attn_tp_size,
        attention.q_size + 2 * attention.kv_size,
    )
    q, k, v = qkv_restored.split(
        [attention.q_size, attention.kv_size, attention.kv_size],
        dim=-1,
    )
    return None, forward_batch, (q, k, v, forward_batch)


def around_attention_forward_prepare(
    original,
    attention,
    positions,
    hidden_states,
    forward_batch,
    rms_weight=None,
    residual=None,
):
    if not is_dcu() or not _minimax_opt_enabled():
        if is_dcu() and rms_weight is not None:
            from hcu_sglang.layers.linear_fusion import normalize_projection_input
            hidden_states = normalize_projection_input(
                hidden_states, rms_weight, residual,
                getattr(attention, "rms_norm_eps", 1e-6),
            )
        return original(attention, positions, hidden_states, forward_batch)
    return attention_forward_prepare_opt(
        attention,
        positions,
        hidden_states,
        forward_batch,
        rms_weight,
        residual,
    )


def around_attention_forward_core(original, attention, intermediate_state):
    if not is_dcu() or not _minimax_opt_enabled():
        return original(attention, intermediate_state)
    hidden_states, forward_batch, inner_state = intermediate_state
    if inner_state is None:
        return hidden_states
    from sglang.srt.runtime_context import get_parallel

    attn_output = attention.attn(*inner_state)
    attn_tp_group = get_parallel().attn_tp_group
    attn_tp_size = attn_tp_group.world_size
    tokens_per_rank = attn_output.shape[0] // attn_tp_size
    attn_output = attn_output.view(attn_tp_size, tokens_per_rank, attention.q_size)
    restored = torch.empty_like(attn_output)
    attn_tp_group.all_to_all_single(restored, attn_output)
    restored = (
        restored.permute(1, 0, 2)
        .contiguous()
        .view(tokens_per_rank, attn_tp_size * attention.q_size)
    )
    output, _ = attention.o_proj(restored)
    return output


def around_attention_forward(
    original,
    attention,
    positions,
    hidden_states,
    forward_batch,
    rms_weight=None,
    residual=None,
):
    if not is_dcu() or not _minimax_opt_enabled():
        if is_dcu() and rms_weight is not None:
            from hcu_sglang.layers.linear_fusion import normalize_projection_input
            hidden_states = normalize_projection_input(
                hidden_states, rms_weight, residual,
                getattr(attention, "rms_norm_eps", 1e-6),
            )
        return original(attention, positions, hidden_states, forward_batch)
    state = attention.forward_prepare(
        positions,
        hidden_states,
        forward_batch,
        rms_weight,
        residual,
    )
    return attention.forward_core(state)


def decoder_forward_opt(layer, positions, hidden_states, forward_batch, residual):
    if hidden_states.shape[0] == 0:
        return hidden_states, hidden_states
    if residual is None:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
    else:
        hidden_states, residual = layer.input_layernorm(hidden_states, residual, None)
    hidden_states = layer.self_attn(
        positions=positions,
        hidden_states=hidden_states,
        forward_batch=forward_batch,
    )
    hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
    hidden_states = layer.block_sparse_moe(hidden_states, forward_batch)
    return hidden_states, residual


def around_decoder_forward(
    original, layer, positions, hidden_states, forward_batch,
    captured_last_layer_outputs=None,
):
    if not is_dcu():
        return original(layer, positions, hidden_states, forward_batch,
                        captured_last_layer_outputs)
    if not _minimax_opt_enabled() and not _fused_rms_quant_enabled():
        return original(layer, positions, hidden_states, forward_batch,
                        captured_last_layer_outputs)
    from hcu_sglang.models.minimax_m2_methods import decoder_forward
    return decoder_forward(layer, positions, hidden_states, forward_batch,
                           captured_last_layer_outputs)


def around_model_forward(
    original, model, input_ids, positions, forward_batch,
    input_embeds=None, pp_proxy_tensors=None,
):
    if not is_dcu() or not _minimax_opt_enabled():
        return original(model, input_ids, positions, forward_batch,
                        input_embeds, pp_proxy_tensors)
    from hcu_sglang.models.minimax_m2_methods import model_forward
    return model_forward(model, input_ids, positions, forward_batch,
                         input_embeds, pp_proxy_tensors)


def around_cuda_graph_batch_size_alignment(
    original, *, gathered_buffer_required=None
):
    # Keep overlap/CP alignment owned by the official implementation. HCU's
    # MiniMax optimization additionally requires TP-aligned graph batches.
    if is_dcu() and _minimax_opt_enabled():
        gathered_buffer_required = True
    return original(gathered_buffer_required=gathered_buffer_required)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    target = "sglang.srt.models.minimax_m2.MiniMaxM2MoE"
    HookRegistry.register(
        f"{target}.forward",
        around_moe_forward,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{target}.forward_normal",
        around_forward_normal,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{target}.forward_deepep",
        around_forward_deepep,
        HookType.AROUND,
    )
    rms_target = "sglang.srt.models.minimax_m2.MiniMaxM2RMSNormTP"
    HookRegistry.register(
        f"{rms_target}.__init__",
        around_rmsnorm_init,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{rms_target}.weight_loader",
        around_rmsnorm_weight_loader,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.models.minimax_m2.MiniMaxM2QKRMSNorm.forward",
        around_qk_norm_forward,
        HookType.AROUND,
    )
    attention_target = "sglang.srt.models.minimax_m2.MiniMaxM2Attention"
    HookRegistry.register(
        f"{attention_target}.__init__",
        around_attention_init,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{attention_target}.forward_prepare",
        around_attention_forward_prepare,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{attention_target}.forward_core",
        around_attention_forward_core,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{attention_target}.forward",
        around_attention_forward,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.models.minimax_m2.MiniMaxM2DecoderLayer.forward",
        around_decoder_forward,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.models.minimax_m2.MiniMaxM2Model.forward",
        around_model_forward,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.utils.common.get_cuda_graph_batch_size_alignment",
        around_cuda_graph_batch_size_alignment,
        HookType.AROUND,
    )
