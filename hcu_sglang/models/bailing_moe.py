"""Bailing-MoE model hooks for DCU fused RMSNorm + RoPE KV-store."""

import logging

import torch
from torch import nn

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.runtime_context import get_exec
from hcu_sglang.models.fused_rotary import (
    fused_rms_rotary_kv_store,
    use_fused_rms_rotary,
)
from hcu_sglang.utils.detection import is_dcu
from hcu_sglang.utils.module_patch import get_original_attr

logger = logging.getLogger(__name__)

_MODULE = "sglang.srt.models.bailing_moe"
_ATTENTION_TARGET = f"{_MODULE}.BailingMoEAttention.forward"
_MLP_FORWARD_TARGET = f"{_MODULE}.BailingMoEMLP.forward"
_GATE_INIT_TARGET = f"{_MODULE}.BailingMoEGate.__init__"
_SPARSE_MOE_INIT_TARGET = f"{_MODULE}.BailingMoESparseMoeBlock.__init__"
_LOAD_WEIGHTS_TARGET = f"{_MODULE}.BailingMoEForCausalLM.load_weights"


def register() -> None:
    targets = (
        (_ATTENTION_TARGET, replacement_forward, HookType.REPLACE),
        (_MLP_FORWARD_TARGET, replacement_mlp_forward, HookType.AROUND),
        (_GATE_INIT_TARGET, replacement_gate_init, HookType.AROUND),
        (_SPARSE_MOE_INIT_TARGET, replacement_sparse_moe_init, HookType.AROUND),
        (_LOAD_WEIGHTS_TARGET, replacement_load_weights, HookType.AROUND),
    )
    for target, hook, hook_type in targets:
        logger.info("[hcu_sglang] register %s hook: %s", hook_type.name, target)
        HookRegistry.register(target, hook, hook_type)


def _use_fused_bailing_silu_mul_fp8_quant() -> bool:
    return is_dcu()


def _num_fused_shared_experts(config) -> int:
    if not is_dcu():
        return 0
    try:
        from sglang.srt.layers.moe.utils import get_moe_a2a_backend

        if (
            get_exec().moe.disable_shared_experts_fusion
            or get_moe_a2a_backend().is_deepep()
        ):
            return 0
    except ImportError as exc:
        logger.warning(
            "[hcu_sglang] bailing fused shared experts disabled: %s", exc
        )
        return 0
    return getattr(config, "num_shared_experts", 0) or 0


def replacement_mlp_forward(
    original_fn,
    self,
    hidden_states,
    forward_batch=None,
    should_allreduce_fusion=False,
    use_reduce_scatter=False,
):
    if not _use_fused_bailing_silu_mul_fp8_quant():
        return original_fn(
            self,
            hidden_states,
            forward_batch,
            should_allreduce_fusion,
            use_reduce_scatter,
        )

    hidden_states_tensor = (
        hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
    )
    if (self.tp_size == 1) and hidden_states_tensor.shape[0] == 0:
        return hidden_states

    gate_up, _ = self.gate_up_proj(hidden_states)
    try:
        hidden_states, _ = self.down_proj(
            gate_up,
            skip_all_reduce=should_allreduce_fusion or use_reduce_scatter,
            use_fused_silu_mul_fp8_quant=True,
        )
        return hidden_states
    except TypeError:
        logger.warning(
            "[hcu_sglang] bailing fused silu mul fp8 quant is unavailable, fallback"
        )
        return original_fn(
            self,
            hidden_states,
            forward_batch,
            should_allreduce_fusion,
            use_reduce_scatter,
        )


def replacement_gate_init(original_fn, self, *args, **kwargs):
    original_fn(self, *args, **kwargs)
    config = kwargs.get("config", args[0] if args else None)
    if config is None:
        return
    if is_dcu() and self.params_dtype == torch.float32:
        self.weight = nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                dtype=torch.bfloat16,
            )
        )


def replacement_sparse_moe_init(original_fn, self, *args, **kwargs):
    original_fn(self, *args, **kwargs)
    config = kwargs.get("config", args[1] if len(args) > 1 else None)
    quant_config = kwargs.get("quant_config", args[2] if len(args) > 2 else None)
    prefix = kwargs.get("prefix", args[4] if len(args) > 4 else "")
    if config is None:
        return
    num_fused_shared_experts = _num_fused_shared_experts(config)
    self.num_fused_shared_experts = num_fused_shared_experts
    if num_fused_shared_experts == 0:
        return

    from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
    from sglang.srt.layers.moe.topk import TopK
    from sglang.srt.model_parallel.parallel_state import (
        get_moe_expert_parallel_world_size,
    )
    from sglang.srt.utils import add_prefix

    self.moe_ep_size = get_moe_expert_parallel_world_size()
    fused_shared_experts_scaling_factor = (
        1.0 / float(self.moe_ep_size)
        if self.moe_ep_size > 1 and num_fused_shared_experts > 0
        else None
    )

    self.topk = TopK(
        top_k=self.top_k + num_fused_shared_experts,
        layer_id=self.layer_id,
        renormalize=self.norm_topk_prob,
        use_grouped_topk=self.use_grouped_topk,
        num_expert_group=self.num_expert_group,
        num_fused_shared_experts=num_fused_shared_experts,
        topk_group=self.topk_group,
        correction_bias=self.correction_bias,
        quant_config=quant_config,
        routed_scaling_factor=self.routed_scaling_factor,
        fused_shared_experts_scaling_factor=fused_shared_experts_scaling_factor,
    )

    self.num_experts = (
        config.num_experts + get_exec().moe.ep_num_redundant_experts
    )
    self.experts = get_moe_impl_class(quant_config)(
        num_experts=config.num_experts
        + num_fused_shared_experts
        + get_exec().moe.ep_num_redundant_experts,
        num_fused_shared_experts=num_fused_shared_experts,
        top_k=self.top_k + num_fused_shared_experts,
        layer_id=self.layer_id,
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_intermediate_size,
        quant_config=quant_config,
        routed_scaling_factor=self.routed_scaling_factor,
        prefix=add_prefix("experts", prefix),
    )
    if hasattr(self, "shared_experts"):
        delattr(self, "shared_experts")
    self.num_shared_experts = 0


def replacement_load_weights(original_fn, self, weights, is_nextn=False):
    num_fused_shared_experts = _num_fused_shared_experts(self.config)
    self.num_fused_shared_experts = num_fused_shared_experts
    if num_fused_shared_experts == 0:
        return original_fn(self, weights, is_nextn=is_nextn)

    def mapped_weights():
        for name, loaded_weight in weights:
            if "mlp.shared_experts" in name:
                name = name.replace(
                    "mlp.shared_experts",
                    f"mlp.experts.{self.config.num_experts}",
                )
            yield name, loaded_weight

    original_num_experts = self.config.num_experts
    self.config.num_experts = original_num_experts + num_fused_shared_experts
    try:
        return original_fn(self, mapped_weights(), is_nextn=is_nextn)
    finally:
        self.config.num_experts = original_num_experts


def replacement_forward(self, positions, hidden_states, forward_batch):
    original = get_original_attr(_MODULE, "BailingMoEAttention.forward")
    if not use_fused_rms_rotary():
        return original(self, positions, hidden_states, forward_batch)

    try:
        from lightop import rms_rotary_embedding_fuse_with_kv_store
    except ImportError as exc:
        logger.warning(
            "[hcu_sglang] bailing_moe fused RMS rotary is unavailable, fallback: %s",
            exc,
        )
        return original(self, positions, hidden_states, forward_batch)

    from sglang.srt.models.utils import (
        create_fused_set_kv_buffer_arg,
        enable_fused_set_kv_buffer,
    )

    if isinstance(hidden_states, tuple):
        if hidden_states[0].shape[0] == 0:
            return hidden_states
    else:
        if hidden_states.shape[0] == 0:
            return hidden_states

    qkv, _ = self.query_key_value(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    can_fuse_set_kv = (
        self.head_dim == self.rotary_emb.rotary_dim
        and enable_fused_set_kv_buffer(forward_batch)
    )

    if self.use_qk_norm:
        q, k, v = fused_rms_rotary_kv_store(
            rms_rotary_embedding_fuse_with_kv_store,
            self,
            positions,
            q,
            k,
            v,
            forward_batch,
            self.query_layernorm,
            self.key_layernorm,
        )
    else:
        q, k = self.rotary_emb(
            positions,
            q,
            k,
            fused_set_kv_buffer_arg=(
                create_fused_set_kv_buffer_arg(
                    value=v,
                    layer=self.attn,
                    forward_batch=forward_batch,
                )
                if can_fuse_set_kv
                else None
            ),
        )

    context_layer = self.attn(
        q,
        k,
        v,
        forward_batch,
        save_kv_cache=not can_fuse_set_kv,
    )
    attn_output, _ = self.dense(context_layer)
    return attn_output
