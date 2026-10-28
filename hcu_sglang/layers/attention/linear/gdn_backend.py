"""REPLACE hook implementations for DCU GDN backend paths."""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

import torch

from sglang.srt.utils import is_cpu, is_npu
from hcu_sglang.utils.module_patch import get_original_attr


logger = logging.getLogger(__name__)

_GDN_BACKEND_MODULE = "sglang.srt.layers.attention.linear.gdn_backend"


def register() -> None:
    hooks = (
        (
            "sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend.forward_decode",
            replacement_forward_decode,
        ),
        (
            "sglang.srt.layers.attention.linear.gdn_backend.GDNAttnBackend.forward_extend",
            replacement_forward_extend,
        ),
    )
    for target, hook in hooks:
        logger.info("[hcu_sglang] register REPLACE hook: %s", target)
        HookRegistry.register(target, hook, HookType.REPLACE)


def _load_causal_conv1d_dcu():
    from causal_conv1d import causal_conv1d_fn_dcu
    from causal_conv1d.causal_conv1d_interface import (
        causal_conv1d_update as causal_conv1d_update_dcu,
    )

    return causal_conv1d_update_dcu, causal_conv1d_fn_dcu


def _load_sglang_gdn_symbols():
    try:
        from sglang.kernels.ops.attention.fla.fused_gdn_gating import (
            fused_gdn_gating,
        )
        from sglang.kernels.ops.mamba.causal_conv1d_triton import (
            causal_conv1d_update,
        )
        from sglang.srt.mem_cache.memory_pool import MambaPool
    except ImportError:
        from sglang.srt.layers.attention.linear.gdn_backend import (
            MambaPool,
            fused_gdn_gating,
        )
        from sglang.kernels.ops.mamba.causal_conv1d_triton import (
            causal_conv1d_update,
        )

    return MambaPool, fused_gdn_gating, causal_conv1d_update


def replacement_forward_decode(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
    """Replacement for GDNAttnBackend.forward_decode."""
    try:
        causal_conv1d_update_dcu, _ = _load_causal_conv1d_dcu()
    except Exception as exc:
        logger.warning(
            "[hcu_sglang] DCU causal_conv1d decode is unavailable, fallback: %s",
            exc,
        )
        original = get_original_attr(_GDN_BACKEND_MODULE, "GDNAttnBackend.forward_decode")
        return original(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)

    layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    conv_states = layer_cache.conv[0]
    ssm_states = layer_cache.temporal
    query_start_loc = self.forward_metadata.query_start_loc
    cache_indices = self.forward_metadata.mamba_cache_indices

    assert isinstance(mixed_qkv, torch.Tensor)
    mixed_qkv = causal_conv1d_update_dcu(
        mixed_qkv,
        conv_states,
        layer.conv_weights,
        layer.bias,
        layer.activation,
        conv_state_indices=cache_indices,
    )

    if self.kernel_dispatcher.supports_packed_decode:
        core_attn_out = self.kernel_dispatcher.packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            scale=layer.head_k_dim**-0.5,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=layer.num_v_heads,
            head_v_dim=layer.head_v_dim,
        )
        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )
        return core_attn_out

    query, key, value = torch.split(
        mixed_qkv,
        [layer.q_dim, layer.k_dim, layer.v_dim],
        dim=-1,
    )
    bs = forward_batch.batch_size
    query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
    key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
    value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

    core_attn_out = self.kernel_dispatcher.decode(
        q=query,
        k=key,
        v=value,
        a=a,
        b=b,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        ssm_states=ssm_states,
        cache_indices=cache_indices,
        query_start_loc=query_start_loc,
    )

    self._track_mamba_state_decode(
        forward_batch, conv_states, ssm_states, cache_indices
    )
    return core_attn_out


def replacement_forward_extend(self, layer, forward_batch, mixed_qkv, a, b, **kwargs):
    """Replacement for GDNAttnBackend.forward_extend."""
    MambaPool, fused_gdn_gating, causal_conv1d_update = _load_sglang_gdn_symbols()
    try:
        _, causal_conv1d_fn_dcu = _load_causal_conv1d_dcu()
    except Exception as exc:
        logger.warning(
            "[hcu_sglang] DCU causal_conv1d extend is unavailable, fallback: %s",
            exc,
        )
        original = get_original_attr(_GDN_BACKEND_MODULE, "GDNAttnBackend.forward_extend")
        return original(self, layer, forward_batch, mixed_qkv, a, b, **kwargs)

    assert isinstance(mixed_qkv, torch.Tensor)
    seq_len = mixed_qkv.shape[0]

    is_target_verify = forward_batch.forward_mode.is_target_verify()
    forward_metadata = self.forward_metadata

    query_start_loc = forward_metadata.query_start_loc
    cache_indices = forward_metadata.mamba_cache_indices
    retrieve_next_token = forward_metadata.retrieve_next_token
    retrieve_next_sibling = forward_metadata.retrieve_next_sibling
    retrieve_parent_token = forward_metadata.retrieve_parent_token

    mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    conv_states = mamba_cache_params.conv[0]
    ssm_states = mamba_cache_params.temporal
    if is_target_verify:
        assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
        intermediate_state_cache = mamba_cache_params.intermediate_ssm
        intermediate_conv_window_cache = mamba_cache_params.intermediate_conv_window[0]
        intermediate_state_indices = self.verify_intermediate_state_indices
    else:
        has_initial_states = forward_batch.extend_prefix_lens > 0

    if is_target_verify:
        batch_size = seq_len // forward_batch.spec_info.draft_token_num
        draft_token_num = forward_batch.spec_info.draft_token_num
        mixed_qkv_reshaped = mixed_qkv.view(
            batch_size, draft_token_num, -1
        ).transpose(1, 2)
        mixed_qkv_processed = causal_conv1d_update(
            mixed_qkv_reshaped,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices[:batch_size],
            intermediate_conv_window=intermediate_conv_window_cache,
            intermediate_state_indices=intermediate_state_indices[:batch_size],
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_parent_token=retrieve_parent_token,
        )
        mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
    else:
        mixed_qkv = mixed_qkv.transpose(0, 1)
        if forward_metadata.has_mamba_track_mask:
            mixed_qkv_to_track = mixed_qkv[
                :, forward_metadata.track_conv_indices
            ].transpose(0, 1)
            conv_states[forward_metadata.conv_states_mask_indices] = mixed_qkv_to_track

        mixed_qkv = causal_conv1d_fn_dcu(
            mixed_qkv,
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            initial_states=conv_states,
            has_initial_state=has_initial_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)[:seq_len]

    query, key, value = torch.split(
        mixed_qkv,
        [layer.q_dim, layer.k_dim, layer.v_dim],
        dim=-1,
    )

    actual_seq_len = query.shape[0]
    query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
    key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
    value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

    if is_target_verify:
        core_attn_out = self.kernel_dispatcher.target_verify(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            intermediate_states_buffer=intermediate_state_cache,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=forward_batch.spec_info.draft_token_num,
            retrieve_parent_token=retrieve_parent_token,
        )
    else:
        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
        core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        if (is_npu() or is_cpu()) and last_recurrent_state is not None:
            last_recurrent_state = last_recurrent_state.to(
                ssm_states.dtype, copy=False
            )
            ssm_states[cache_indices] = last_recurrent_state

        if h is not None:
            self._track_mamba_state_extend(
                forward_batch, h, ssm_states, forward_metadata
            )

    return core_attn_out
