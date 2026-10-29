"""Optional HCU Triton attention extensions."""

from __future__ import annotations

import os

from hcu_sglang.compat.aiter import install_aiter_compat

install_aiter_compat()

import sglang.srt.layers.attention.triton_backend as _upstream

globals().update({name: value for name, value in vars(_upstream).items() if not name.startswith("__")})



def after_init(result, backend, *args, **kwargs):
    from hcu_sglang.utils.detection import is_dcu

    if not is_dcu():
        return result
    backend.needs_cpu_seq_lens = True
    backend.use_aiter_triton_extend_fwd = False
    if os.getenv("SGLANG_USE_TRITON_EXTEND_FROM_AITER", "0") == "1":
        try:
            from aiter.ops.triton.extend_attention import extend_attention_fwd

            backend.extend_attention_fwd = torch.compiler.disable(extend_attention_fwd)
            backend.use_aiter_triton_extend_fwd = True
        except ImportError:
            pass
    return result


def around_forward_extend(original, backend, *args, **kwargs):
    if not getattr(backend, "use_aiter_triton_extend_fwd", False):
        return original(backend, *args, **kwargs)
    return forward_extend(backend, *args, **kwargs)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    target = "sglang.srt.layers.attention.triton_backend.TritonAttnBackend"
    HookRegistry.register(f"{target}.__init__", after_init, HookType.AFTER)
    HookRegistry.register(
        f"{target}.forward_extend", around_forward_extend, HookType.AROUND
    )
# Fork implementation is used only when the optional AITER Triton extend path is enabled.
def forward_extend(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    save_kv_cache=True,
    sinks=None,
    score_mod=None,
    aux_tensors=None,
):
    # TODO: reuse the buffer across layers
    attn_out = getattr(forward_batch, "_attn_output", None)
    if attn_out is not None:
        o = attn_out
    elif layer.qk_head_dim != layer.v_head_dim:
        o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
    else:
        o = torch.empty_like(q)

    if k is None and v is None:
        pool = self.token_to_kv_pool
        cache_loc = forward_batch.out_cache_loc
        if isinstance(pool, SWAKVPool) and pool.layers_mapping[layer.layer_id][1]:
            cache_loc = pool.translate_loc_from_full_to_swa(cache_loc)
        elif self._translate_kv_loc is not None:
            # Unified pool: buffers are indexed in the kernel-facing id space.
            cache_loc = self._translate_kv_loc(cache_loc)
        k_buffer, v_buffer = pool.get_kv_buffer(layer.layer_id)
        k = k_buffer[cache_loc]
        v = v_buffer[cache_loc]
    elif k is None or v is None:
        raise ValueError("Both k and v should be None or not None")
    else:
        # Save KV cache first (must do this before unified kernel)
        if save_kv_cache:
            loc_info = KVWriteLoc(
                forward_batch.out_cache_loc,
                self.forward_metadata.swa_out_cache_loc,
                full_loc=self.forward_metadata.out_cache_loc_full_physical,
            )
            if layer.k_scale is None:
                self._set_kv_buffer(forward_batch, layer, loc_info, k, v)
            elif self.use_mla:
                # For MLA, scale K manually before storing since MLATokenToKVPool
                # doesn't accept scale parameters. Clone to protect k from mutation
                # since it's used later in the attention kernel.
                k_scaled = k.clone().div_(layer.k_scale)
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    loc_info,
                    k_scaled,
                    v,
                )
            else:
                self._set_kv_buffer(
                    forward_batch,
                    layer,
                    loc_info,
                    k.clone(),  # cloned to protect k,v from in-place mutation in set_kv_buffer
                    v.clone(),
                    layer.k_scale,
                    layer.v_scale,
                )

    logits_soft_cap = logit_capping_mod(layer.logit_capping_method, layer.logit_cap)

    causal = True
    if (
        layer.is_cross_attention
        or layer.attn_type == AttentionType.ENCODER_ONLY
        or (
            layer.attn_type == AttentionType.DECODER_BIDIRECTIONAL
            and self.allow_bidirectional_attention_in_extend
        )
    ):
        causal = False

    if self.dcp_size > 1:
        if score_mod is not None:
            raise NotImplementedError(
                "DCP Triton extend does not support score_mod"
            )
        return self._forward_extend_dcp(
            q, k, v, layer, forward_batch, causal, logits_soft_cap, sinks
        )

    # Deterministic mode: use unified 1-stage kernel
    if self.enable_deterministic:
        return self._forward_extend_unified(
            q,
            o,
            layer,
            forward_batch,
            causal,
            logits_soft_cap,
            sinks,
            score_mod=score_mod,
            aux_tensors=aux_tensors,
        )

    # Normal mode: use original 2-stage kernel
    if layer.sliding_window_size is not None and layer.sliding_window_size > -1:
        sliding_window_size = (
            layer.sliding_window_size
        )  # Needed for sliding window mask
        kv_indptr = self.forward_metadata.window_kv_indptr
        kv_indices = self.forward_metadata.window_kv_indices
        window_kv_offsets = self.forward_metadata.window_kv_offsets
    else:
        sliding_window_size = -1
        kv_indptr = self.forward_metadata.kv_indptr
        kv_indices = self.forward_metadata.kv_indices
        window_kv_offsets = None

    if layer.k_scale is not None and layer.v_scale is not None:
        k_descale = layer.k_scale_float
        v_descale = layer.v_scale_float
    else:
        k_descale = 1.0
        v_descale = 1.0

    if self.use_aiter_triton_extend_fwd:
        self.extend_attention_fwd(
            q_extend=q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k_extend=k.contiguous(),
            v_extend=v.contiguous(),
            o_extend=o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            k_buffer=self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            v_buffer=self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            qo_indptr=self.forward_metadata.qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            custom_mask=self.forward_metadata.custom_mask,
            is_causal=causal,
            mask_indptr=self.forward_metadata.mask_indptr,
            max_len_extend=self.forward_metadata.max_extend_len,
            sm_scale=layer.scaling,
            logit_cap=logits_soft_cap,
            k_scale=k_descale,
            v_scale=v_descale,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=layer.xai_temperature_len,
        )
        return o
    # Split-KV EAGLE-verify fast path (ROCm/Triton). On target-verify
    # (topk=1 causal chain), run the bandwidth-efficient split-KV kernel
    # instead of the serial-prefix extend kernel. verify_splitkv_fwd()
    # returns True if it ran (o written), or False for any case it cannot
    # serve bit-equivalently (its can_handle() gates on non-causal / sinks /
    # sliding-window / ragged / topk>1), so we fall through to
    # extend_attention_fwd below. Correctness is never at risk.
    # Route target-verify to the grouped-head kernel when eligible, else the
    # per-head split-KV kernel.
    if self.use_verify_shared_kv:
        verify_fwd = self.verify_shared_kv_fwd
    elif self.use_verify_splitkv:
        verify_fwd = self.verify_splitkv_fwd
    else:
        verify_fwd = None
    if (
        verify_fwd is not None
        and score_mod is None
        and forward_batch.forward_mode.is_target_verify()
        and verify_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.forward_metadata.qo_indptr,
            kv_indptr,
            kv_indices,
            self.forward_metadata.custom_mask,
            causal,
            self.forward_metadata.mask_indptr,
            self.forward_metadata.max_extend_len,
            k_descale,
            v_descale,
            layer.scaling,
            logit_cap=logits_soft_cap,
            sliding_window_size=sliding_window_size,
            sinks=sinks,
            window_kv_offsets=window_kv_offsets,
            xai_temperature_len=layer.xai_temperature_len,
            max_bs=self.req_to_token_pool.size,
        )
    ):
        return o

    self.extend_attention_fwd(
        q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
        k.contiguous(),
        v.contiguous(),
        o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
        self.token_to_kv_pool.get_key_buffer(layer.layer_id),
        self.token_to_kv_pool.get_value_buffer(layer.layer_id),
        self.forward_metadata.qo_indptr,
        kv_indptr,
        kv_indices,
        self.forward_metadata.custom_mask,
        causal,
        self.forward_metadata.mask_indptr,
        self.forward_metadata.max_extend_len,
        k_descale,
        v_descale,
        layer.scaling,
        logit_cap=logits_soft_cap,
        sliding_window_size=sliding_window_size,
        sinks=sinks,
        window_kv_offsets=window_kv_offsets,
        xai_temperature_len=layer.xai_temperature_len,
        page_size=self.page_size,
        score_mod=score_mod,
        aux_tensors=aux_tensors,
        extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
    )
    return o
