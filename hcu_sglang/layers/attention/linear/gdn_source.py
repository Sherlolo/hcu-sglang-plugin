from __future__ import annotations
import sglang.srt.layers.attention.linear.gdn_backend as upstream
from hcu_sglang.utils.detection import is_hcu
from sglang.srt.utils import get_bool_env_var

globals().update({name: value for name, value in vars(upstream).items()
                  if not name.startswith("__")})
_is_hcu = is_hcu()
_use_hcu_causal_conv1d = False
if _is_hcu and get_bool_env_var("SGLANG_USE_CAUSAL_CONV1D"):
    try:
        from causal_conv1d import causal_conv1d_fn_hcu
        from causal_conv1d.causal_conv1d_interface import (
            causal_conv1d_update as causal_conv1d_update_hcu,
        )

        _use_hcu_causal_conv1d = True
        rank0_log("Using HCU causal_conv1d for GDN decode/prefill.")
    except ImportError as exc:
        rank0_log(
            "SGLANG_USE_CAUSAL_CONV1D=1 but HCU causal_conv1d import failed "
            f"({exc}); falling back to Triton."
        )



def forward_decode(
    self,
    layer: RadixLinearAttention,
    forward_batch: ForwardBatch,
    mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    a: torch.Tensor,
    b: torch.Tensor,
    **kwargs,
):
    global _fused_decode_proj_conv_fallback_logged
    global _fused_decode_proj_conv_logged
    global _fused_decode_proj_conv_layers_logged
    global _fused_decode_real_tensor_verified_layers

    if _is_hip and isinstance(mixed_qkv, torch.Tensor) and mixed_qkv.shape[0] == 0:
        return mixed_qkv.new_zeros((1, 0, layer.num_v_heads, layer.head_v_dim))

    layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
    conv_states = layer_cache.conv[0]
    ssm_states = layer_cache.temporal
    query_start_loc = self.forward_metadata.query_start_loc
    cache_indices = self.forward_metadata.mamba_cache_indices
    # GDN ReplaySSM (slice 1a): per-layer ring slices + the once-per-forward
    # per-row write cursor. All None unless --enable-linear-replayssm, so the
    # legacy dispatch below is byte-identical when the flag is off.
    replayssm_write_pos = self.forward_metadata.replayssm_write_pos
    # GDN ReplaySSM (slice 2b): per-row force-flush at radix track
    # boundaries (None unless --enable-linear-replayssm). When present the
    # kernel folds the ring into temporal[slot] on the snapshot steps.
    replayssm_force_flush = self.forward_metadata.replayssm_force_flush
    replayssm_d = layer_cache.replayssm_d
    replayssm_k = layer_cache.replayssm_k
    replayssm_g = layer_cache.replayssm_g

    return_z = False
    conv_already_applied = False
    if isinstance(mixed_qkv, tuple):
        if len(mixed_qkv) != 2:
            raise ValueError(
                "Fused GDN decode projection input must be "
                "(projected_qkvz, projected_ba)"
            )
        projected_qkvz, projected_ba = mixed_qkv
        eligible, eligibility_reason = (
            can_use_fused_qkvzba_causal_conv1d_update_contiguous(
                projected_qkvz,
                projected_ba,
                conv_states,
                layer.conv_weights,
                layer.bias,
                activation=layer.activation,
            )
        )
        if eligible and not _use_hcu_causal_conv1d:
            qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
            fused_backend = "triton_direct_oracle_exact"
            if not _fused_decode_proj_conv_logged:
                rank0_log("Using fused GDN decode QKVZ/BA unpack + indexed Conv1D.")
                _fused_decode_proj_conv_logged = True
            if (
                _fused_decode_log_layer_hits or _fused_decode_verify_real_tensors
            ) and layer.layer_id not in _fused_decode_proj_conv_layers_logged:
                rank0_log(
                    "GDN_FUSED_DECODE_BACKEND "
                    f"layer_id={layer.layer_id} backend={fused_backend} "
                    f"batch={projected_qkvz.shape[0]} "
                    f"qkv_dim={qkv_dim} state_shape={tuple(conv_states.shape)} "
                    f"state_indices_dtype={cache_indices.dtype}"
                )
                _fused_decode_proj_conv_layers_logged.add(layer.layer_id)

            # Compare real activations against the direct-Triton update on a
            # compact state copy, leaving the live cache to the candidate.
            verify_real_tensors = (
                _fused_decode_verify_real_tensors
                and layer.layer_id not in _fused_decode_real_tensor_verified_layers
            )
            if verify_real_tensors:
                if bool(torch.any(cache_indices < 0).item()):
                    raise AssertionError(
                        "Real-tensor GDN fusion verification requires "
                        "non-padding cache indices"
                    )
                ref_indices = torch.arange(
                    cache_indices.numel(),
                    device=cache_indices.device,
                    dtype=torch.int32,
                )
                ref_state = torch.index_select(
                    conv_states, 0, cache_indices.to(torch.int64)
                )
                ref_mixed_qkv, ref_z, ref_b, ref_a = (
                    fused_qkvzba_split_reshape_cat_contiguous(
                        projected_qkvz,
                        projected_ba,
                        layer.num_q_heads,
                        layer.num_v_heads,
                        layer.head_q_dim,
                        layer.head_v_dim,
                    )
                )
                ref_mixed_qkv = causal_conv1d_update(
                    ref_mixed_qkv,
                    ref_state,
                    layer.conv_weights,
                    layer.bias,
                    layer.activation,
                    conv_state_indices=ref_indices,
                )

            mixed_qkv, z, b, a = fused_qkvzba_causal_conv1d_update_contiguous(
                projected_qkvz,
                projected_ba,
                conv_states,
                layer.conv_weights,
                layer.bias,
                cache_indices,
                qkv_dim=qkv_dim,
                v_dim=layer.v_dim,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
                activation=layer.activation,
            )
            if verify_real_tensors:
                candidate_state = torch.index_select(
                    conv_states, 0, cache_indices.to(torch.int64)
                )
                named_pairs = (
                    ("qkv", mixed_qkv, ref_mixed_qkv),
                    ("z", z, ref_z),
                    ("b", b, ref_b),
                    ("a", a, ref_a),
                    ("state", candidate_state, ref_state),
                )
                report = []
                mismatch = False
                for tensor_name, candidate, reference in named_pairs:
                    diff = (candidate.float() - reference.float()).abs()
                    nonzero = int(torch.count_nonzero(diff).item())
                    mismatch |= nonzero != 0
                    report.append(
                        f"{tensor_name}_nonzero={nonzero}/"
                        f"{diff.numel()} {tensor_name}_max="
                        f"{diff.max().item()}"
                    )
                rank0_log(
                    "GDN_FUSED_REAL_TENSOR_PARITY "
                    f"layer_id={layer.layer_id} backend={fused_backend} "
                    + " ".join(report)
                )
                _fused_decode_real_tensor_verified_layers.add(layer.layer_id)
                if mismatch:
                    raise AssertionError(
                        "GDN fused real-tensor parity failed at "
                        f"layer_id={layer.layer_id}; " + " ".join(report)
                    )
            conv_already_applied = True
        else:
            if not _fused_decode_proj_conv_fallback_logged:
                rank0_log(
                    "Falling back from fused GDN decode projection/Conv1D: "
                    f"{eligibility_reason}"
                )
                _fused_decode_proj_conv_fallback_logged = True
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
                projected_qkvz,
                projected_ba,
                layer.num_q_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_v_dim,
            )
        return_z = True
    else:
        assert isinstance(mixed_qkv, torch.Tensor)

    if not conv_already_applied:
        conv_update = (
            causal_conv1d_update_hcu
            if _use_hcu_causal_conv1d
            else causal_conv1d_update
        )
        mixed_qkv = conv_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )

    # Skip split + reshape + separate gating kernel by consuming
    # the packed mixed_qkv directly in a single fused Triton kernel.
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
            replayssm_d=replayssm_d,
            replayssm_k=replayssm_k,
            replayssm_g=replayssm_g,
            replayssm_write_pos=replayssm_write_pos,
            replayssm_force_flush=replayssm_force_flush,
        )
        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
        )
        return (core_attn_out, z) if return_z else core_attn_out

    query, key, value = torch.split(
        mixed_qkv,
        [layer.q_dim, layer.k_dim, layer.v_dim],
        dim=-1,
    )
    # Reshape from [bs, h*d] to [1, bs, h, d]
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
        forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
    )

    return (core_attn_out, z) if return_z else core_attn_out

def _convolve_prefill(
    self,
    layer: RadixLinearAttention,
    forward_batch: ForwardBatch,
    mixed_qkv: torch.Tensor,
    conv_states: torch.Tensor,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    block_table = self.forward_metadata.conv_block_table
    use_hcu_conv = _use_hcu_causal_conv1d and block_table is None
    conv_fn = causal_conv1d_fn_hcu if use_hcu_conv else causal_conv1d_fn
    conv_kwargs = {"initial_states" if use_hcu_conv else "conv_states": conv_states}
    if block_table is not None:
        conv_kwargs["block_table"] = block_table
    return conv_fn(
        mixed_qkv.transpose(0, 1),
        layer.conv_weights,
        layer.bias,
        activation=layer.activation,
        has_initial_state=forward_batch.extend_prefix_lens > 0,
        cache_indices=cache_indices,
        query_start_loc=self.forward_metadata.query_start_loc,
        seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        **conv_kwargs,
    ).transpose(0, 1)[: mixed_qkv.shape[0]]
