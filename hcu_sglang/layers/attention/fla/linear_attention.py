"""REPLACE hook implementations for AITER linear-attention paths."""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from typing import Optional, Tuple

import torch
import triton

try:
    from sglang.kernels.ops.attention.fla.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
except ImportError:
    from sglang.kernels.ops.attention.fla.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )

CHUNK_SIZE = 64
from hcu_sglang.utils.module_patch import get_original_attr


logger = logging.getLogger(__name__)

try:
    import sglang.kernels.ops.attention.fla.chunk_delta_h  # noqa: F401

    _FLA_MODULE_ROOT = "sglang.kernels.ops.attention.fla"
except ImportError:
    _FLA_MODULE_ROOT = "sglang.srt.layers.attention.fla"

_CHUNK_DELTA_H_MODULE = f"{_FLA_MODULE_ROOT}.chunk_delta_h"
_CHUNK_O_MODULE = f"{_FLA_MODULE_ROOT}.chunk_o"
_FUSED_RECURRENT_MODULE = f"{_FLA_MODULE_ROOT}.fused_recurrent"
_GDN_TRITON_MODULE = "sglang.srt.layers.attention.linear.kernels.gdn_triton"


def register() -> None:
    hooks = (
        (
            f"{_CHUNK_DELTA_H_MODULE}.chunk_gated_delta_rule_fwd_h",
            replacement_chunk_gated_delta_rule_fwd_h,
        ),
        (
            f"{_CHUNK_O_MODULE}.chunk_fwd_o",
            replacement_chunk_fwd_o,
        ),
        (
            f"{_FUSED_RECURRENT_MODULE}.fused_recurrent_gated_delta_rule_packed_decode",
            replacement_fused_recurrent_gated_delta_rule_packed_decode,
        ),
        (
            "sglang.srt.layers.attention.linear.kernels.gdn_triton.TritonGDNKernel.target_verify",
            replacement_triton_gdn_target_verify,
        ),
    )
    for target, hook in hooks:
        logger.info("[hcu_sglang] register REPLACE hook: %s", target)
        HookRegistry.register(target, hook, HookType.REPLACE)


def replacement_chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    save_new_value: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replacement for chunk_delta_h.chunk_gated_delta_rule_fwd_h."""
    try:
        from aiter.ops.triton.fla.sglang.chunk_delta_h import (
            launch_chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
        )
    except Exception as exc:
        logger.warning(
            "[hcu_sglang] AITER chunk_delta_h is unavailable, fallback: %s",
            exc,
        )
        original = get_original_attr(
            _CHUNK_DELTA_H_MODULE,
            "chunk_gated_delta_rule_fwd_h",
        )
        return original(
            k,
            w,
            u,
            g=g,
            gk=gk,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            save_new_value=save_new_value,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = CHUNK_SIZE

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, V, K)
    v_new = torch.empty_like(u) if save_new_value else None

    launch_chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
        k=k,
        u=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        N=N,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        kernel_cfg=None,
    )
    return h, v_new


def replacement_chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Replacement for chunk_o.chunk_fwd_o."""
    try:
        from aiter.ops.triton.fla.sglang.chunk_o import launch_chunk_fwd_kernel_o
    except Exception as exc:
        logger.warning(
            "[hcu_sglang] AITER chunk_o is unavailable, fallback: %s",
            exc,
        )
        original = get_original_attr(_CHUNK_O_MODULE, "chunk_fwd_o")
        return original(
            q,
            k,
            v,
            h,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
        )

    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = min(chunk_size, max(16, triton.next_power_of_2(T)))
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    )
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.zeros_like(v)
    launch_chunk_fwd_kernel_o(
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        NT=NT,
        B=B,
        kernel_cfg=None,
    )
    return o


def replacement_fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Replacement for fused_recurrent_gated_delta_rule_packed_decode."""
    try:
        from aiter.ops.triton.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )
    except Exception as exc:
        logger.warning(
            "[hcu_sglang] AITER fused_recurrent packed decode is unavailable, fallback: %s",
            exc,
        )
        original = get_original_attr(
            _FUSED_RECURRENT_MODULE,
            "fused_recurrent_gated_delta_rule_packed_decode",
        )
        return original(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            initial_state=initial_state,
            out=out,
            ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        out=out,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    return out, initial_state


def replacement_triton_gdn_target_verify(
    self,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    intermediate_states_buffer: torch.Tensor,
    intermediate_state_indices: torch.Tensor,
    cache_steps: int,
    retrieve_parent_token: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Replacement for TritonGDNKernel.target_verify."""
    try:
        from aiter.ops.triton.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )
    except Exception as exc:
        logger.warning(
            "[hcu_sglang] AITER GDN target_verify is unavailable, fallback: %s",
            exc,
        )
        original = get_original_attr(_GDN_TRITON_MODULE, "TritonGDNKernel.target_verify")
        return original(
            self,
            A_log,
            dt_bias,
            q,
            k,
            v,
            a,
            b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            intermediate_states_buffer=intermediate_states_buffer,
            intermediate_state_indices=intermediate_state_indices,
            cache_steps=cache_steps,
            retrieve_parent_token=retrieve_parent_token,
            **kwargs,
        )

    return fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state_source=ssm_states,
        initial_state_indices=cache_indices,
        cu_seqlens=query_start_loc,
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=False,
        disable_state_update=True,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_steps,
        retrieve_parent_token=retrieve_parent_token,
    )
