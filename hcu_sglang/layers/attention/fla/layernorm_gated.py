"""REPLACE hook implementation for DCU optimized gated layer norm."""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

import torch
import triton

from sglang.srt.utils import cdiv, device_context
from hcu_sglang.utils.module_patch import get_original_attr


logger = logging.getLogger(__name__)

try:
    import sglang.kernels.ops.attention.fla.layernorm_gated  # noqa: F401

    _LAYERNORM_MODULE = "sglang.kernels.ops.attention.fla.layernorm_gated"
except ImportError:
    _LAYERNORM_MODULE = "sglang.kernels.ops.attention.fla.layernorm_gated"


def register() -> None:
    target = f"{_LAYERNORM_MODULE}._layer_norm_fwd"
    logger.info("[hcu_sglang] register REPLACE hook: %s", target)
    HookRegistry.register(target, replacement_layer_norm_fwd, HookType.REPLACE)


def replacement_layer_norm_fwd(
    x,
    weight,
    bias,
    eps,
    z=None,
    out=None,
    group_size=None,
    norm_before_gate=True,
    is_rms_norm=False,
    activation: str = "swish",
):
    """Replacement for layernorm_gated._layer_norm_fwd."""
    try:
        from lightop import op
        if _LAYERNORM_MODULE.startswith("sglang.kernels"):
            from sglang.kernels.ops.attention.fla.layernorm_gated import (
                calc_rows_per_block,
            )
        else:
            from sglang.kernels.ops.attention.fla.layernorm_gated import (
                calc_rows_per_block,
            )
    except Exception as exc:
        logger.warning(
            "[hcu_sglang] optimized layer_norm_fwd is unavailable, fallback: %s",
            exc,
        )
        original = get_original_attr(_LAYERNORM_MODULE, "_layer_norm_fwd")
        return original(
            x,
            weight,
            bias,
            eps,
            z=z,
            out=out,
            group_size=group_size,
            norm_before_gate=norm_before_gate,
            is_rms_norm=is_rms_norm,
            activation=activation,
        )

    M, N = x.shape
    if group_size is None:
        group_size = N
    assert N % group_size == 0
    ngroups = N // group_size
    assert x.stride(-1) == 1
    if z is not None:
        assert z.stride(-1) == 1
        assert z.shape == (M, N)
    assert weight.shape == (N,)
    assert weight.stride(-1) == 1
    if bias is not None:
        assert bias.stride(-1) == 1
        assert bias.shape == (N,)

    if out is not None:
        assert out.shape == x.shape
    else:
        out = torch.empty_like(x)
    assert out.stride(-1) == 1

    mean = (
        torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
        if not is_rms_norm
        else None
    )
    rstd = torch.empty((ngroups * M,), dtype=torch.float32, device=x.device)
    max_fused_size = 65536 // x.element_size()
    block_n = min(max_fused_size, triton.next_power_of_2(group_size))
    if group_size > block_n:
        raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

    rows_per_block = calc_rows_per_block(M, x.device)
    with device_context(x.device):
        op.layer_norm_fwd_1pass_opt(
            x,
            out,
            weight,
            bias,
            z,
            mean,
            rstd,
            x.stride(0),
            out.stride(0),
            z.stride(0) if z is not None else 0,
            M,
            group_size,
            eps,
            BLOCK_N=block_n,
            ROWS_PER_BLOCK=rows_per_block,
            HAS_BIAS=bias is not None,
            HAS_Z=z is not None,
            NORM_BEFORE_GATE=norm_before_gate,
            IS_RMS_NORM=is_rms_norm,
            ACTIVATION=activation,
        )
    return out, mean, rstd
