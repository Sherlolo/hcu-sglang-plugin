"""HCU compatibility for DSA quantization and index-cache kernels."""

from __future__ import annotations

from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_hcu


def around_tilelang_act_quant(
    original: Callable[..., Any],
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
):
    if not is_hcu():
        return original(x, block_size, scale_fmt)

    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert x.size(-1) % block_size == 0, (
        f"Last dimension size must be divisible by block_size "
        f"(block_size={block_size})"
    )

    from lightop import op
    from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz

    output_dtype = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn
    output = torch.empty_like(x, dtype=output_dtype)
    scale = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    op.per_token_group_quant_fp8(
        output,
        x,
        scale,
        block_size,
        1e-5,
        scale_fmt is not None,
    )
    return output, scale


def before_set_k_and_s_triton(*args, **kwargs):
    if not is_hcu():
        return None
    updated_args = list(args)
    updated_kwargs = dict(kwargs)
    for index, name in ((2, "index_k"), (3, "index_k_scale")):
        if name in updated_kwargs:
            value = updated_kwargs[name]
            if value.dim() > 2:
                updated_kwargs[name] = value.reshape(-1, value.size(-1))
        elif len(updated_args) > index:
            value = updated_args[index]
            if value.dim() > 2:
                updated_args[index] = value.reshape(-1, value.size(-1))
    return tuple(updated_args), updated_kwargs


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.kernels.ops.attention.dsa.tilelang_kernel.act_quant",
        around_tilelang_act_quant,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.kernels.ops.attention.dsa.index_buf_accessor._set_k_and_s_triton",
        before_set_k_and_s_triton,
        HookType.BEFORE,
    )
