"""HCU compatibility hooks for shared SGLang kernel helpers."""

from __future__ import annotations

import importlib
from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_dcu


def after_get_default_target_flags(flags: list[str], arch: Any = None) -> list[str]:
    if not is_dcu():
        return flags
    return [
        "-DHIP_FP8_TYPE_FNUZ=1" if flag == "-DHIP_FP8_TYPE_E4M3=1" else flag
        for flag in flags
    ]


def around_assign_extend_cache_locs_func(
    original: Callable[..., torch.Tensor],
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    device: Any,
) -> torch.Tensor:
    if not is_dcu():
        return original(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            batch_size,
            draft_token_num,
            device,
        )

    from sglang.srt.utils import get_bool_env_var

    if not get_bool_env_var("SGLANG_ASSIGN_EXTEND_CACHE_LOCS", default="true"):
        return original(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            batch_size,
            draft_token_num,
            device,
        )

    from sgl_kernel.kvcacheio import hcu_assign_extend_cache_locs

    out_cache_loc = torch.empty(
        (batch_size * draft_token_num,), dtype=torch.int64, device=device
    )
    hcu_assign_extend_cache_locs(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        req_to_token.shape[1],
        batch_size,
    )
    return out_cache_loc


def around_scaled_fp8_quant(
    original: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    num_token_padding: int | None = None,
    use_per_token_if_dynamic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (is_dcu() and scale is None and use_per_token_if_dynamic):
        return original(input, scale, num_token_padding, use_per_token_if_dynamic)

    from lightop.quant import per_token_quant_fp8
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

    if not input.is_contiguous():
        input = input.contiguous()
    output = torch.empty_like(input, dtype=fp8_dtype)
    output_scale = torch.empty(
        (input.shape[0], 1), device=input.device, dtype=torch.float32
    )
    return per_token_quant_fp8(
        input,
        dtype=output.dtype,
        out_q=output,
        out_scale=output_scale,
    )


def _dispatch_mhc(name: str, original: Callable[..., Any], *args: Any, **kwargs: Any):
    if not is_dcu():
        return original(*args, **kwargs)
    from hcu_sglang.kernels import mhc

    return getattr(mhc, name)(*args, **kwargs)


def around_hc_split_sinkhorn(original: Callable[..., Any], *args: Any, **kwargs: Any):
    return _dispatch_mhc("hc_split_sinkhorn", original, *args, **kwargs)


def around_prewarm_mhc_pre(original: Callable[..., Any], *args: Any, **kwargs: Any):
    return _dispatch_mhc("prewarm_mhc_pre", original, *args, **kwargs)


def around_mhc_pre(original: Callable[..., Any], *args: Any, **kwargs: Any):
    return _dispatch_mhc("mhc_pre", original, *args, **kwargs)


def around_mhc_post(original: Callable[..., Any], *args: Any, **kwargs: Any):
    return _dispatch_mhc("mhc_post", original, *args, **kwargs)


def around_mhc_fused_post_pre(
    original: Callable[..., Any], *args: Any, **kwargs: Any
):
    return _dispatch_mhc("mhc_fused_post_pre", original, *args, **kwargs)


def around_vision_flash3_init(
    original: Callable[..., Any], instance: Any, *args: Any, **kwargs: Any
):
    if not is_dcu():
        return original(instance, *args, **kwargs)
    vision = importlib.import_module("sglang.srt.layers.attention.vision")
    from hcu_sglang.layers.attention.flashattention_interface import (
        flash_attn_varlen_func,
    )

    vision.flash_attn_func = flash_attn_varlen_func
    previous_is_cuda = vision._is_cuda
    vision._is_cuda = True
    try:
        return original(instance, *args, **kwargs)
    finally:
        vision._is_cuda = previous_is_cuda


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.kernels.jit.utils.arch.get_default_target_flags",
        after_get_default_target_flags,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.kernels.ops.speculative.cache_locs."
        "assign_extend_cache_locs_func",
        around_assign_extend_cache_locs_func,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.kernels.ops.quantization.fp8_kernel.scaled_fp8_quant",
        around_scaled_fp8_quant,
        HookType.AROUND,
    )
    mhc_target = "sglang.kernels.ops.layernorm.mhc"
    for name, hook in (
        ("hc_split_sinkhorn", around_hc_split_sinkhorn),
        ("prewarm_mhc_pre", around_prewarm_mhc_pre),
        ("mhc_pre", around_mhc_pre),
        ("mhc_post", around_mhc_post),
        ("mhc_fused_post_pre", around_mhc_fused_post_pre),
    ):
        HookRegistry.register(f"{mhc_target}.{name}", hook, HookType.AROUND)
    HookRegistry.register(
        "sglang.srt.layers.attention.vision.VisionFlash3Attention.__init__",
        around_vision_flash3_init,
        HookType.AROUND,
    )
