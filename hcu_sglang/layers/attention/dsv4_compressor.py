"""HCU DeepSeek-V4 compressor feature switches."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from hcu_sglang.utils.detection import is_dcu


def _use_fused_compress() -> bool:
    from sglang.srt.environ import envs

    return envs.SGLANG_OPT_USE_FUSED_COMPRESS.get()


def around_fused_softmax_pool(
    original: Callable[..., torch.Tensor],
    kv_score: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    if not is_dcu() or _use_fused_compress():
        return original(kv_score, head_dim)

    kv = kv_score[..., :head_dim]
    score = kv_score[..., head_dim:]
    return (kv * score.softmax(dim=1)).sum(dim=1)


def around_fused_norm_rope_inplace(
    original: Callable[..., Any],
    kv: torch.Tensor,
    weight: Optional[torch.Tensor],
    eps: float,
    freqs_cis: torch.Tensor,
    positions: Optional[torch.Tensor] = None,
) -> None:
    if not is_dcu() or _use_fused_compress():
        return original(kv, weight, eps, freqs_cis, positions)

    from sglang.kernels.ops.attention.deepseek_v4_rope import (
        apply_rotary_emb_triton,
    )
    from sglang.kernels.ops.layernorm.rms_normalize_hip import (
        rms_normalize_triton,
    )

    kv.copy_(rms_normalize_triton(kv, eps, weight))
    freqs_real = torch.view_as_real(freqs_cis).flatten(-2)
    rope_dim = freqs_real.shape[-1]
    apply_rotary_emb_triton(
        kv[..., -rope_dim:],
        freqs_cis,
        positions=positions,
    )
    return None


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    target = "sglang.srt.layers.attention.dsv4.compress_hip"
    HookRegistry.register(
        f"{target}.fused_softmax_pool_triton",
        around_fused_softmax_pool,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{target}.fused_norm_rope_inplace_triton",
        around_fused_norm_rope_inplace,
        HookType.AROUND,
    )
