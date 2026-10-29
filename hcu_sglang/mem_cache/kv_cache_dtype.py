"""HCU KV-cache dtype selection hooks."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from hcu_sglang.utils.detection import is_dcu


def around_configure_kv_cache_dtype(
    original: Callable[..., tuple[Optional[str], torch.dtype]],
    *,
    server_args_kv_cache_dtype: str,
    model: Any,
    model_dtype: torch.dtype,
    is_draft_worker: bool,
    is_dflash: bool,
    speculative_draft_attention_backend: str,
    speculative_draft_kv_cache_dtype: Optional[str] = None,
) -> tuple[Optional[str], torch.dtype]:
    effective_dtype = server_args_kv_cache_dtype
    resolved_dtype: Optional[str] = None
    if is_draft_worker and speculative_draft_kv_cache_dtype is not None:
        effective_dtype = speculative_draft_kv_cache_dtype
        if effective_dtype != "auto":
            resolved_dtype = effective_dtype

    if not is_dcu() or effective_dtype != "fp8_e5m2":
        return original(
            server_args_kv_cache_dtype=server_args_kv_cache_dtype,
            model=model,
            model_dtype=model_dtype,
            is_draft_worker=is_draft_worker,
            is_dflash=is_dflash,
            speculative_draft_attention_backend=speculative_draft_attention_backend,
            speculative_draft_kv_cache_dtype=speculative_draft_kv_cache_dtype,
        )

    kv_cache_dtype = torch.float8_e5m2
    if (
        is_draft_worker
        and is_dflash
        and speculative_draft_attention_backend == "fa4"
        and kv_cache_dtype != model_dtype
    ):
        kv_cache_dtype = model_dtype
        resolved_dtype = "auto"
    return resolved_dtype, kv_cache_dtype


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.mem_cache.kv_cache_dtype.configure_kv_cache_dtype",
        around_configure_kv_cache_dtype,
        HookType.AROUND,
    )
