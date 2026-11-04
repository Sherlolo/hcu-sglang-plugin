"""HCU sliding-window allocator safety hooks."""

from __future__ import annotations

from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_hcu


def around_free_swa(
    original: Callable[..., Any],
    allocator: Any,
    free_index: torch.Tensor,
) -> Any:
    return original(allocator, free_index)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.mem_cache.allocator.swa.SWATokenToKVPoolAllocator.free_swa",
        around_free_swa,
        HookType.AROUND,
    )
