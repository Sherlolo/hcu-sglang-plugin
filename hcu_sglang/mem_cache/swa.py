"""HCU sliding-window allocator safety hooks."""

from __future__ import annotations

from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_dcu


def around_free_swa(
    original: Callable[..., Any],
    allocator: Any,
    free_index: torch.Tensor,
) -> Any:
    if not is_dcu():
        return original(allocator, free_index)
    if free_index.numel() == 0:
        return None

    if not allocator.is_not_in_free_group:
        allocator.swa_free_group.append(allocator._copy_for_free_group(free_index))
        return None

    mapping_indices = (
        free_index
        if allocator.page_size == 1
        else allocator._expand_to_full_pages(free_index)
    )
    mapping = allocator.full_to_swa_index_mapping
    swa_indices = mapping[mapping_indices]
    swa_indices = swa_indices[swa_indices > 0]
    if swa_indices.numel() == 0:
        mapping[mapping_indices] = 0
        return None

    unique_swa = torch.unique(swa_indices)
    page_size = allocator.swa_attn_allocator.page_size
    freed_swa_pages = torch.unique(unique_swa // page_size)
    if len(allocator.swa_attn_allocator.release_pages) > 0:
        allocator.swa_attn_allocator.merge_and_sort_free()
    already_free = torch.isin(
        unique_swa // page_size,
        allocator.swa_attn_allocator.free_pages,
    )
    unique_swa = unique_swa[~already_free]
    if unique_swa.numel() > 0:
        allocator.swa_attn_allocator.free(unique_swa)

    all_swa_pages = mapping // page_size
    stale = torch.isin(all_swa_pages, freed_swa_pages) & (mapping > 0)
    mapping[stale] = 0
    return None


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.mem_cache.allocator.swa.SWATokenToKVPoolAllocator.free_swa",
        around_free_swa,
        HookType.AROUND,
    )
