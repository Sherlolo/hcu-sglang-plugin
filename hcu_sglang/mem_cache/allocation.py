"""HCU request-table and paged-KV allocation hooks."""

from __future__ import annotations

from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_dcu


def _enabled(name: str) -> bool:
    from sglang.srt.utils import get_bool_env_var

    return get_bool_env_var(name, default="true")


def _load_kvcacheio():
    from sgl_kernel import kvcacheio

    # sgl_kernel.kvcacheio may be shadowed by a port-only copy (the adapter
    # ships pure-torch ports of the transfer ops but not the alloc kernels).
    # The compiled library still registers the ops on torch.ops.sgl_kernel,
    # so shim any missing attribute with a wrapper carrying the wheel's
    # python signature (keyword-friendly, forwards positionally).
    needed = (
        "hcu_alloc_extend_kernel",
        "hcu_alloc_decode_kernel",
        "hcu_get_last_loc",
        "hcu_assign_req_to_token_pool",
    )
    missing = [n for n in needed if not hasattr(kvcacheio, n)]
    if missing:
        ops = torch.ops.sgl_kernel

        def _alloc_extend(pre_lens_ptr, seq_lens_ptr, last_loc_ptr,
                          free_page_ptr, out_indices, bs, page_size):
            ops.hcu_alloc_extend_kernel(
                pre_lens_ptr, seq_lens_ptr, last_loc_ptr, free_page_ptr,
                out_indices, bs, page_size)

        def _alloc_decode(seq_lens_ptr, last_loc_ptr, free_page_ptr,
                          out_indices, bs, page_size):
            ops.hcu_alloc_decode_kernel(
                seq_lens_ptr, last_loc_ptr, free_page_ptr, out_indices,
                bs, page_size)

        def _get_last_loc(req_to_token, req_pool_indices, prefix_lens):
            return ops.hcu_get_last_loc(
                req_to_token, req_pool_indices, prefix_lens)

        def _assign_req_to_token_pool(req_pool_indices, req_to_token,
                                      allocate_lens, new_allocate_lens,
                                      out_cache_loc, shape, bs):
            ops.hcu_assign_req_to_token_pool(
                req_pool_indices, req_to_token, allocate_lens,
                new_allocate_lens, out_cache_loc, shape, bs)

        shims = {
            "hcu_alloc_extend_kernel": _alloc_extend,
            "hcu_alloc_decode_kernel": _alloc_decode,
            "hcu_get_last_loc": _get_last_loc,
            "hcu_assign_req_to_token_pool": _assign_req_to_token_pool,
        }
        for name in missing:
            if hasattr(ops, name):
                setattr(kvcacheio, name, shims[name])
    return kvcacheio


def around_get_last_loc(
    original: Callable[..., torch.Tensor],
    req_to_token: torch.Tensor,
    req_pool_indices_tensor: torch.Tensor,
    prefix_lens_tensor: torch.Tensor,
) -> torch.Tensor:
    if not is_dcu() or not _enabled("SGLANG_GET_LAST_LOC"):
        return original(
            req_to_token,
            req_pool_indices_tensor,
            prefix_lens_tensor,
        )
    return _load_kvcacheio().hcu_get_last_loc(
        req_to_token,
        req_pool_indices_tensor,
        prefix_lens_tensor,
    )


def around_assign_req_to_token_pool(
    original: Callable[..., Any],
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
) -> Any:
    if not is_dcu() or not _enabled("SGLANG_ASSIGN_REQ_TO_TOKEN_POOL"):
        return original(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            batch_size,
        )
    _load_kvcacheio().hcu_assign_req_to_token_pool(
        req_pool_indices=req_pool_indices,
        req_to_token=req_to_token,
        allocate_lens=start_offset,
        new_allocate_lens=end_offset,
        out_cache_loc=out_cache_loc,
        shape=req_to_token.shape[1],
        bs=batch_size,
    )
    return None


def after_paged_allocator_init(result: Any, allocator: Any, *args, **kwargs) -> Any:
    if is_dcu():
        allocator.sglang_kvalloc_kernel = _enabled("SGLANG_KVALLOC_KERNEL")
    return result


def around_alloc_extend(
    original: Callable[..., torch.Tensor | None],
    allocator: Any,
    prefix_lens: torch.Tensor,
    prefix_lens_cpu: torch.Tensor,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
    extend_num_tokens: int,
    num_new_pages: int | None = None,
) -> torch.Tensor | None:
    if not is_dcu() or not getattr(allocator, "sglang_kvalloc_kernel", False):
        return original(
            allocator,
            prefix_lens,
            prefix_lens_cpu,
            seq_lens,
            seq_lens_cpu,
            last_loc,
            extend_num_tokens,
            num_new_pages,
        )

    from sglang.srt.utils import get_num_new_pages

    if allocator.debug_mode:
        assert torch.all(
            (last_loc + 1) % allocator.page_size == prefix_lens % allocator.page_size
        )

    batch_size = len(prefix_lens)
    if (
        allocator.need_sort
        and extend_num_tokens // allocator.page_size + batch_size + 1
        > len(allocator.free_pages)
    ):
        allocator.merge_and_sort_free()

    out_indices = torch.empty(
        (extend_num_tokens,),
        dtype=torch.int64,
        device=allocator.device,
    )
    _load_kvcacheio().hcu_alloc_extend_kernel(
        pre_lens_ptr=prefix_lens.to(torch.int64),
        seq_lens_ptr=seq_lens.to(torch.int64),
        last_loc_ptr=last_loc.to(torch.int64),
        free_page_ptr=allocator.free_pages.to(torch.int64),
        out_indices=out_indices,
        bs=batch_size,
        page_size=allocator.page_size,
    )

    if allocator.debug_mode:
        assert len(torch.unique(out_indices)) == len(out_indices)

    if num_new_pages is None:
        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=allocator.page_size,
            prefix_lens=prefix_lens_cpu,
        )
    if num_new_pages > len(allocator.free_pages):
        return None

    allocator.free_pages = allocator.free_pages[num_new_pages:]
    return out_indices


def around_alloc_decode(
    original: Callable[..., torch.Tensor | None],
    allocator: Any,
    seq_lens: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    last_loc: torch.Tensor,
) -> torch.Tensor | None:
    if not is_dcu() or not getattr(allocator, "sglang_kvalloc_kernel", False):
        return original(
            allocator,
            seq_lens,
            seq_lens_cpu,
            last_loc,
        )

    from sglang.srt.utils import get_num_new_pages

    if allocator.debug_mode:
        assert torch.all(
            (last_loc + 2) % allocator.page_size == seq_lens % allocator.page_size
        )

    batch_size = len(seq_lens)
    if allocator.need_sort and batch_size > len(allocator.free_pages):
        allocator.merge_and_sort_free()

    out_indices = torch.empty(
        (batch_size,),
        dtype=torch.int64,
        device=allocator.device,
    )
    _load_kvcacheio().hcu_alloc_decode_kernel(
        seq_lens_ptr=seq_lens,
        last_loc_ptr=last_loc.to(dtype=torch.int32, copy=False),
        free_page_ptr=allocator.free_pages,
        out_indices=out_indices,
        bs=batch_size,
        page_size=allocator.page_size,
    )

    if allocator.debug_mode:
        assert len(torch.unique(out_indices)) == len(out_indices)

    num_new_pages = get_num_new_pages(
        seq_lens=seq_lens_cpu,
        page_size=allocator.page_size,
        decode=True,
    )
    if num_new_pages > len(allocator.free_pages):
        return None

    allocator.free_pages = allocator.free_pages[num_new_pages:]
    return out_indices


def around_free(
    original: Callable[..., Any],
    allocator: Any,
    free_index: torch.Tensor,
) -> Any:
    if not is_dcu():
        return original(allocator, free_index)
    if free_index.numel() == 0:
        return None

    if allocator.is_not_in_free_group:
        free_page_indices = torch.unique(free_index // allocator.page_size)
        free_page_indices = free_page_indices[free_page_indices > 0]
        if free_page_indices.numel() == 0:
            return None
        free_page_indices = free_page_indices[
            ~torch.isin(free_page_indices, allocator.free_pages)
        ]
        if allocator.release_pages.numel() > 0:
            free_page_indices = free_page_indices[
                ~torch.isin(free_page_indices, allocator.release_pages)
            ]
        if free_page_indices.numel() == 0:
            return None
        allocator._release_page_ids(free_page_indices)
    else:
        allocator.free_group.append(allocator._copy_for_free_group(free_index))

    if allocator.debug_mode:
        allocator._debug_check_no_duplicate_pages()
    return None


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.mem_cache.allocation.get_last_loc",
        around_get_last_loc,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.mem_cache.allocation.assign_req_to_token_pool_func",
        around_assign_req_to_token_pool,
        HookType.AROUND,
    )
    target = "sglang.srt.mem_cache.allocator.paged.PagedTokenToKVPoolAllocator"
    HookRegistry.register(
        f"{target}.__init__",
        after_paged_allocator_init,
        HookType.AFTER,
    )
    HookRegistry.register(
        f"{target}.alloc_extend",
        around_alloc_extend,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{target}.alloc_decode",
        around_alloc_decode,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{target}.free",
        around_free,
        HookType.AROUND,
    )
