"""HCU LightOp routing for legacy DSA fused Top-K transforms."""

from __future__ import annotations

from typing import Any, Callable, List, Optional

import torch

from hcu_sglang.utils.detection import is_dcu


def _uses_v2_transform(
    backend,
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    topk_transform_method,
    attn_metadata,
    topk_indices_offset: Optional[torch.Tensor],
    row_starts: Optional[torch.Tensor],
    batch_idx_list: Optional[List[int]],
) -> bool:
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod

    if not backend.should_use_topk_v2() or not 0 < topk <= 2048:
        return False
    if topk_transform_method == TopkTransformMethod.PAGED:
        return (
            row_starts is None
            and batch_idx_list is None
            and lengths.shape[0]
            == logits.shape[0]
            == attn_metadata.real_page_table.shape[0]
        )
    if topk_transform_method == TopkTransformMethod.RAGGED:
        return (
            topk_indices_offset is not None
            and batch_idx_list is None
            and lengths.shape[0] == logits.shape[0] == topk_indices_offset.shape[0]
        )
    return False


def around_topk_transform(
    original: Callable[..., torch.Tensor],
    backend,
    logits: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    topk_transform_method,
    attn_metadata,
    cu_seqlens_q_topk: Optional[torch.Tensor] = None,
    topk_indices_offset: Optional[torch.Tensor] = None,
    row_starts: Optional[torch.Tensor] = None,
    batch_idx_list: Optional[List[int]] = None,
    force_unfused_topk: bool = False,
) -> torch.Tensor:
    from sglang.srt.environ import envs

    if (
        not is_dcu()
        or not backend.is_sgl_kernel()
        or not envs.SGLANG_DSA_FUSE_TOPK.get()
        or force_unfused_topk
        or _uses_v2_transform(
            backend,
            logits,
            lengths,
            topk,
            topk_transform_method,
            attn_metadata,
            topk_indices_offset,
            row_starts,
            batch_idx_list,
        )
    ):
        return original(
            backend,
            logits,
            lengths,
            topk,
            topk_transform_method,
            attn_metadata,
            cu_seqlens_q_topk,
            topk_indices_offset,
            row_starts,
            batch_idx_list,
            force_unfused_topk,
        )

    from lightop import (
        fast_topk_transform_fused,
        fast_topk_transform_ragged_fused,
    )
    from sglang.srt.layers.attention.dsa.dsa_topk_backend import TopkTransformMethod

    if attn_metadata.page_table_1 is None:
        raise RuntimeError("Legacy DSA Top-K requires a page_size=1 page table")
    if topk_transform_method == TopkTransformMethod.PAGED:
        page_table_size_1 = (
            attn_metadata.page_table_1[batch_idx_list]
            if batch_idx_list is not None
            else attn_metadata.page_table_1
        )
        return fast_topk_transform_fused(
            score=logits,
            lengths=lengths,
            page_table_size_1=page_table_size_1,
            cu_seqlens_q=cu_seqlens_q_topk,
            topk=topk,
            row_starts=row_starts,
        )
    if topk_transform_method == TopkTransformMethod.RAGGED:
        if topk_indices_offset is None:
            raise RuntimeError(
                "RAGGED topk_transform requires topk_indices_offset; "
                "expected extend-without-speculative metadata."
            )
        return fast_topk_transform_ragged_fused(
            score=logits,
            lengths=lengths,
            topk_indices_offset=topk_indices_offset,
            topk=topk,
            row_starts=row_starts,
        )
    raise RuntimeError(f"Unsupported topk transform method: {topk_transform_method}")


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.attention.dsa.dsa_topk_backend."
        "DSATopKBackend.topk_transform",
        around_topk_transform,
        HookType.AROUND,
    )
