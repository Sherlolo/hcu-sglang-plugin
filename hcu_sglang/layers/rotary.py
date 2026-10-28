# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Licensed under the Apache License, Version 2.0; see LICENSE.
"""HCU rotary fallback consumer: typed cache view and bounded launches.

Head splitting derives from pinned HCU 5f67ce01c3 rotary_embedding/base.py.
The canonical FP32 cache remains owned by the upstream layer and its indexers.
"""
from __future__ import annotations
from typing import Optional
import torch
from hcu_sglang.utils.detection import is_dcu

_HCU_ROPE_MAX_THREADS = 256

def _max_q_heads_within_launch_bound(rot_dim: int) -> int:
    if rot_dim <= 0:
        return 1
    heads = max(1, (_HCU_ROPE_MAX_THREADS * 2) // rot_dim)
    while heads > 1 and heads * rot_dim // 2 > _HCU_ROPE_MAX_THREADS:
        heads -= 1
    return heads


def _launch_rotary_embedding(
    fn,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: Optional[torch.Tensor],
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    """Launch sgl_kernel.rotary_embedding without exceeding the HCU block limit.

    The kernel strides with blockDim.x, so splitting query heads across several
    launches is numerically the same as one launch. Key is applied once.
    """
    num_tokens = int(positions.numel())
    if num_tokens == 0:
        return
    rot_dim = int(cos_sin_cache.shape[-1])
    nq = query.numel() // num_tokens // head_size
    nk = 0 if key is None else key.numel() // num_tokens // head_size
    if nq * rot_dim // 2 <= _HCU_ROPE_MAX_THREADS:
        fn(positions, query, key, head_size, cos_sin_cache, is_neox)
        return

    max_q_heads = _max_q_heads_within_launch_bound(rot_dim)
    if max_q_heads * rot_dim // 2 > _HCU_ROPE_MAX_THREADS:
        fn(positions, query, key, head_size, cos_sin_cache, is_neox)
        return

    pos = positions.reshape(-1)
    q = query.reshape(num_tokens, nq, head_size)
    k = None if key is None else key.reshape(num_tokens, nk, head_size)

    def _launch(q_slice, k_slice):
        fn(pos, q_slice, k_slice, head_size, cos_sin_cache, is_neox)

    if k is None or nk == nq:
        for start in range(0, nq, max_q_heads):
            end = min(start + max_q_heads, nq)
            _launch(q[:, start:end], None if k is None else k[:, start:end])
        return

    if nk == 1:
        _launch(q[:, :max_q_heads], k)
        for start in range(max_q_heads, nq, max_q_heads):
            end = min(start + max_q_heads, nq)
            _launch(q[:, start:end], None)
        return

    group = nq // nk
    q_step = (max_q_heads // group) * group
    if group == 0 or q_step < group or nq % nk != 0:
        fn(positions, query, key, head_size, cos_sin_cache, is_neox)
        return
    for q_start in range(0, nq, q_step):
        q_end = min(q_start + q_step, nq)
        _launch(q[:, q_start:q_end], k[:, q_start // group : q_end // group])


class DcuRotaryFallback:
    def __init__(self, original):
        self.original = original
        self._cache_memo = None

    def __call__(self, positions, query, key, head_size, cos_sin_cache, is_neox):
        cache = cos_sin_cache
        if cache.device != query.device or cache.dtype != query.dtype:
            memo = self._cache_memo
            if (memo is None or memo[0] is not cache or
                    memo[1].device != query.device or memo[1].dtype != query.dtype):
                memo = (cache, cache.to(device=query.device, dtype=query.dtype))
                self._cache_memo = memo
            cache = memo[1]
        _launch_rotary_embedding(self.original, positions, query, key,
                                 head_size, cache, is_neox)


def after_rotary_init(result, layer, *args, **kwargs):
    if is_dcu() and getattr(layer, 'use_fallback_kernel', False):
        current = layer.fallback_rotary_embedding
        if not isinstance(current, DcuRotaryFallback):
            layer.fallback_rotary_embedding = DcuRotaryFallback(current)
    return result


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register(
        'sglang.srt.layers.rotary_embedding.base.RotaryEmbedding.__init__',
        after_rotary_init, HookType.AFTER,
    )
