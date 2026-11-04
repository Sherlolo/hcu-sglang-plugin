from __future__ import annotations


def get_attn_cp_group():
    from sglang.srt.distributed.parallel_state import get_attn_cp_group as get_group
    return get_group()


def get_attn_cp_overlap_group():
    from sglang.srt.distributed.parallel_state import get_attn_cp_overlap_group as get_group
    return get_group()
