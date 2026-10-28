"""Second CP communicator owned by the plugin, with the HCU partition contract."""
from __future__ import annotations
from typing import Optional
import torch
from sglang.srt.distributed.parallel_state import GroupCoordinator, get_world_group, init_model_parallel_group
from sglang.srt.runtime_context import get_parallel
_ATTN_CP_OVERLAP = None

def get_attn_cp_group():
    return get_parallel().attn_cp_group

def get_attn_cp_overlap_group() -> GroupCoordinator:
    return _ATTN_CP_OVERLAP if _ATTN_CP_OVERLAP is not None else get_attn_cp_group()

def _init_attn_cp_overlap_group(*, world_size: int, attn_cp_size: int, attn_tp_size: int, backend: Optional[str], recovered_rank: bool, rank_offset: int, max_world_size: Optional[int]) -> None:
    """Second communicator over the attention CP ranks; RCCL deadlocks when one
    communicator is driven from two streams at once."""
    global _ATTN_CP_OVERLAP
    assert _ATTN_CP_OVERLAP is None, 'attention context parallel overlap group is already initialized'
    if attn_cp_size <= 1:
        return
    span = attn_tp_size * attn_cp_size
    group_ranks = [list(range(base + i, base + i + span, attn_tp_size)) for base in range(0, world_size, span) for i in range(attn_tp_size)]
    rank = torch.distributed.get_rank()
    mine = next((ranks for ranks in group_ranks if rank in ranks))
    assert mine == get_attn_cp_group().ranks, f'attn_cp_overlap partition {mine} does not match attn_cp {get_attn_cp_group().ranks}; the two communicators must span the same ranks or the overlapped collectives will not pair up'
    _ATTN_CP_OVERLAP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, use_message_queue_broadcaster=False, group_name='attn_cp_overlap', recovered_rank=recovered_rank, rank_offset=rank_offset, max_world_size=max_world_size)



def around_initialize_model_parallel(original,*args,**kwargs):
    import inspect
    bound=inspect.signature(original).bind(*args,**kwargs)
    bound.apply_defaults()
    result=original(*args,**kwargs)
    parallel=get_parallel()
    if parallel.attn_cp_size > 1:
        world=get_world_group()
        backend=bound.arguments['backend']
        if backend is None:
            backend=torch.distributed.get_backend(world.device_group)
        _init_attn_cp_overlap_group(world_size=world.world_size,
            attn_cp_size=parallel.attn_cp_size,attn_tp_size=parallel.attn_tp_size,
            backend=backend,recovered_rank=bound.arguments['recovered_rank'],
            rank_offset=bound.arguments['rank_offset'],max_world_size=bound.arguments['max_world_size'])
    return result


def around_destroy_model_parallel(original,*args,**kwargs):
    global _ATTN_CP_OVERLAP
    group=_ATTN_CP_OVERLAP
    _ATTN_CP_OVERLAP=None
    try:
        if group is not None:
            group.destroy()
    finally:
        result=original(*args,**kwargs)
    return result
