"""Bridges for pinned official graph and CP APIs used by the HCU backend."""
from typing import Union, List, Optional
import torch
from sglang.srt.runtime_context import get_parallel
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp,is_dsa_prefill_cp_interleave
from hcu_sglang.utils.detection import is_dcu as is_hcu


def is_capturing() -> bool:
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import _current_capture_var
    return _current_capture_var.get() is not None


def is_dsa_prefill_cp_round_robin_split():
    if is_hcu():
        mode=getattr(get_parallel(), 'dsa_prefill_cp_mode', None)
        if mode is not None:
            return is_dsa_enable_prefill_cp() and mode == 'round-robin-split'
    return is_dsa_prefill_cp_interleave()


def dsa_cp_round_robin_split_data(input_: Union[torch.Tensor, List]):
    cp_size=get_parallel().attn_cp_size
    cp_rank=get_parallel().attn_cp_rank
    if isinstance(input_, (tuple,list)):
        return input_[cp_rank::cp_size]
    tokens=len(input_)
    if tokens % cp_size != 0:
        cur_len=tokens // cp_size + (tokens % cp_size > cp_rank)
        if cur_len == 0:
            return input_.new_empty(0,*input_.shape[1:])
        indices=torch.arange(cp_rank,tokens,cp_size,device=input_.device)
        return input_[indices]
    shard=input_.view(-1,cp_size,*input_.shape[1:])[:,cp_rank]
    if shard.stride(0) != input_.stride(0):
        shard=shard.clone(memory_format=torch.contiguous_format)
    return shard


def attn_cp_overlap_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_parallel().attn_cp_group.all_gather_into_tensor(output,input)


def attn_cp_all_to_all_single(output: torch.Tensor, input: torch.Tensor, output_split_sizes: Optional[List[int]]=None, input_split_sizes: Optional[List[int]]=None):
    """All-to-all over dim 0 within the attention CP group (variable splits ok).

    Used by the DSV4 compressor RLC path to repartition round-robin-scattered
    tokens into per-rank contiguous blocks.
    """
    return get_parallel().attn_cp_group.all_to_all_single(output, input, output_split_sizes=output_split_sizes, input_split_sizes=input_split_sizes)
