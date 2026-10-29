"""Align Torch-produced candidate scores at the native consumer boundary."""
import torch
from sglang.srt.environ import envs
from hcu_sglang.utils.detection import is_dcu

_is_hcu=is_dcu()


def aligned_candidate_scores(scores):
    if not (_is_hcu and scores.is_cuda and envs.SGLANG_USE_LIGHTOP_DSV41_CANDIDATE_BLOCKS.get()):
        return scores
    # Leave invalid dtypes/shapes to the native selector's existing diagnostics.
    if scores.ndim != 2 or scores.dtype != torch.float32:
        return scores
    if scores.stride(1)==1 and scores.stride(0)%4==0 and scores.data_ptr()%16==0:
        return scores
    rows,width=scores.shape
    storage=torch.empty((rows,(width+3)//4*4),device=scores.device,dtype=scores.dtype)
    aligned=storage[:,:width]
    aligned.copy_(scores)
    return aligned
