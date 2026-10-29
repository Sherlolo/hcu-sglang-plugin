"""DCU sglang.srt.models.utils fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

from hcu_sglang.models.model_utils import constants as C
from hcu_sglang.models.model_utils.methods._runtime import (
    ForwardBatch, SWAKVPool, _is_cuda, _is_hip, is_cp_active, torch,
)

def enable_fused_set_kv_buffer(forward_batch: ForwardBatch):
    """Enable fused set_kv_buffer only on CUDA with bfloat16 KV cache."""
    token_to_kv_pool = get_token_to_kv_pool()
    cuda_enabled = (
        _is_cuda
        and hasattr(token_to_kv_pool, "dtype")
        and token_to_kv_pool.dtype == torch.bfloat16
        and not isinstance(token_to_kv_pool, SWAKVPool)
        and not is_cp_active(forward_batch)
        and getattr(forward_batch, "dcp_kv_mask", None) is None
    )
    if C._is_dcu:
        return cuda_enabled
    return cuda_enabled or (
        _is_hip and getattr(forward_batch, "dcp_kv_mask", None) is None
    )
