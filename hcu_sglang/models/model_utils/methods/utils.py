"""DCU sglang.srt.models.utils fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from hcu_sglang.models.model_utils import constants as C
from hcu_sglang.models.model_utils.methods._runtime import *  # noqa: F403

def enable_fused_set_kv_buffer(forward_batch: ForwardBatch):
    """Enable fused set_kv_buffer only on CUDA with bfloat16 KV cache."""
    if C._is_dcu:
        return (
            _is_cuda
            and hasattr(forward_batch.token_to_kv_pool, "dtype")
            and forward_batch.token_to_kv_pool.dtype == torch.bfloat16  # gitleaks:allow
            and not isinstance(forward_batch.token_to_kv_pool, SWAKVPool)
            and not is_prefill_context_parallel_enabled()
        )
    else:
        return (
            _is_cuda
            and hasattr(forward_batch.token_to_kv_pool, "dtype")
            and forward_batch.token_to_kv_pool.dtype == torch.bfloat16  # gitleaks:allow
            and not isinstance(forward_batch.token_to_kv_pool, SWAKVPool)
            and not is_prefill_context_parallel_enabled()
        ) or (_is_hip and not is_prefill_context_parallel_enabled())
