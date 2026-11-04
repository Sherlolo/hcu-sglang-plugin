from __future__ import annotations

import torch
from hcu_sglang.utils.detection import is_hcu

HCU_MLA_FP8_PADDED_KV_CACHE_DIM = 576


def is_hcu_mla_fp8_kv_dtype(kv_cache_dtype: torch.dtype) -> bool:
    return kv_cache_dtype in (
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    )


def should_pad_hcu_mla_fp8_kv_cache(
    kv_cache_dim: int,
    qk_rope_head_dim: int,
    kv_cache_dtype: torch.dtype,
    uses_hcu_mla: bool,
) -> bool:
    return (
        is_hcu()
        and uses_hcu_mla
        and qk_rope_head_dim == 0
        and kv_cache_dim == 512
        and is_hcu_mla_fp8_kv_dtype(kv_cache_dtype)
    )


def get_hcu_mla_fp8_kv_cache_dim(
    kv_cache_dim: int,
    qk_rope_head_dim: int,
    kv_cache_dtype: torch.dtype,
    uses_hcu_mla: bool,
) -> int:
    if should_pad_hcu_mla_fp8_kv_cache(
        kv_cache_dim, qk_rope_head_dim, kv_cache_dtype, uses_hcu_mla
    ):
        return HCU_MLA_FP8_PADDED_KV_CACHE_DIM
    return kv_cache_dim
