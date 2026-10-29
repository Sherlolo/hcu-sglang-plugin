"""HCU DSA index-key cache compatibility."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable, Optional

import torch

from hcu_sglang.utils.detection import (
    is_dcu,
    is_dcu_native_fp8_supported,
)


def _uses_bf16_index_cache(pool: Any) -> bool:
    return is_dcu() and not getattr(pool, "use_fp8_index_k_cache", True)


def _should_use_bf16_index_cache(dtype: torch.dtype) -> bool:
    return is_dcu() and (
        dtype not in (torch.float8_e4m3fn, torch.float8_e5m2)
        or not is_dcu_native_fp8_supported()
    )


def _index_cache_bytes_per_token(dtype: torch.dtype, index_head_dim: int) -> int:
    if _should_use_bf16_index_cache(dtype):
        return index_head_dim * torch.bfloat16.itemsize
    quant_block_size = 128
    return index_head_dim + index_head_dim // quant_block_size * 4


def _create_bf16_index_k_buffer(pool: Any) -> None:
    num_pages = (pool.index_buf_size + pool.page_size + 1) // pool.page_size
    with (
        torch.cuda.use_mem_pool(pool.custom_mem_pool)
        if pool.custom_mem_pool
        else nullcontext()
    ):
        pool.index_k_buffer = [
            torch.zeros(
                (
                    0 if pool.skip_topk_layers[layer_idx] else num_pages,
                    pool.page_size,
                    1,
                    pool.index_head_dim,
                ),
                dtype=pool.index_k_buffer_dtype,
                device=pool.device,
            )
            for layer_idx in range(pool.layer_num)
        ]


def around_init(
    original: Callable[..., None],
    pool: Any,
    size: int,
    page_size: int,
    kv_lora_rank: int,
    dtype: torch.dtype,
    qk_rope_head_dim: int,
    layer_num: int,
    device: str,
    index_head_dim: int,
    enable_memory_saver: bool,
    kv_cache_dim: int,
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
    index_buf_size: Optional[int] = None,
    skip_topk_layers: Optional[list[bool]] = None,
) -> None:
    if not is_dcu():
        return original(
            pool,
            size,
            page_size,
            kv_lora_rank,
            dtype,
            qk_rope_head_dim,
            layer_num,
            device,
            index_head_dim,
            enable_memory_saver,
            kv_cache_dim,
            start_layer,
            end_layer,
            index_buf_size,
            skip_topk_layers,
        )

    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    override_dim = (
        kv_cache_dim if kv_cache_dim != kv_lora_rank + qk_rope_head_dim else None
    )
    MLATokenToKVPool.__init__(
        pool,
        size,
        page_size,
        dtype,
        kv_lora_rank,
        qk_rope_head_dim,
        layer_num,
        device,
        enable_memory_saver,
        start_layer,
        end_layer,
        use_dsa=True,
        override_kv_cache_dim=override_dim,
    )

    pool.index_head_dim = index_head_dim
    pool.index_buf_size = size if index_buf_size is None else index_buf_size
    assert index_head_dim == 128
    pool.skip_topk_layers = (
        list(skip_topk_layers) if skip_topk_layers is not None else [False] * layer_num
    )
    assert len(pool.skip_topk_layers) == layer_num

    pool.use_fp8_index_k_cache = not _should_use_bf16_index_cache(dtype)
    pool.index_k_buffer_dtype = dtype if pool.use_fp8_index_k_cache else torch.bfloat16
    assert page_size == 64, f"HCU DSA path requires page_size == 64, got {page_size}"
    pool.index_key_cache = pool._create_index_key_cache()
    pool._finalize_allocation_log(size)


def around_create_index_key_cache(original: Callable[..., Any], pool: Any):
    if not is_dcu():
        return original(pool)
    if not pool.use_fp8_index_k_cache:
        _create_bf16_index_k_buffer(pool)
        return None
    pool.index_k_buffer = None
    return original(pool)


def get_index_k_with_scale_buffer_property(pool: Any):
    index_key_cache = getattr(pool, "index_key_cache", None)
    return None if index_key_cache is None else index_key_cache.buffer


def around_clear_buffers(original: Callable[..., None], pool: Any) -> None:
    if not _uses_bf16_index_cache(pool):
        return original(pool)

    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    MLATokenToKVPool._clear_buffers(pool)
    if hasattr(pool, "index_k_buffer"):
        del pool.index_k_buffer


def around_move_kv_cache(
    original: Callable[..., None],
    pool: Any,
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
) -> None:
    if not _uses_bf16_index_cache(pool):
        return original(pool, tgt_loc, src_loc)

    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    MLATokenToKVPool.move_kv_cache(pool, tgt_loc, src_loc)
    if tgt_loc.numel() == 0:
        return None
    tgt_loc_flat = tgt_loc.view(-1).long()
    src_loc_flat = src_loc.view(-1).long()
    for index_k in pool.index_k_buffer:
        if index_k.shape[0] == 0:
            continue
        flat_index_k = index_k.view(-1, 1, pool.index_head_dim)
        flat_index_k[tgt_loc_flat] = flat_index_k[src_loc_flat]
    return None


def get_index_k_buffer(pool: Any, layer_id: int) -> torch.Tensor:
    assert pool.index_k_buffer is not None, "BF16 index K cache is not enabled"
    if pool.layer_transfer_counter is not None:
        pool.layer_transfer_counter.wait_until(layer_id - pool.start_layer)
    return pool.index_k_buffer[layer_id - pool.start_layer]


def around_get_index_k_with_scale_buffer(
    original: Callable[..., torch.Tensor], pool: Any, layer_id: int
):
    assert not _uses_bf16_index_cache(pool), "FP8 index K cache is not enabled"
    return original(pool, layer_id)


def around_get_index_k_continuous(
    original: Callable[..., torch.Tensor],
    pool: Any,
    layer_id: int,
    seq_len: int,
    page_indices: torch.Tensor,
):
    if not _uses_bf16_index_cache(pool):
        return original(pool, layer_id, seq_len, page_indices)
    num_pages = (seq_len + pool.page_size - 1) // pool.page_size
    buf = get_index_k_buffer(pool, layer_id)
    return buf[page_indices[:num_pages]].view(-1, 1, pool.index_head_dim)[:seq_len]


def _assert_fp8_index_cache(pool: Any) -> None:
    assert not _uses_bf16_index_cache(pool), "FP8 index K cache is not enabled"


def around_fp8_only(original: Callable[..., Any], pool: Any, *args, **kwargs):
    _assert_fp8_index_cache(pool)
    return original(pool, *args, **kwargs)


def set_index_k_buffer(
    pool: Any,
    layer_id: int,
    loc: torch.Tensor,
    index_k: torch.Tensor,
) -> None:
    assert pool.index_k_buffer is not None, "BF16 index K cache is not enabled"
    if index_k.dtype != pool.index_k_buffer_dtype:
        index_k = index_k.to(pool.index_k_buffer_dtype)
    pool.index_k_buffer[layer_id - pool.start_layer][
        loc // pool.page_size, loc % pool.page_size
    ] = index_k


def _copy_bf16_index_cache_to_cpu(pool: Any, indices: torch.Tensor):
    page_indices = indices[:: pool.page_size] // pool.page_size
    torch.cuda.synchronize()
    index_k_cpu = []
    page_chunk_size = max(1, pool.cpu_offloading_chunk_size // pool.page_size)
    for index_k in pool.index_k_buffer:
        layer_chunks = []
        index_k_cpu.append(layer_chunks)
        if index_k.shape[0] == 0:
            continue
        for offset in range(0, len(page_indices), page_chunk_size):
            chunk_page_indices = page_indices[offset : offset + page_chunk_size]
            layer_chunks.append(
                index_k[chunk_page_indices].to("cpu", non_blocking=True)
            )
    torch.cuda.synchronize()
    return index_k_cpu


def around_get_cpu_copy(
    original: Callable[..., Any],
    pool: Any,
    indices: torch.Tensor,
    mamba_indices=None,
):
    if not _uses_bf16_index_cache(pool):
        return original(pool, indices, mamba_indices=mamba_indices)

    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    kv_cache_cpu = MLATokenToKVPool.get_cpu_copy(
        pool, indices, mamba_indices=mamba_indices
    )
    return {
        "kv": kv_cache_cpu,
        "index_k": _copy_bf16_index_cache_to_cpu(pool, indices),
    }


def _load_bf16_index_cache_from_cpu(
    pool: Any, index_k_cpu: list[list[torch.Tensor]], indices: torch.Tensor
) -> None:
    page_indices = indices[:: pool.page_size] // pool.page_size
    torch.cuda.synchronize()
    page_chunk_size = max(1, pool.cpu_offloading_chunk_size // pool.page_size)
    for layer_id, index_k in enumerate(pool.index_k_buffer):
        if index_k.shape[0] == 0:
            continue
        for offset in range(0, len(page_indices), page_chunk_size):
            chunk_page_indices = page_indices[offset : offset + page_chunk_size]
            idx_cpu = index_k_cpu[layer_id][offset // page_chunk_size]
            assert idx_cpu.shape[0] == len(chunk_page_indices)
            index_k[chunk_page_indices] = idx_cpu.to(index_k.device, non_blocking=True)
    torch.cuda.synchronize()


def around_load_cpu_copy(
    original: Callable[..., None],
    pool: Any,
    kv_cache_cpu_dict: Any,
    indices: torch.Tensor,
    mamba_indices=None,
) -> None:
    if not _uses_bf16_index_cache(pool):
        return original(pool, kv_cache_cpu_dict, indices, mamba_indices=mamba_indices)

    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool

    MLATokenToKVPool.load_cpu_copy(
        pool,
        kv_cache_cpu_dict["kv"],
        indices,
        mamba_indices=mamba_indices,
    )
    _load_bf16_index_cache_from_cpu(pool, kv_cache_cpu_dict["index_k"], indices)


def around_get_state_buf_infos(original: Callable[..., Any], pool: Any):
    if not _uses_bf16_index_cache(pool):
        return original(pool)
    data_ptrs = [buffer.data_ptr() for buffer in pool.index_k_buffer]
    data_lens = [buffer.nbytes for buffer in pool.index_k_buffer]
    item_lens = [
        0 if buffer.shape[0] == 0 else buffer[0].nbytes
        for buffer in pool.index_k_buffer
    ]
    return data_ptrs, data_lens, item_lens


def around_get_kv_size_bytes(original: Callable[..., int], pool: Any) -> int:
    if not _uses_bf16_index_cache(pool):
        return original(pool)

    from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool, get_tensor_size_bytes

    size_bytes = MLATokenToKVPool.get_kv_size_bytes(pool)
    return size_bytes + sum(
        get_tensor_size_bytes(index_k) for index_k in pool.index_k_buffer
    )


def around_compute_dsa_indexer_cell_size(
    original: Callable[..., int],
    configurator: Any,
    *,
    kvc: Any,
    num_layers: int,
    allocate_all_layers: bool = False,
) -> int:
    if not _should_use_bf16_index_cache(kvc.kv_cache_dtype):
        return original(
            configurator,
            kvc=kvc,
            num_layers=num_layers,
            allocate_all_layers=allocate_all_layers,
        )

    from sglang.srt.configs.model_config import (
        dsa_layer_skips_topk,
        get_dsa_index_head_dim,
    )
    from sglang.srt.mem_cache.kv_cache_configurator import (
        _should_elide_dsa_index_k,
    )
    from sglang.srt.runtime_context import get_memory

    memory_config = get_memory()
    indexer_ratio = 1
    if memory_config.enable_hisparse:
        from sglang.srt.mem_cache.sparsity import parse_hisparse_config

        indexer_ratio = parse_hisparse_config(kvc.server_args).host_to_device_ratio

    if allocate_all_layers or not _should_elide_dsa_index_k(
        is_draft_worker=kvc.is_draft_worker
    ):
        num_indexer_layers = num_layers
    else:
        active_indexer_layers = [
            layer_id
            for layer_id in range(kvc.layer_info.start_layer, kvc.layer_info.end_layer)
            if not dsa_layer_skips_topk(kvc.model_config.hf_config, layer_id)
        ]
        from sglang.srt.layers.cp.utils import (
            get_glm_dsa_cp_layer_shard_info,
            get_layer_shard_range,
        )

        _, shard_size = get_glm_dsa_cp_layer_shard_info(kvc)
        if shard_size > 1:
            active_set = set(active_indexer_layers)
            max_owned = 0
            for rank in range(shard_size):
                start, end = get_layer_shard_range(rank, shard_size, num_layers)
                max_owned = max(
                    max_owned,
                    sum(
                        kvc.layer_info.start_layer + offset in active_set
                        for offset in range(start, end)
                    ),
                )
            num_indexer_layers = max_owned + 1
        else:
            num_indexer_layers = len(active_indexer_layers)

    index_head_dim = get_dsa_index_head_dim(kvc.model_config.hf_config)
    return int(
        _index_cache_bytes_per_token(kvc.kv_cache_dtype, index_head_dim)
        * num_indexer_layers
        * indexer_ratio
    )


def after_hybrid_swa_init(result: Any, configurator: Any, kvc: Any):
    if not _should_use_bf16_index_cache(kvc.kv_cache_dtype):
        return result

    from sglang.srt.configs.model_config import (
        get_dsa_index_head_dim,
        is_deepseek_dsa,
    )

    if not is_deepseek_dsa(kvc.model_config.hf_config):
        return result
    index_head_dim = get_dsa_index_head_dim(kvc.model_config.hf_config)
    quantized_bytes = index_head_dim + index_head_dim // 128 * 4
    bf16_bytes = _index_cache_bytes_per_token(kvc.kv_cache_dtype, index_head_dim)
    configurator._full_per_token += bf16_bytes - quantized_bytes
    return result


def _fp8_contiguous(pool, layer_id, seq_len, page_indices):
    lengths = torch.tensor([seq_len], device=page_indices.device, dtype=torch.int32)
    return pool.index_key_cache.get_k_and_scale(
        layer_id, lengths, page_indices.reshape(1, -1), seq_len, seq_len
    )


def _compat_index_k_continuous(pool, layer_id, seq_len, page_indices):
    key, _scale = _fp8_contiguous(pool, layer_id, seq_len, page_indices)
    return key.reshape(seq_len, 1, pool.index_head_dim)


def _compat_index_scale_continuous(pool, layer_id, seq_len, page_indices):
    _key, scale = _fp8_contiguous(pool, layer_id, seq_len, page_indices)
    return scale


def _install_compat_members() -> None:
    from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool

    if not hasattr(DSATokenToKVPool, "get_index_k_continuous"):
        DSATokenToKVPool.get_index_k_continuous = _compat_index_k_continuous
    if not hasattr(DSATokenToKVPool, "get_index_k_scale_continuous"):
        DSATokenToKVPool.get_index_k_scale_continuous = _compat_index_scale_continuous
    DSATokenToKVPool.index_k_with_scale_buffer = property(
        get_index_k_with_scale_buffer_property
    )
    if not hasattr(DSATokenToKVPool, "get_index_k_buffer"):
        DSATokenToKVPool.get_index_k_buffer = get_index_k_buffer
    if not hasattr(DSATokenToKVPool, "set_index_k_buffer"):
        DSATokenToKVPool.set_index_k_buffer = set_index_k_buffer


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _install_compat_members()
    target = "sglang.srt.mem_cache.memory_pool.DSATokenToKVPool"
    hooks = {
        "__init__": around_init,
        "_create_index_key_cache": around_create_index_key_cache,
        "_clear_buffers": around_clear_buffers,
        "move_kv_cache": around_move_kv_cache,
        "get_index_k_with_scale_buffer": around_get_index_k_with_scale_buffer,
        "get_index_k_continuous": around_get_index_k_continuous,
        "get_index_k_scale_continuous": around_fp8_only,
        "get_index_k_scale_buffer": around_fp8_only,
        "set_index_k_scale_buffer": around_fp8_only,
        "get_cpu_copy": around_get_cpu_copy,
        "load_cpu_copy": around_load_cpu_copy,
        "get_state_buf_infos": around_get_state_buf_infos,
        "get_kv_size_bytes": around_get_kv_size_bytes,
    }
    for method_name, hook in hooks.items():
        HookRegistry.register(f"{target}.{method_name}", hook, HookType.AROUND)
    HookRegistry.register(
        "sglang.srt.model_executor.pool_configurator.DefaultPoolConfigurator._compute_dsa_indexer_cell_size",
        around_compute_dsa_indexer_cell_size,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.model_executor.pool_configurator.HybridSWAPoolConfigurator.__init__",
        after_hybrid_swa_init,
        HookType.AFTER,
    )
