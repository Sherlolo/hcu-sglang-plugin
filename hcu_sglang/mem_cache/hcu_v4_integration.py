"""HCU pool registration and CP-aware factory hooks."""
from __future__ import annotations
from typing import Optional
import torch
from sglang.srt.runtime_context import (get_exec, get_memory, get_parallel, get_schedule,
    max_speculative_num_draft_tokens)
from sglang.srt.utils import is_npu
from hcu_sglang.config import cp_cache_layer_split_enabled
from hcu_sglang.mem_cache.hcu_v4_pool import (
    DeepSeekV4TokenToKVPool, select_dsv4_kv_layout,
)
from sglang.srt.mem_cache.kv_cache_configurator import unified_fp8_for_dsv4_pool
_is_npu = is_npu()

def _build_dsv4_kv_pool(
    self,
    *,
    max_running_requests: int,
    full_max_total_num_tokens: int,
    swa_max_total_num_tokens: Optional[int],
    c4_max_total_num_tokens: int,
    c128_max_total_num_tokens: int,
    c4_state_pool_size: int,
    c128_state_pool_size: int,
    c4_state_dtype: Optional[torch.dtype],
    c128_state_dtype: Optional[torch.dtype],
    req_to_token_pool: ReqToTokenPool,
) -> KVCache:
    swa_page_size = get_schedule().page_size
    if not _is_npu:
        assert swa_page_size == 256, "In paged swa mode, page_size must be 256."

    if self.is_draft_worker:
        from sglang.srt.models.deepseek_v4_nextn import (
            COMPRESS_RATIO_NEXTN_LAYER,
        )

        compression_ratios = [
            COMPRESS_RATIO_NEXTN_LAYER
        ] * self.layer_info.num_effective_layers
        kv_source_layers = []
    else:
        compression_ratios = self.model_config.compress_ratios
        kv_source_layers = list(self.model_config.hf_config.kv_source_layer_ids)

    # NPU keeps its PA_ND KV-pool subclass, while Compressor state sizing
    # follows the same fixed ring ownership as GPU. Do not replace the
    # configurator's C4-SWA/C128-request budgets with a paged allocator
    # estimate: Atlas A3 cache_mode=2 consumes explicit flat state_locs.
    if _is_npu:
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_memory_pool import (
            DSV4NPUTokenToKVPool,
        )

        pool_cls = DSV4NPUTokenToKVPool
        kv_layout_kwargs = {}
    else:
        pool_cls = DeepSeekV4TokenToKVPool
        kv_layout, compressed_kv_layout = select_dsv4_kv_layout()
        kv_layout_kwargs = dict(
            kv_layout=kv_layout, compressed_kv_layout=compressed_kv_layout
        )
        if (
            cp_cache_layer_split_enabled()
            and not self.is_draft_worker
        ):
            from hcu_sglang.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
                CpCacheLayerSplitDeepSeekV4TokenToKVPool,
            )

            pool_cls = CpCacheLayerSplitDeepSeekV4TokenToKVPool
            kv_layout_kwargs.update(
                cp_rank=get_parallel().attn_cp_rank,
                cp_size=get_parallel().attn_cp_size,
            )

    unified_fp8 = unified_fp8_for_dsv4_pool(
        is_draft_worker=self.is_draft_worker,
        spec_algorithm=self.spec_algorithm,
    )

    token_to_kv_pool = pool_cls(
        max_num_reqs=max_running_requests,
        # SWA ring is indexed by req_pool_idx; PD decode inflates req_to_token
        # past max_running_requests (pre-alloc), so size to the real capacity.
        num_req_slots=req_to_token_pool.req_to_token.shape[0],
        swa_size=swa_max_total_num_tokens,
        c4_size=c4_max_total_num_tokens,
        c128_size=c128_max_total_num_tokens,
        c4_state_pool_size=c4_state_pool_size,
        c128_state_pool_size=c128_state_pool_size,
        page_size=self.pool_page_size,
        swa_page_size=swa_page_size,
        sliding_window=self.model_config.window_size,
        dtype=self.kv_cache_dtype,
        c4_state_dtype=c4_state_dtype,
        c128_state_dtype=c128_state_dtype,
        qk_nope_head_dim=self.model_config.qk_nope_head_dim,
        qk_rope_head_dim=self.model_config.qk_rope_head_dim,
        indexer_head_dim=self.model_config.index_head_dim,
        layer_num=self.layer_info.num_effective_layers,
        device=self.device,
        enable_memory_saver=get_exec().features.enable_memory_saver,
        compression_ratios=compression_ratios,
        start_layer=self.layer_info.start_layer,
        end_layer=self.layer_info.end_layer,
        enable_hisparse=get_memory().enable_hisparse,
        online_mtp_max_draft_tokens=(max_speculative_num_draft_tokens() or 0),
        unified_fp8=unified_fp8,
        kv_source_layers=kv_source_layers,
        full_size=full_max_total_num_tokens,
        **({"is_draft_worker": self.is_draft_worker} if not _is_npu else {}),
        **kv_layout_kwargs,
    )
    if not self.is_draft_worker and token_to_kv_pool._unified_kv:
        # The draft pool has no C4 layers and shares this req pool, so only
        # the target registers the per-slot C4 reset.
        req_to_token_pool.register_on_alloc_rows(
            token_to_kv_pool.clear_c4_req_states
        )
    return token_to_kv_pool



def around_build_dsv4_kv_pool(original, configurator, **kwargs):
    if not cp_cache_layer_split_enabled():
        return original(configurator, **kwargs)
    return _build_dsv4_kv_pool(configurator, **kwargs)


def around_apply_token_constraints(original, configurator, token_capacity):
    result = original(configurator, token_capacity)
    if cp_cache_layer_split_enabled() and not configurator.is_draft_worker:
        tensor = torch.tensor(result, dtype=torch.int64)
        torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN,
            group=get_parallel().attn_cp_group.cpu_group)
        return tensor.item()
    return result


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from hcu_sglang.mem_cache import hcu_v4_pool, hcu_v4_sizing
    for name in ("DeepSeekV4SingleKVPool", "DeepSeekV4UniformFP8KVPool",
                 "HiSparseC4DevicePool", "DeepSeekV4IndexerPool", "DeepSeekV4TokenToKVPool"):
        HookRegistry.register("sglang.srt.mem_cache.deepseek_v4_memory_pool." + name,
            getattr(hcu_v4_pool, name), HookType.REPLACE)
    HookRegistry.register("sglang.srt.mem_cache.deepseek_v4_memory_pool.select_dsv4_kv_layout",
        hcu_v4_pool.select_dsv4_kv_layout, HookType.REPLACE)
    HookRegistry.register("sglang.srt.model_executor.pool_configurator.DSV4PoolConfigurator",
        hcu_v4_sizing.DSV4PoolConfigurator, HookType.REPLACE)
    base = "sglang.srt.mem_cache.kv_cache_configurator.KVCacheConfigurator"
    HookRegistry.register(base + "._build_dsv4_kv_pool", around_build_dsv4_kv_pool, HookType.AROUND)
    HookRegistry.register(base + "._apply_token_constraints", around_apply_token_constraints, HookType.AROUND)

    from hcu_sglang.mem_cache.cp_cache_layer_split.handshake import get_dsv41_spec_layout
    HookRegistry.register("sglang.srt.disaggregation.utils.get_dsv41_spec_layout",
        get_dsv41_spec_layout, HookType.REPLACE)
