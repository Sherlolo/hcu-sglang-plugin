from __future__ import annotations
from enum import Enum
import copyreg
from sglang.srt.mem_cache.hicache_storage import PoolName as _NativePoolName
from hcu_sglang.utils.detection import is_hcu

_LEGACY = [_NativePoolName]


class PoolName(str, Enum):
    """Well-known pool names used as PoolTransfer/PoolEntry identifiers."""

    KV = "kv"
    MAMBA = "mamba"
    SWA = "swa"
    AUXILIARY_SWA = "auxiliary_swa"
    INDEXER = "indexer"
    # TODO(hzh0425): Current DeepSeek V4 pool naming is verbose; will be normalized to
    # 'COMPRESSED_KV / COMPRESSED_INDEXER / COMPRESSED_STATE' in the next PR.
    DEEPSEEK_V4_C1 = "deepseek_v4_c1"
    DEEPSEEK_V4_C1_INDEXER = "deepseek_v4_c1_indexer"
    DEEPSEEK_V4_C1_INDEXER_SCALE = "deepseek_v4_c1_indexer_scale"
    DEEPSEEK_V4_C2 = "deepseek_v4_c2"
    DEEPSEEK_V4_C2_INDEXER = "deepseek_v4_c2_indexer"
    DEEPSEEK_V4_C2_INDEXER_SCALE = "deepseek_v4_c2_indexer_scale"
    DEEPSEEK_V4_C4 = "deepseek_v4_c4"
    DEEPSEEK_V4_C4_INDEXER = "deepseek_v4_c4_indexer"
    # FP4 indexer splits the indexer cache into separate payload/scale buffers,
    # so it needs a second pool alongside DEEPSEEK_V4_C4_INDEXER.
    DEEPSEEK_V4_C4_INDEXER_SCALE = "deepseek_v4_c4_indexer_scale"
    DEEPSEEK_V4_C128 = "deepseek_v4_c128"
    # fp8 unified_kv splits a row across a packed fp8 nope pool and a parallel
    # bf16 rope pool, so each compressed region mirrors to two host pools.
    DEEPSEEK_V4_C4_ROPE = "deepseek_v4_c4_rope"
    DEEPSEEK_V4_C128_ROPE = "deepseek_v4_c128_rope"
    DEEPSEEK_V4_C4_STATE = "deepseek_v4_c4_state"
    DEEPSEEK_V4_C4_INDEXER_STATE = "deepseek_v4_c4_indexer_state"
    DEEPSEEK_V4_C128_STATE = "deepseek_v4_c128_state"

    # Draft KV pool
    DRAFT = "draft"
    DRAFT_INDEXER = "draft_indexer"
    DRAFT_SWA = "draft_swa"

    def __str__(self) -> str:
        return self.value


def _reduce_legacy(member):
    return PoolName, (member.value,)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    copyreg.pickle(_LEGACY[0], _reduce_legacy)
    HookRegistry.register("sglang.srt.mem_cache.hicache_storage.PoolName", PoolName, HookType.REPLACE)
