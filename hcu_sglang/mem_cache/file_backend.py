from __future__ import annotations

import logging
import os

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import HiCacheFile as _NativeHiCacheFile
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig, MetadataCache, PoolName
from hcu_sglang.utils.detection import is_hcu

logger = logging.getLogger(__name__)


class HiCacheFile(_NativeHiCacheFile):
    def __init__(
        self, storage_config: HiCacheStorageConfig, file_path: str = "/tmp/hicache"
    ):
        self.file_path = (
            envs.SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR.get()
            or (storage_config.extra_config or {}).get("file_storage_path")
            or file_path
        )

        tp_rank, tp_size, pp_rank, pp_size, model_name, is_mla_model = (
            storage_config.tp_rank,
            storage_config.tp_size,
            storage_config.pp_rank,
            storage_config.pp_size,
            storage_config.model_name,
            storage_config.is_mla_model,
        )
        attn_cp_rank = storage_config.attn_cp_rank
        attn_cp_size = storage_config.attn_cp_size
        model_name = "-".join(model_name.split("/")) if model_name else ""
        enable_pp = pp_size > 1
        shard_suffix = f"_{pp_size}_{pp_rank}" if enable_pp else ""
        # Under NSA context parallel each CP rank holds a disjoint slice of every
        # page, so give each rank its own file key to avoid a cross-rank write race.
        if attn_cp_size > 1:
            shard_suffix += f"_cp{attn_cp_rank}_{attn_cp_size}"
        # MLA KV is replicated across TP ranks and shares one file per page;
        # TP-sharded state is keyed by tp_config_suffix instead.
        self.tp_config_suffix = f"_{model_name}_{tp_rank}_{tp_size}{shard_suffix}"
        self.config_suffix = (
            f"_{model_name}{shard_suffix}" if is_mla_model else self.tp_config_suffix
        )

        if not os.path.exists(self.file_path) and tp_rank == 0 and attn_cp_rank == 0:
            os.makedirs(self.file_path)
            logger.info(f"Created HiCacheFile storage directory at {self.file_path}")

        # Metadata cache positive lookup toggle & TTL
        enable_cache_raw = None
        if storage_config.extra_config:
            enable_cache_raw = storage_config.extra_config.get("enable_metadata_cache")
        if enable_cache_raw is None:
            enable_cache_raw = (
                envs.SGLANG_HICACHE_FILE_BACKEND_ENABLE_METADATA_CACHE.get()
            )

        self.enable_metadata_cache = bool(enable_cache_raw)

        if self.enable_metadata_cache:
            ttl_raw = None
            if storage_config.extra_config:
                ttl_raw = storage_config.extra_config.get("metadata_ttl")
            if ttl_raw is None:
                ttl_raw = envs.SGLANG_HICACHE_FILE_BACKEND_METADATA_TTL.get()

            self.metadata_ttl = float(ttl_raw) if ttl_raw is not None else 5.0
            self.metadata_cache = MetadataCache(self.metadata_ttl)
            self._scan_existing_files_to_metadata_cache()
        else:
            self.metadata_cache = None

        # All LRU / size accounting and disk eviction lives in the evictor so
        # this backend stays a thin raw-bytes store. Imported lazily: the storage
        # package __init__ pulls in the backend factory, which imports this
        # module, so a top-level import here would be circular.
        from sglang.srt.mem_cache.storage.file.lru_file_evictor import LRUFileEvictor

        # Each rank creates and evicts its TP-sharded files; rank 0 also owns
        # the shared MLA files.
        owned_suffixes = (self.tp_config_suffix,)
        if is_mla_model and tp_rank == 0:
            owned_suffixes += (self.config_suffix,)
        self._evictor = LRUFileEvictor(
            self.file_path,
            owned_suffixes=owned_suffixes,
            tp_rank=tp_rank,
            extra_config=storage_config.extra_config,
            on_evict=(
                self.metadata_cache.remove if self.metadata_cache is not None else None
            ),
        )

    def _get_suffixed_key(self, key: str) -> str:
        # Mamba state is TP-sharded even when the model's MLA KV is replicated.
        if key.endswith(f".{PoolName.MAMBA.value}"):
            return key + self.tp_config_suffix
        return key + self.config_suffix

    def _scan_existing_files_to_metadata_cache(self) -> None:
        try:
            names = os.listdir(self.file_path)
        except FileNotFoundError:
            return
        for fn in names:
            if not fn.endswith(".bin"):
                continue
            stem = fn[:-4]
            # Only files belonging to this rank/model.
            if stem.endswith((self.config_suffix, self.tp_config_suffix)):
                self.metadata_cache.add(stem)



def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.mem_cache.hicache_storage.HiCacheFile", HiCacheFile, HookType.REPLACE)
