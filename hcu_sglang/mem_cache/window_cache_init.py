from __future__ import annotations
from sglang.srt.mem_cache import unified_radix_cache as N
from hcu_sglang.mem_cache.window_pipeline import validate_buffer_only_stack
from hcu_sglang.utils.detection import is_hcu


def create_buffer_pipeline(
    self,
    *,
    swa_component: Optional[SWAComponent],
    write_backlog_cap: int,
    max_context_len: int = 0,
    auxiliary_swa_component: Optional[SWAComponent] = None,
) -> BufferModePipeline:
    """Called on the FULL component to construct the buffer-mode pipeline."""
    from hcu_sglang.mem_cache.window_pipeline import BufferModePipeline

    return BufferModePipeline(
        cache=self.cache,
        swa_component=swa_component,
        auxiliary_swa_component=auxiliary_swa_component,
        write_backlog_cap=write_backlog_cap,
        max_context_len=max_context_len,
    )


def init_hicache(self, server_args: ServerArgs, params: CacheInitParams) -> None:
    """Initialize HiCache infrastructure."""
    self.host_memory_mode = N.get_memory().hicache_host_memory_mode
    if self.host_memory_mode == "buffer_only":
        supported = {
            N.ComponentType.FULL,
            N.ComponentType.SWA,
            N.ComponentType.AUXILIARY_SWA,
            N.ComponentType.MAMBA,
        }
        if not set(self.tree_components) <= supported:
            raise ValueError(
                "--hicache-host-memory-mode buffer_only supports only "
                "FULL/SWA/AUXILIARY_SWA/MAMBA unified trees; got components "
                f"{sorted(ct.name for ct in self.tree_components)}."
            )
    from sglang.srt.mem_cache.hybrid_cache.hybrid_pool_assembler import (
        attach_hybrid_pool_to_unified_cache,
    )

    self.load_cache_event = N.threading.Event()
    self.sidecar_pool_specs.clear()
    self.extra_metric_labels = N.get_observability().extra_metric_labels
    self._storage_attachment = N.StorageAttachment(self)

    # Parse storage config once, share with assembler and tree
    storage_backend = N.get_memory().hicache_storage_backend
    storage_extra_config = None
    storage_prefetch_threshold = 256
    prefetch_timeout_base = 1.0
    prefetch_timeout_per_ki_token = 0.25
    hicache_storage_pass_prefix_keys = False
    if storage_backend is not None:
        (
            storage_extra_config,
            storage_prefetch_threshold,
            prefetch_timeout_base,
            prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys,
        ) = N.HybridCacheController.parse_storage_backend_extra_config(
            N.get_memory().hicache_storage_backend_extra_config
        )
        if server_args.file_storage_path:
            storage_extra_config = dict(storage_extra_config or {})
            storage_extra_config.setdefault(
                "file_storage_path", server_args.file_storage_path
            )
    storage_prefetch_threshold = (
        self._storage_attachment.resolve_prefetch_threshold(
            storage_prefetch_threshold
        )
    )

    attach_hybrid_pool_to_unified_cache(
        self,
        params,
        server_args,
        load_cache_event=self.load_cache_event,
        storage_backend=storage_backend,
        storage_extra_config=storage_extra_config,
        storage_prefetch_threshold=storage_prefetch_threshold,
    )
    # Tag HiCache enablement on the TreeCore.
    if self.cache_controller is not None:
        self.tree_core.set_hicache_enabled()
        if self.supports_swa():
            swa = self.components[N.ComponentType.SWA]
            self.tree_core.has_swa_host_pool = swa._swa_kv_pool_host is not None

    if self.host_memory_mode == "buffer_only":
        self.tree_core.set_host_memory_buffer_only()
        swa = self.components.get(N.ComponentType.SWA)
        auxiliary_swa = self.components.get(N.ComponentType.AUXILIARY_SWA)
        validate_buffer_only_stack(
            sidecar_pool_specs=self.sidecar_pool_specs,
            host_pool_group=self.host_pool_group,
            swa_component=swa,
            auxiliary_swa_component=auxiliary_swa,
            storage_prefetch_threshold=storage_prefetch_threshold,
        )
        self.buffer_pipeline = self.components[
            N.ComponentType.FULL
        ].create_buffer_pipeline(
            max_context_len=N.get_model().context_length or 0,
            swa_component=swa,
            auxiliary_swa_component=auxiliary_swa,
            # Leak backstop only: the queue is charged once per unique node
            # span across pools; a cap binding on live content would
            # drop-newest and punch storage holes.
            write_backlog_cap=2 * self.token_to_kv_pool_allocator.size_full,
        )
    # State initialization
    if self.buffer_pipeline is not None:
        self.cache_controller.host_write_staged_tokens_fn = lambda: (
            self.buffer_pipeline.write_staged_tokens_
        )
        # Pools a component's backups stage through besides the stock
        # ones (an entry may alias the anchor's host pool).
        for component in self.components.values():
            for entry in component.buffer_mode_host_pool_entries(
                self.host_pool_group
            ):
                self.cache_controller.register_host_pool_entry(entry)
    self.write_through_threshold = (
        1 if N.get_memory().hicache_write_policy == "write_through" else 2
    )
    self.is_write_back = (
        self.cache_controller is not None
        and self.cache_controller.write_policy == "write_back"
    )
    # Pre-seed the logical dropped-tokens series.
    if self.metrics_collector is not None and self.cache_controller is not None:
        reasons = ["host_pressure"]
        if self._tracks_write_through_unbacked_evictions():
            reasons.append("write_through_unbacked_eviction")
        for reason in reasons:
            self.metrics_collector.increment_dropped_tokens(
                num_tokens=0,
                reason=reason,
                pool=N.PoolName.KV.value,
            )
    self.load_back_threshold = 10
    self.prefetch_stop_policy = N.get_memory().hicache_storage_prefetch_policy

    # Runtime attach/detach of the L3 backend (startup, admin API, atexit).
    N.atexit.register(self.shutdown)

    if storage_backend is not None:
        self._storage_attachment.apply_runtime_config(
            storage_backend=storage_backend,
            prefetch_threshold=storage_prefetch_threshold,
            prefetch_timeout_base=prefetch_timeout_base,
            prefetch_timeout_per_ki_token=prefetch_timeout_per_ki_token,
            hicache_storage_pass_prefix_keys=hicache_storage_pass_prefix_keys,
            enable_storage=self.cache_controller.enable_storage,
            enable_storage_metrics=self._enable_metrics_flag,
            extra_metric_labels=self.extra_metric_labels,
        )


def around_create(original, self, *args, **kwargs):
    if not is_hcu():
        return original(self, *args, **kwargs)
    return create_buffer_pipeline(self, *args, **kwargs)


def around_init(original, self, *args, **kwargs):
    if not is_hcu():
        return original(self, *args, **kwargs)
    return init_hicache(self, *args, **kwargs)


def register_hooks():
    from hcu_sglang.mem_cache import cache_retention
    cache_retention.register_hooks()
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.mem_cache.unified_cache.components.base.TreeComponent.create_buffer_pipeline", around_create, HookType.AROUND)
    HookRegistry.register("sglang.srt.mem_cache.unified_radix_cache.UnifiedRadixCache.init_hicache", around_init, HookType.AROUND)
