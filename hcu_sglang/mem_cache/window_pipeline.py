from __future__ import annotations
from dataclasses import replace
from typing import Optional
from sglang.srt.mem_cache.buffer_mode import pipeline as N
from hcu_sglang.mem_cache.window_plan import StagedPrefetchPlan, WindowLoadBackPlan
from hcu_sglang.mem_cache.window_component import adapt_window_component, extend_storage_prefix
from hcu_sglang.utils.detection import is_hcu


class BufferModePipeline(N.BufferModePipeline):
    _swa_tail_only: bool = False

    def __init__(
        self,
        cache: UnifiedRadixCache,
        swa_component: Optional[SWAComponent],
        write_backlog_cap: int,
        max_context_len: int = 0,
        *,
        auxiliary_swa_component: Optional[SWAComponent] = None,
    ):
        self._cache = cache
        self._swa_component = adapt_window_component(swa_component)
        self._auxiliary_swa_component = adapt_window_component(auxiliary_swa_component)
        self._swa_tail_only = N.envs.SGLANG_ENABLE_HICACHE_SWA_TAIL_ONLY_BACKUP.get()
        # Metadata-only pending-write backlog cap, measured in unique FULL
        # snapshot spans; beyond it new spans are dropped at admission.
        self.write_backlog_cap = write_backlog_cap
        from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

        kvcache = cache.token_to_kv_pool_allocator.get_kvcache()
        full_pool = kvcache.full_kv_pool if isinstance(kvcache, SWAKVPool) else kvcache
        # Clamp by admission headroom: pins must leave room for the largest
        # allowed request, else a queued hold can wedge admission permanently
        # (pool full, nothing retractable). No-headroom pools take no pins.
        self.anchor_lock_cap_tokens = max(
            0,
            min(
                int(N.envs.SGLANG_HICACHE_BUFFER_ANCHOR_LOCK_CAP.get() * full_pool.size),
                full_pool.size - max_context_len,
            ),
        )
        if self.anchor_lock_cap_tokens == 0:
            N.logger.warning(
                "BufferModePipeline anchor_lock_cap_tokens=0 (pool=%d, "
                "max_context_len=%d): every prefetch launches with its splice "
                "base unpinned. Shrink --context-length or grow the KV pool.",
                full_pool.size,
                max_context_len,
            )
        else:
            N.logger.info(
                "BufferModePipeline anchor_lock_cap_tokens=%d",
                self.anchor_lock_cap_tokens,
            )
        self.reset()

    def _swa_components(self):
        return (
            component
            for component in (self._swa_component, self._auxiliary_swa_component)
            if component is not None
        )

    def _shared_load_reserve_requests(self) -> list[tuple[str, int]]:
        cc = self._cache.cache_controller
        anchor = cc.mem_pool_host.anchor_entry.host_pool
        min_full_tokens = N._minimum_full_transfer_tokens(
            anchor, self._cache.prefetch_threshold
        )
        requests = [(anchor.pool_label, min_full_tokens)]
        for component in self._swa_components():
            entry = cc.mem_pool_host.entry_map[component.pool_name]
            requests.append(
                (
                    entry.host_pool.pool_label,
                    component.full_window_pages * entry.host_pool.page_size,
                )
            )
        return requests

    def _swa_tail_pages(self, pool: N.PoolName) -> int:
        """The pool's backup window; zero keeps every page."""
        if self._swa_tail_only:
            for component in self._swa_components():
                if component.pool_name == pool:
                    return component.full_window_pages
        return 0

    def _trim_swa_backup_transfer(
        self, transfer: N.PoolTransfer, keys: list[str]
    ) -> list[str]:
        """Stage and store only the node's tail window; rows follow the keys."""
        tail = self._swa_tail_pages(transfer.name)
        if not tail or len(keys) <= tail:
            return keys
        entry = self._cache.cache_controller.mem_pool_host.entry_map[transfer.name]
        page_size = entry.host_pool.page_size
        rows = transfer.device_indices
        assert rows is not None and len(rows) == len(keys) * page_size
        transfer.device_indices = rows[-tail * page_size :]
        transfer.keys = keys[-tail:]
        transfer.hit_policy = N.PoolHitPolicy.TRAILING_PAGES
        return transfer.keys

    def _backup_pool_keys(
        self, snapshot: N.BufferBackupSnapshot
    ) -> dict[N.PoolName, list[str]]:
        """Admission view of the keys a backup writes per pool: the KV chain,
        each aux component's BACKUP_HOST transfers, component overrides on
        top, the SWA tail under tail-only backups, every sidecar under its
        source's keys."""
        hash_values = snapshot.hash_values
        tree_core = self._cache.tree_core
        pool_keys: dict[N.PoolName, list[str]] = {N.PoolName.KV: list(hash_values)}
        overrides = tree_core.buffer_backup_pool_keys(snapshot.node_id, hash_values)
        for ct in self._cache.components:
            # A component that names its pools' keys spares the transfer build.
            if ct == N.ComponentType.FULL or ct in overrides:
                continue
            transfers = tree_core.build_hicache_transfers(
                ct, snapshot.node_id, N.CacheTransferPhase.BACKUP_HOST
            )
            for transfer in transfers or ():
                keys = self._transfer_keys(transfer, hash_values)
                if keys:
                    pool_keys.setdefault(transfer.name, []).extend(keys)
        for named in overrides.values():
            for pool, keys in named.items():
                if keys:
                    pool_keys[pool] = list(keys)
                else:
                    pool_keys.pop(pool, None)
        for pool, keys in pool_keys.items():
            tail = self._swa_tail_pages(pool)
            if tail and len(keys) > tail:
                pool_keys[pool] = keys[-tail:]
        for spec in self._cache.sidecar_pool_specs:
            source = pool_keys.get(spec.indices_from_pool)
            if source:
                pool_keys[spec.pool_name] = list(source)
        return pool_keys

    def _backup_oversize(self, kv_tokens: int, aux_xfers: list[N.PoolTransfer]) -> bool:
        """True if any pool's staging need exceeds that pool's write-usable
        capacity (total for KV, total minus the loads-priority margin for aux
        pools — matching ``_aux_budget_blocked``'s admission ceiling): such an
        intent could never stage and would wedge the FIFO head. Aux pools
        staged through the anchor's host pool count against the KV total."""
        cc = self._cache.cache_controller
        shared_requests = self._shared_host_requests(kv_tokens, aux_xfers)
        if shared_requests is not None:
            pool_tokens = cc.mem_pool_host.size
            max_write_units = pool_tokens - pool_tokens // 10
            if self._host_request_units(kv_tokens, aux_xfers) > max_write_units:
                return True
            return not self._shared_backup_fits(shared_requests, empty=True)
        anchor_pool = cc.mem_pool_host.anchor_entry.host_pool
        anchor_need = kv_tokens
        for t in aux_xfers:
            entry = cc.mem_pool_host.entry_map.get(t.name)
            if entry is None:
                continue
            need = len(t.keys) * entry.host_pool.page_size
            if entry.host_pool is anchor_pool:
                anchor_need += need
            elif need > entry.host_pool.size - self._aux_loads_margin(
                t.name, entry.host_pool
            ):
                return True
        return anchor_need > cc.mem_pool_host.size

    def _aux_loads_margin(self, pool_name, host_pool) -> int:
        """Aux-pool tokens reserved for loads: at least one trailing window
        (prepare_prefetch allocates its window here and a failed alloc
        forfeits the whole prefetch), plus a 10% burst absorber mirroring
        live_cap."""
        window_pages = 0
        if pool_name == N.PoolName.SWA and self._swa_component is not None:
            window_pages = self._swa_component.full_window_pages
        elif (
            pool_name == N.PoolName.AUXILIARY_SWA
            and self._auxiliary_swa_component is not None
        ):
            window_pages = self._auxiliary_swa_component.full_window_pages
        return max(window_pages * host_pool.page_size, host_pool.size // 10)

    def _launch_transfers(
        self, intent: N._UnifiedBackupIntent
    ) -> Optional[tuple[N.torch.Tensor, dict[N.PoolName, list[str]], list[N.PoolTransfer]]]:
        """(KV rows, keys per written pool, transfers to stage) of one intent,
        off the node's fresh spec; None when nothing is left to write."""
        snapshot = intent.snapshot
        hash_values = snapshot.hash_values
        device_value, comp_xfers, kv_keys = self._build_backup_transfers(
            snapshot.node_id, kv_only=intent.pool == N.PoolName.KV
        )
        transfers: list[N.PoolTransfer] = []
        if intent.pool == N.PoolName.KV:
            keys = hash_values if kv_keys is None else kv_keys
            if not keys or len(device_value) == 0:
                return None
            comp_xfers = {}
        else:
            device_value = device_value[:0]
            keys = []
            for ct, xfers in comp_xfers.items():
                kept = []
                for transfer in xfers:
                    if transfer.name != intent.pool or transfer.indices_from_pool:
                        continue
                    t_keys = self._transfer_keys(transfer, hash_values)
                    if not t_keys:
                        continue
                    t_keys = self._trim_swa_backup_transfer(transfer, t_keys)
                    if transfer.keys is None:
                        transfer.keys = t_keys
                        transfer.hit_policy = N.PoolHitPolicy.TRAILING_PAGES
                    kept.append(transfer)
                    keys.extend(t_keys)
                if kept:
                    transfers.extend(kept)
                    comp_xfers[ct] = kept
            if not transfers:
                return None
            comp_xfers = {ct: xfers for ct, xfers in comp_xfers.items() if xfers}
        keys_by_pool = {intent.pool: list(keys)}
        # Sidecars ride their source pool's slots and D2H operation; they
        # allocate no staging of their own.
        for sidecar in self._cache._build_backup_sidecar(device_value, comp_xfers):
            if sidecar.indices_from_pool != intent.pool:
                continue
            sidecar.keys = list(keys)
            keys_by_pool[sidecar.name] = list(keys)
            transfers.append(sidecar)
        return device_value, keys_by_pool, transfers

    def _aux_budget_blocked(self, kv_tokens: int, aux: list[N.PoolTransfer]) -> bool:
        """True when an aux pool cannot stage this intent right now (free
        minus the loads-priority margin falls short of the need): defer at
        the gate instead of failing the alloc inside cc.write and blocking
        pure-KV intents behind an unallocatable head. The margin enforces
        loads-have-priority on aux pools the way live_cap does on the KV
        pool; avail already reflects prefetch-held slots, so no occupancy
        subtraction here. Aux pools staged through the anchor's host pool
        are gated by live_cap and the allocation itself."""
        cc = self._cache.cache_controller
        shared_requests = self._shared_host_requests(kv_tokens, aux)
        if shared_requests is not None:
            return not self._shared_backup_fits(shared_requests)
        anchor_pool = cc.mem_pool_host.anchor_entry.host_pool
        for t in aux:
            entry = cc.mem_pool_host.entry_map.get(t.name)
            if entry is None or entry.host_pool is anchor_pool:
                continue
            need = len(t.keys) * entry.host_pool.page_size
            headroom = entry.host_pool.available_size() - self._aux_loads_margin(
                t.name, entry.host_pool
            )
            if need > headroom:
                return True
        return False

    def finish_backup_ack(self, ack_id: int) -> None:
        """D2H confirmed: drop the device lock and enqueue the storage write
        (which reads from the staging copy, so device eviction may proceed)."""
        entry = self.ongoing_write_through.pop(ack_id)
        intent = entry.intent
        snapshot = intent.snapshot
        self._cache.dec_lock_ref(snapshot.node_id, entry.lock_params)

        storage_xfers = self._storage_transfers(entry.aux_xfers)
        # Another pool's write carries no KV pages; its keys ride the transfer.
        hash_value = (
            list(entry.keys_by_pool[N.PoolName.KV]) if intent.pool == N.PoolName.KV else []
        )
        prefix_keys = snapshot.prefix_keys
        if intent.pool in (N.PoolName.SWA, N.PoolName.AUXILIARY_SWA):
            keys = entry.keys_by_pool[intent.pool]
            skipped_pages = len(snapshot.hash_values) - len(keys)
            if skipped_pages > 0:
                # A pool-only suffix starts after the omitted node pages.
                prefix_keys = extend_storage_prefix(
                    self._cache.cache_controller, prefix_keys, snapshot.hash_values[:skipped_pages]
                )
        operation_id = self._cache.cache_controller.write_storage(
            entry.host_indices,
            snapshot.key.token_ids,
            hash_value,
            prefix_keys,
            extra_pools=storage_xfers or None,
        )
        self.ongoing_backup[operation_id] = entry

    def prepare_staged_prefetch(self, req: Req) -> bool:
        """Rebuild the admission plan from this pass's joint component match."""
        req.staged_prefetch_plan = None
        f = self.staged_prefetches.get(req.cache_request_handle)
        if f is None:
            if not (req.host_hit_is_storage and req.host_loaded_length > 0):
                self._clear_storage_hit(req)
            return True
        joint_len = req.prefix_len
        if joint_len >= f.matched_len + f.num_tokens:
            # The joint match already covers the staged span; a shorter FULL-only
            # prefix would strand the slots recomputed below cache_protected_len.
            self._resolve_device_covered(req)
            return True
        key = N.RadixKey(
            f.key_tokens,
            extra_key=f.extra_key,
            is_bigram=self._cache.tree_core.is_eagle,
            cache_salt=f.cache_salt,
        )
        matched_len, node_id, _ = self._cache.tree_core.match_full_device_prefix(key)
        if matched_len < f.matched_len:
            N.logger.warning(
                "HiCache staged prefetch deferred req=%s reason=shrunk "
                "matched=%d now=%d tokens=%d",
                req.rid,
                f.matched_len,
                matched_len,
                f.num_tokens,
            )
            self._refetch_staged(f)
            return False
        self._set_request_device_prefix(req, key, matched_len, node_id)
        plan = self._build_staged_prefetch_plan(f, key, matched_len)
        if plan is None:
            self._resolve_device_covered(req)
            return True
        req.host_hit_length = plan.full_tokens
        req.swa_host_hit_length = plan.window_tokens(N.ComponentType.SWA)
        req.mamba_host_hit_length = plan.mamba_slots
        # The device alone serves only this pass's joint match; the rest of the
        # span, fetched FULL or resident FULL the aux tail unlocks, is storage's.
        req.storage_hit_length = f.matched_len + f.num_tokens - joint_len
        req.storage_hit_start = joint_len
        req.host_hit_is_storage = True
        req.staged_prefetch_plan = plan
        return True

    def _build_staged_prefetch_plan(
        self, f: N._StagedPrefetch, key: N.RadixKey, matched_len: int
    ) -> Optional[StagedPrefetchPlan]:
        full_tokens = max(0, f.matched_len + f.num_tokens - matched_len)
        span_end = f.matched_len + f.num_tokens
        windows = {}
        for component in self._swa_components():
            staged_tokens = sum(
                len(t.host_indices)
                for t in f.aux_xfers
                if t.name == component.pool_name and t.host_indices is not None
            )
            window_start = span_end - staged_tokens
            repair_end = min(matched_len, span_end)
            missing_ranges = (
                self._cache.tree_core.swa_tombstone_ranges(
                    key,
                    window_start,
                    repair_end,
                    component_type=component.component_type,
                )
                if window_start < repair_end
                else []
            )
            tail_start = max(matched_len, window_start)
            if tail_start < span_end:
                missing_ranges.append((tail_start, span_end))
            windows[component.component_type] = WindowLoadBackPlan(
                window_start, span_end, tuple(missing_ranges)
            )
        mamba_slots = sum(
            len(t.host_indices)
            for t in f.aux_xfers
            if t.name == N.PoolName.MAMBA and t.host_indices is not None
        )
        if (
            full_tokens == 0
            and mamba_slots == 0
            and not any(w.num_tokens for w in windows.values())
        ):
            # The device already covers every pool's remaining demand.
            return None
        return StagedPrefetchPlan(
            f.operation_id, key, matched_len, full_tokens, windows, mamba_slots
        )

    def init_load_back(
        self, params: N.InitLoadBackParams
    ) -> Optional[tuple[int, NodeId]]:
        """Materialize a selected prefill under the caller's prefix lock.

        The caller has finished selecting its prefill shape and must acquire
        the request lock after success, without further admission gates. The
        prefix lock protects allocation-time eviction. None retains staging
        and its anchor for the next admission attempt.

        cc.load queues H2D before insert. Insert must preserve all FULL
        destinations queued for H2D; freeing them during dedup would let the
        pending copy overwrite reused slots. Free redundant auxiliary
        destinations only after the transfer ack."""
        cache = self._cache
        req = params.req
        assert req is not None
        request = req.cache_request_handle
        unchanged = (0, req.last_node)
        f = self.staged_prefetches.get(request)
        if f is None:
            self.release_anchor_lock(request)
            return unchanged
        plan = req.staged_prefetch_plan
        assert plan is not None, f"staged prefetch was not planned for {req.rid}"
        assert f.operation_id == plan.operation_id
        assert (f.extra_key, f.cache_salt) == (req.extra_key, req.cache_salt)
        assert (
            req.host_hit_length,
            req.swa_host_hit_length,
            req.mamba_host_hit_length,
        ) == (
            plan.full_tokens,
            plan.window_tokens(N.ComponentType.SWA),
            plan.mamba_slots,
        ), f"staged load-back budget changed for {req.rid}"

        splice_base = plan.device_prefix_len
        assert req.prefix_len == splice_base
        prefix_indices = self._cache.prefix_device_indices(req)
        trim_tokens = splice_base - f.matched_len
        assert trim_tokens % cache.page_size == 0, (
            f"staged splice trim not page-aligned req={req.rid}: "
            f"matched={f.matched_len} splice_base={splice_base}"
        )

        key = plan.key
        span_end = f.matched_len + f.num_tokens
        load_back_id = -(f.operation_id) - 1
        # Preserve the complete bounce in f until ack; H2D uses only the rows
        # admitted by the plan, including repairs inside the resident FULL prefix.
        load_xfers = list(f.aux_xfers)
        for component in self._swa_components():
            pool_name = component.pool_name
            window = plan.windows[component.component_type]
            if not window.missing_ranges:
                load_xfers = [
                    t
                    for t in load_xfers
                    if t.name != pool_name and t.indices_from_pool != pool_name
                ]
                continue
            host_parts = [
                slice(start - window.start, end - window.start)
                for start, end in window.missing_ranges
            ]
            for i, transfer in enumerate(load_xfers):
                if transfer.name != pool_name:
                    continue
                load_xfers[i] = replace(
                    transfer,
                    host_indices=N.torch.cat(
                        [transfer.host_indices[part] for part in host_parts]
                    ),
                )

        state, failure_pool = self._allocate_and_queue_load_back(
            req, f, plan, load_back_id, load_xfers
        )
        if state is None:
            # load() allocates all pools atomically before queueing H2D, so the
            # staged host buffers remain reusable after either pool is short.
            return self.defer_staged_admission(req, failure_pool)
        device_indices = state.device_indices
        del self.staged_prefetches[request]
        self._staged_admission_defers.pop(request, None)
        req.staged_prefetch_plan = None
        cache._settle_storage_prefetch_hit(
            request, credited_tokens=req.storage_hit_length
        )

        full_indices = N.torch.cat([prefix_indices, device_indices])
        for component in self._swa_components():
            pool_name = component.pool_name
            window_dev = next(
                (
                    t.device_indices
                    for t in state.load_transfers
                    if t.name == pool_name
                    and t.device_indices is not None
                    and t.device_indices.numel() > 0
                ),
                None,
            )
            if window_dev is None:
                continue
            ct = component.component_type
            window = plan.windows[ct]
            assert len(window_dev) == window.num_tokens
            full_parts = []
            offset = 0
            for start, end in window.missing_ranges:
                full_part = full_indices[start:end]
                full_parts.append(full_part)
                if end <= splice_base:
                    for action in cache.tree_core.attach_swa_window(
                        key,
                        start,
                        end,
                        window_dev[offset : offset + end - start],
                        component_type=ct,
                    ):
                        cache._apply_cache_action(action)
                offset += end - start
            component.restore_buffer_mapping(N.torch.cat(full_parts), window_dev)

        insert_result = self._insert_loaded_prefix(
            state, req, plan, prefix_indices=prefix_indices
        )
        self.ongoing_buffer_load_back[load_back_id] = state
        match = cache.match_prefix(N.MatchPrefixParams(key=key))
        canonical = cache.path_device_indices(match.last_device_node)[
            splice_base:span_end
        ]
        self.release_anchor_lock(request)
        owned = match.device_prefix_len >= span_end and N.torch.equal(
            canonical, device_indices
        )
        if not owned:
            # Fail-stop: the insert freed or replaced slots the in-flight H2D
            # still targets; continuing risks silent KV corruption.
            raise RuntimeError(
                "HiCache buffer load-back ownership violation "
                f"req={f.request.rid}: "
                f"insert prefix_len={insert_result.prefix_len} "
                f"expected={splice_base}, matched={match.device_prefix_len} "
                f"span_end={span_end} splice_base={splice_base}; "
                f"in-flight H2D targets freed slots"
            )
        return len(canonical), match.last_device_node

    def _insert_loaded_prefix(
        self,
        state: N._OngoingBufferLoadBack,
        req: Req,
        plan: StagedPrefetchPlan,
        *,
        prefix_indices: N.torch.Tensor,
    ) -> N.InsertResult:
        # Publish via a plain insert under the admission lock choreography;
        # the caller's request lock then pins the span (load_back pattern).
        # prev_prefix_len covers the already-device-resident head.
        cache = self._cache
        insert_fields = {}
        for load_context in state.load_contexts:
            insert_fields.update(load_context.get_insert_fields())
        insert_result = cache.insert(
            N.InsertParams(
                key=plan.key,
                value=N.torch.cat([prefix_indices, state.device_indices]),
                prev_prefix_len=plan.device_prefix_len,
                component_evicted_seqlens={
                    ct: window.start for ct, window in plan.windows.items()
                },
                **insert_fields,
            )
        )
        for load_context in state.load_contexts:
            state.aux_device_releases.extend(
                load_context.get_redundant_device_slots(insert_result)
            )
        return insert_result

    def _free_completed_load_back_buffers(self, f: N._OngoingBufferLoadBack) -> None:
        cc = self._cache.cache_controller
        # The H2D consumed the bounce buffers; free them outright.
        self._free_staging_now(f.host_indices, f.aux_xfers)
        for pool_name, device_indices in f.aux_device_releases:
            entry = cc.mem_pool_host.entry_map[pool_name]
            free_fn = entry.device_free_fn or entry.device_pool.free
            free_fn(device_indices)


def validate_buffer_only_stack(
    sidecar_pool_specs: list[N.SidecarPoolSpec],
    host_pool_group: HostPoolGroup,
    swa_component: Optional[SWAComponent],
    storage_prefetch_threshold: int = 256,
    *,
    auxiliary_swa_component: Optional[SWAComponent] = None,
) -> None:
    """Post-assembly buffer-mode fences.

    Sidecars reuse their source pool's transient slot ids, so every sidecar
    host pool must expose the full source slot namespace.  unified_kv SWA
    (device-only ring, never offloaded) still has no staging path.
    """
    entry_map = host_pool_group.entry_map
    for spec in sidecar_pool_specs:
        source = entry_map.get(spec.indices_from_pool)
        sidecar = entry_map.get(spec.pool_name)
        if source is None or sidecar is None:
            raise ValueError(
                "--hicache-host-memory-mode buffer_only sidecar pool mapping "
                f"is incomplete: pool={spec.pool_name}, "
                f"indices_from_pool={spec.indices_from_pool}."
            )
        source_size = source.host_pool.logical_size
        sidecar_size = sidecar.host_pool.logical_size
        if sidecar_size < source_size:
            raise ValueError(
                "--hicache-host-memory-mode buffer_only sidecar host pool is "
                "smaller than its index source: "
                f"pool={spec.pool_name}, host_slots={sidecar_size}, "
                f"source={spec.indices_from_pool}, source_slots={source_size}."
            )
    window_pools = []
    for component in (swa_component, auxiliary_swa_component):
        if component is None:
            continue
        pool = component._swa_kv_pool_host
        if pool is None:
            raise ValueError(
                "--hicache-host-memory-mode buffer_only on SWA models "
                "requires an SWA host staging pool; the unified_kv layout "
                "keeps SWA as a device-only ring."
            )
        window_pools.append((pool, component.full_window_pages * pool.page_size))

    for pool, window_tokens in window_pools:
        # Every window needs a write bounce plus a loads-priority reserve.
        # For a shared arena the reserve must hold all windows together.
        shared_domain = (
            pool.shared_allocation_domain if isinstance(pool, N.HostKVCache) else None
        )
        if shared_domain is not None:
            full_host_pool = host_pool_group.anchor_entry.host_pool
            min_full_tokens = N._minimum_full_transfer_tokens(
                full_host_pool, storage_prefetch_threshold
            )
            shared_pools = [
                (host, tokens)
                for host, tokens in [(full_host_pool, min_full_tokens), *window_pools]
                if isinstance(host, N.HostKVCache)
                and host.shared_allocation_domain is shared_domain
            ]
            one_transfer_bytes = sum(
                tokens * host.size_per_token for host, tokens in shared_pools
            )
            one_transfer = [(host.pool_label, tokens) for host, tokens in shared_pools]
            enough_capacity = shared_domain.can_fit_many_then(
                one_transfer, one_transfer, empty=True
            )
            capacity = f"{shared_domain.capacity_bytes} shared bytes"
            requirement = f"{2 * one_transfer_bytes} shared bytes"
        else:
            enough_capacity = pool.size >= 2 * window_tokens
            capacity = f"{pool.size} SWA tokens"
            requirement = f"{2 * window_tokens} SWA tokens"
        if not enough_capacity:
            raise ValueError(
                "--hicache-host-memory-mode buffer_only requires a host arena "
                "large enough for two minimum Full/SWA transfers "
                f"({requirement}; got {capacity}): one staging a write "
                "while one stays reserved for prefetch window allocs."
            )


def around_validate_stack(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return validate_buffer_only_stack(*args, **kwargs)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.mem_cache.buffer_mode.pipeline.BufferModePipeline", BufferModePipeline, HookType.REPLACE)
    HookRegistry.register("sglang.srt.mem_cache.buffer_mode.pipeline.validate_buffer_only_stack", around_validate_stack, HookType.AROUND)
