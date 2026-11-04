from __future__ import annotations
from sglang.srt.mem_cache.unified_cache.cache_action import RebuildFullToSWAMapping
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.hicache_storage import PoolName


def restore_buffer_mapping(self, full_indices, window_indices):
    self.cache._apply_cache_action(
        RebuildFullToSWAMapping([full_indices], [window_indices])
    )


class _NativeSWAView:
    pool_name = PoolName.SWA

    def __init__(self, component):
        self._component = component

    def __getattr__(self, name):
        return getattr(self._component, name)

    def restore_buffer_mapping(self, full_indices, window_indices):
        callback = getattr(self._component, "restore_buffer_mapping", None)
        if callback is not None:
            return callback(full_indices, window_indices)
        return restore_buffer_mapping(self._component, full_indices, window_indices)


def adapt_window_component(component):
    if component is None:
        return None
    if hasattr(component, "pool_name") and hasattr(component, "restore_buffer_mapping"):
        return component
    if component.component_type == ComponentType.SWA:
        return _NativeSWAView(component)
    return component


def source_extend_storage_prefix(self, prefix_keys, hash_values):
    """Advance the backend's prefix context past omitted pages."""
    return prefix_keys + hash_values if prefix_keys is not None else None


def extend_storage_prefix(controller, prefix_keys, hash_values):
    callback = getattr(controller, "extend_storage_prefix", None)
    if callback is not None:
        return callback(prefix_keys, hash_values)
    return source_extend_storage_prefix(controller, prefix_keys, hash_values)
