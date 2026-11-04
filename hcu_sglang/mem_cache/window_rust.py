from __future__ import annotations
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
import torch
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import CacheAction, ComponentAction
from functools import wraps
from types import MethodType
from hcu_sglang.utils.detection import is_hcu
from sglang.srt.mem_cache.unified_cache import tree_core_registry

_NATIVE_FACTORY = [tree_core_registry._rust_tree_core_factory]


def swa_tombstone_ranges(
    self,
    key: RadixKey,
    start: int,
    end: int,
    component_type: ComponentType = ComponentType.SWA,
) -> list[tuple[int, int]]:
    if component_type != ComponentType.SWA:
        raise NotImplementedError("The Rust tree core supports one SWA window")
    from sglang.srt.mem_cache.rust_tree_core import adapter as N
    return self._binding.swa_tombstone_ranges(
        self._bindings.MatchParamsBinding(
            key=N._radix_key_buffer(key),
            extra_key=key.extra_key,
            cache_salt=key.cache_salt,
        ),
        start,
        end,
    )


def attach_swa_window(
    self,
    key: RadixKey,
    window_start: int,
    window_end: int,
    swa_values: torch.Tensor,
    component_type: ComponentType = ComponentType.SWA,
) -> list[CacheAction | ComponentAction]:
    if component_type != ComponentType.SWA:
        raise NotImplementedError("The Rust tree core supports one SWA window")
    from sglang.srt.mem_cache.rust_tree_core import adapter as N
    return N._cache_actions_from_tagged(
        self._binding.attach_swa_window(
            self._bindings.MatchParamsBinding(
                key=N._radix_key_buffer(key),
                extra_key=key.extra_key,
                cache_salt=key.cache_salt,
            ),
            window_start,
            window_end,
            swa_values,
        )
    )



def around_get_factory(original, name):
    factory = original(name)
    if not is_hcu() or name != "rust" or factory is not _NATIVE_FACTORY[0]:
        return factory

    @wraps(factory)
    def create(params, components):
        core = factory(params, components)
        from sglang.srt.mem_cache.rust_tree_core import adapter
        if type(core) is adapter.RustUnifiedTreeCore:
            core.swa_tombstone_ranges = MethodType(swa_tombstone_ranges, core)
            core.attach_swa_window = MethodType(attach_swa_window, core)
            from hcu_sglang.mem_cache.rust_tree_core.provider import attach_event_recorder
            attach_event_recorder(core, params)
        return core

    return create


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.mem_cache.unified_cache.tree_core_registry.get_tree_core_factory", around_get_factory, HookType.AROUND)
