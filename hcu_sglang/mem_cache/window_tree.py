from __future__ import annotations
from sglang.srt.mem_cache.unified_cache import unified_tree_core as N
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeNode
from sglang.srt.mem_cache.unified_cache.cache_action import CacheAction, ComponentAction
from hcu_sglang.utils.detection import is_hcu


def swa_tombstone_ranges(
    self,
    key: RadixKey,
    start: int,
    end: int,
    component_type: N.ComponentType = N.ComponentType.SWA,
) -> list[tuple[int, int]]:
    """Return the maximal SWA-tombstoned ranges within ``[start, end)``."""
    ranges: list[tuple[int, int]] = []
    for child, pos, prefix_len in self._walk_span(key, end):
        seg_end = pos + prefix_len
        if seg_end <= start:
            continue
        if child.component_data[component_type].value is not None:
            continue
        lo, hi = max(start, pos), min(end, seg_end)
        if ranges and ranges[-1][1] == lo:
            ranges[-1] = (ranges[-1][0], hi)
        else:
            ranges.append((lo, hi))
        if hi >= end:
            break
    return ranges


def attach_swa_window(
    self,
    key: RadixKey,
    window_start: int,
    window_end: int,
    swa_values: N.torch.Tensor,
    component_type: N.ComponentType = N.ComponentType.SWA,
) -> list[CacheAction | ComponentAction]:
    """Attach a loaded SWA slice to an existing tombstoned tree span."""
    assert len(swa_values) == window_end - window_start, (
        f"attach_swa_window size mismatch: got {len(swa_values)} "
        f"for [{window_start}, {window_end})"
    )
    actions: list[CacheAction | ComponentAction] = []
    covered = window_start
    for child, pos, prefix_len in list(self._walk_span(key, window_end)):
        seg_end = pos + prefix_len
        if seg_end <= window_start:
            continue
        seg_start = max(pos, window_start)
        assign_end = min(seg_end, window_end)
        assert child.component_data[component_type].value is None, (
            f"attach_swa_window over live SWA at [{seg_start}, {assign_end}) "
            f"of [{window_start}, {window_end})"
        )
        self._attach_swa_segment(
            child,
            pos,
            seg_start,
            assign_end,
            swa_values[seg_start - window_start : assign_end - window_start],
            actions,
            component_type,
        )
        covered = assign_end
        if covered >= window_end:
            break
    assert covered == window_end, (
        f"attach_swa_window covered {covered} of [{window_start}, {window_end})"
    )
    return actions


def _attach_swa_segment(
    self,
    node: UnifiedTreeNode,
    node_start: int,
    seg_start: int,
    seg_end: int,
    values: N.torch.Tensor,
    actions: list[CacheAction | ComponentAction],
    component_type: N.ComponentType,
) -> None:
    target = node
    if seg_start > node_start:
        _, action = self._split_node(target.key, target, seg_start - node_start)
        if action is not None:
            actions.append(action)
    if seg_start + len(target.key) > seg_end:
        fragment, action = self._split_node(target.key, target, seg_end - seg_start)
        if action is not None:
            actions.append(action)
        target = fragment
    self.set_component_device_value(target.id, component_type, values.clone())


def around_swa_tombstone_ranges(original, self, *args, **kwargs):
    if not is_hcu():
        return original(self, *args, **kwargs)
    return swa_tombstone_ranges(self, *args, **kwargs)


def around_attach_swa_window(original, self, *args, **kwargs):
    if not is_hcu():
        return original(self, *args, **kwargs)
    return attach_swa_window(self, *args, **kwargs)


def around__attach_swa_segment(original, self, *args, **kwargs):
    if not is_hcu():
        return original(self, *args, **kwargs)
    return _attach_swa_segment(self, *args, **kwargs)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.mem_cache.unified_cache.unified_tree_core.UnifiedTreeCore.swa_tombstone_ranges", around_swa_tombstone_ranges, HookType.AROUND)
    HookRegistry.register("sglang.srt.mem_cache.unified_cache.unified_tree_core.UnifiedTreeCore.attach_swa_window", around_attach_swa_window, HookType.AROUND)
    HookRegistry.register("sglang.srt.mem_cache.unified_cache.unified_tree_core.UnifiedTreeCore._attach_swa_segment", around__attach_swa_segment, HookType.AROUND)
