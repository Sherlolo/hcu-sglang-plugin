from __future__ import annotations
from dataclasses import dataclass
import msgspec
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType


class WindowLoadBackPlan(msgspec.Struct, frozen=True):
    """Logical coverage and the missing subranges that need new device rows."""

    start: int
    end: int
    missing_ranges: tuple[tuple[int, int], ...]

    @property
    def num_tokens(self) -> int:
        return sum(end - start for start, end in self.missing_ranges)


@dataclass(frozen=True)
class StagedPrefetchPlan:
    """One admission pass, computed before promoting the joint component match."""

    operation_id: int
    key: RadixKey
    device_prefix_len: int
    full_tokens: int
    windows: dict[ComponentType, WindowLoadBackPlan]
    mamba_slots: int = 0

    def window_tokens(self, component_type: ComponentType) -> int:
        window = self.windows.get(component_type)
        return window.num_tokens if window is not None else 0
