"""HCU single-batch-overlap enablement."""

from __future__ import annotations

from typing import Callable

from hcu_sglang.utils.detection import is_dcu


def around_enable_combine_down_gemm(original: Callable[..., bool], cls: type) -> bool:
    if is_dcu():
        from sglang.srt.layers.moe.utils import is_sbo_enabled

        return is_sbo_enabled()
    return original(cls)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.batch_overlap.single_batch_overlap."
        "SboFlags.enable_combine_down_gemm_two_stream_overlap",
        around_enable_combine_down_gemm,
        HookType.AROUND,
    )
