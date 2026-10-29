"""HCU QuickAllReduce compatibility."""

from __future__ import annotations

import os
from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_dcu


def around_rocm_arch_available(original: Callable[[], bool]) -> bool:
    if original():
        return True
    if not is_dcu():
        return False
    try:
        arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    except Exception:
        return False
    return any(name in arch for name in ("gfx936", "gfx938"))


def around_init_quick_all_reduce(
    original: Callable[..., Any], quick_reduce: Any
) -> Any:
    if not is_dcu() or "ROCM_QUICK_REDUCE_QUANTIZATION" in os.environ:
        return original(quick_reduce)
    os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = "INT8"
    try:
        return original(quick_reduce)
    finally:
        os.environ.pop("ROCM_QUICK_REDUCE_QUANTIZATION", None)


def around_should_quick_allreduce(
    original: Callable[..., bool], quick_reduce: Any, tensor: torch.Tensor
) -> bool:
    if not is_dcu():
        return original(quick_reduce, tensor)
    if quick_reduce.disabled or tensor.dtype not in quick_reduce._SUPPORTED_DTYPES:
        return False
    input_size = tensor.numel() * tensor.element_size()
    if input_size % 16 != 0:
        return False
    from sglang.srt.distributed.device_communicators.quick_all_reduce import (
        is_weak_contiguous,
    )

    return is_weak_contiguous(tensor) and input_size <= quick_reduce.qr_max_size


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    target = "sglang.srt.distributed.device_communicators.quick_all_reduce"
    HookRegistry.register(
        f"{target}.qr_rocm_arch_available",
        around_rocm_arch_available,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{target}.QuickAllReduce.init_quick_all_reduce",
        around_init_quick_all_reduce,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{target}.QuickAllReduce.should_quick_allreduce",
        around_should_quick_allreduce,
        HookType.AROUND,
    )
