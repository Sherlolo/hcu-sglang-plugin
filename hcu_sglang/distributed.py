"""HCU distributed runtime adaptations."""

from __future__ import annotations

import os
from functools import partial

from hcu_sglang.utils.detection import is_dcu


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def around_dispatch_custom_allreduce(original, group, device):
    if not is_dcu():
        return original(group, device)

    from sglang.srt.distributed.device_communicators import custom_all_reduce

    if custom_all_reduce._use_amd_deterministic_impl():
        return custom_all_reduce.CustomAllreduce
    if not _env_enabled("SGLANG_USE_AITER_AR", default=False):
        return custom_all_reduce.CustomAllreduce

    try:
        from aiter.dist.device_communicators.custom_all_reduce import (
            CustomAllreduce as AiterCustomAllreduce,
        )
        from sglang.srt.environ import envs

        return partial(
            AiterCustomAllreduce,
            enable_register_for_capturing=envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get(),
        )
    except ImportError:
        return custom_all_reduce.CustomAllreduce


def around_custom_all_reduce(original, communicator, input):
    """HCU graph variants stage through the communicator-owned IPC buffer."""
    if not is_dcu():
        return original(communicator, input)
    import torch
    if communicator.disabled or not communicator.should_custom_ar(input):
        return None
    if communicator._IS_CAPTURING and torch.cuda.is_current_stream_capturing():
        return communicator._all_reduce_impl(input, registered=False)
    return original(communicator, input)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.distributed.device_communicators.custom_all_reduce."
        "CustomAllreduce.custom_all_reduce",
        around_custom_all_reduce,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.distributed.device_communicators.custom_all_reduce."
        "dispatch_custom_allreduce",
        around_dispatch_custom_allreduce,
        HookType.AROUND,
    )

