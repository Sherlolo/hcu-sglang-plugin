"""HCU distributed runtime adaptations."""

from __future__ import annotations

from hcu_sglang.utils.detection import is_dcu


def around_dispatch_custom_allreduce(original, group, device, backend="auto"):
    if not is_dcu():
        return original(group, device)

    from hcu_sglang.distributed_aiter import dispatch_custom_allreduce

    return dispatch_custom_allreduce(group, device, backend=backend)


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

