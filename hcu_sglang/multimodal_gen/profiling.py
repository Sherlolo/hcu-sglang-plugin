"""Diffusion profiler start-step compatibility."""

from __future__ import annotations

import contextlib
import dataclasses
import os
from types import MappingProxyType
from typing import Any, Callable

import torch


def patch_sampling_params_field() -> None:
    from hcu_sglang.compat.aiter import install_aiter_compat

    install_aiter_compat()
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams

    if "profile_start_step" in SamplingParams.__dataclass_fields__:
        return
    profile_start_step = dataclasses.field(
        default=0, metadata={"batch_sig_exclude": True}
    )
    profile_start_step.name = "profile_start_step"
    profile_start_step.type = int
    profile_start_step._field_type = dataclasses._FIELD
    profile_start_step.metadata = MappingProxyType({"batch_sig_exclude": True})
    SamplingParams.__dataclass_fields__["profile_start_step"] = profile_start_step
    SamplingParams.profile_start_step = 0


def after_add_cli_args(result: Any, parser: Any) -> Any:
    if not any(
        "--profile-start-step" in action.option_strings
        for action in parser._actions
    ):
        parser.add_argument(
            "--profile-start-step",
            type=int,
            default=None,
            dest="profile_start_step",
            help=(
                "Step index at which to start profiling. The warmup step runs "
                "at this index and trace capture starts one step later."
            ),
        )
    return result


def around_from_user_sampling_params(
    original: Callable[..., Any], model_path: str, server_args: Any, *args, **kwargs
) -> Any:
    explicit = "profile_start_step" in kwargs
    start_step = kwargs.pop("profile_start_step", 0)
    sampling_params = original(model_path, server_args, *args, **kwargs)
    sampling_params.profile_start_step = 0 if start_step is None else int(start_step)
    if sampling_params.profile_start_step < 0:
        raise ValueError("profile_start_step must be non-negative")
    if explicit and hasattr(sampling_params, "_explicit_fields"):
        sampling_params._explicit_fields.add("profile_start_step")
    return sampling_params


def around_from_pretrained(
    original: Callable[..., Any], cls, model_path: str, **kwargs
) -> Any:
    start_step = kwargs.pop("profile_start_step", 0)
    sampling_params = original(cls, model_path, **kwargs)
    sampling_params.profile_start_step = 0 if start_step is None else int(start_step)
    if sampling_params.profile_start_step < 0:
        raise ValueError("profile_start_step must be non-negative")
    return sampling_params


def _create_profiler_with_start(
    request_id: str,
    rank: int,
    full_profile: bool,
    num_steps: int,
    num_inference_steps: int,
    start_step: int,
):
    from sglang.multimodal_gen.runtime.platforms import current_platform
    from sglang.multimodal_gen.runtime.utils import profiler as profiler_module

    profiler_cls = profiler_module.SGLDiffusionProfiler
    profiler = profiler_cls.__new__(profiler_cls)
    profiler.request_id = request_id or "profile_trace"
    profiler.rank = rank
    profiler.full_profile = full_profile
    profiler.log_dir = profiler_module._resolve_profiler_log_dir(None)
    try:
        os.makedirs(profiler.log_dir, exist_ok=True)
    except OSError:
        pass

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available() or (
        hasattr(torch, "musa") and torch.musa.is_available()
    ):
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    if current_platform.is_npu():
        activities.append(profiler_module.torch_npu.profiler.ProfilerActivity.NPU)
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        activities.append(torch.profiler.ProfilerActivity.XPU)

    common_args = dict(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        on_trace_ready=(
            None
            if not current_platform.is_npu()
            else profiler_module.torch_npu.profiler.tensorboard_trace_handler(
                profiler.log_dir
            )
        ),
    )
    if full_profile:
        profiler.profiler = torch.profiler.profile(**common_args)
        profiler.profile_mode_id = "full stages"
    else:
        warmup = 1
        num_actual_steps = num_inference_steps if num_steps == -1 else num_steps
        profiler.num_active_steps = start_step + num_actual_steps + warmup
        profiler.profiler = torch.profiler.profile(
            **common_args,
            schedule=torch.profiler.schedule(
                skip_first=0,
                wait=start_step,
                warmup=warmup,
                active=num_actual_steps + warmup,
                repeat=1,
            ),
        )
        profiler.profile_mode_id = (
            f"{num_actual_steps} steps from step {start_step}"
            if start_step > 0
            else f"{num_actual_steps} steps"
        )
    profiler.has_stopped = False
    profiler_cls._instance = profiler
    profiler.start()
    return profiler


def around_profile_execution(
    original: Callable[..., Any], executor, batch, dump_rank: int = 0
):
    start_step = int(getattr(batch, "profile_start_step", 0) or 0)
    if start_step <= 0 or not batch.profile or batch.is_warmup:
        return original(executor, batch, dump_rank=dump_rank)

    @contextlib.contextmanager
    def profile_context():
        from sglang.multimodal_gen.runtime.distributed import get_world_rank

        profiler = _create_profiler_with_start(
            request_id=batch.request_id,
            rank=get_world_rank(),
            full_profile=batch.profile_all_stages,
            num_steps=batch.num_profiled_timesteps,
            num_inference_steps=batch.num_inference_steps,
            start_step=start_step,
        )
        try:
            yield
        finally:
            profiler.stop(dump_rank=dump_rank)

    return profile_context()


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    patch_sampling_params_field()
    target = (
        "sglang.multimodal_gen.configs.sample.sampling_params.SamplingParams"
    )
    HookRegistry.register(
        f"{target}.add_cli_args", after_add_cli_args, HookType.AFTER
    )
    HookRegistry.register(
        f"{target}.from_user_sampling_params_args",
        around_from_user_sampling_params,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{target}.from_pretrained", around_from_pretrained, HookType.AROUND
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.pipelines_core.executors."
        "pipeline_executor.PipelineExecutor.profile_execution",
        around_profile_execution,
        HookType.AROUND,
    )
