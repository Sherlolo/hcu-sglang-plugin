"""HCU speculative-decoding and optional dependency compatibility."""

from __future__ import annotations

from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_dcu


def around_handle_dspark(original: Callable[..., None], server_args: Any) -> None:
    if not is_dcu() or server_args.device.startswith(("cuda", "npu")):
        return original(server_args)
    original_device = server_args.device
    server_args.device = "cuda"
    try:
        return original(server_args)
    finally:
        server_args.device = original_device


def around_inference_only(original: Callable[..., Any], *args, **kwargs):
    if not is_dcu():
        return original(*args, **kwargs)
    with torch.inference_mode():
        return original(*args, **kwargs)


def around_dspark_dequant_supported(original: Callable[..., bool], linear: Any) -> bool:
    if (
        is_dcu()
        and getattr(getattr(linear, "weight", None), "dtype", None)
        == torch.float8_e4m3fn
        and not hasattr(linear, "weight_scale_inv")
    ):
        return False
    return original(linear)


def _get_dspark_ragged_settings():
    from sglang.srt.environ import envs
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )

    return (
        read_ragged_verify_mode(),
        RaggedVerifyMode.STATIC,
        envs.SGLANG_PREP_IN_CUDA_GRAPH.get(),
    )


def before_dspark_verify_planner_init(*args, **kwargs) -> None:
    del args, kwargs
    if not is_dcu():
        return None

    ragged_verify_mode, static_mode, prep_in_cuda_graph = _get_dspark_ragged_settings()
    if ragged_verify_mode is not static_mode and not prep_in_cuda_graph:
        raise ValueError(
            f"DSpark ragged-verify mode {ragged_verify_mode.value!r} "
            f"requires SGLANG_PREP_IN_CUDA_GRAPH=1 (the captured-graph prepare "
            f"path). It is currently disabled, which would put per-step "
            f"verify_lens_cpu host reads on the critical path. Set "
            f"SGLANG_PREP_IN_CUDA_GRAPH=1 or run SGLANG_RAGGED_VERIFY_MODE=static."
        )
    return None


def _needs_mhc_hidden_states(model_runner: Any) -> bool:
    model_config = getattr(model_runner, "model_config", None)
    return getattr(model_config, "hc_hidden_size", None) is not None


def around_prepare_for_draft(original: Callable[..., Any], *args, **kwargs):
    # Official v0.5.22 returns (forward_batch, can_run_decode_cuda_graph,
    # kv_loc_plan); the das fork returned a 2-tuple. Stay arity-agnostic:
    # only the ForwardBatch is touched (in place), the rest passes through.
    result = original(*args, **kwargs)
    forward_batch = result[0]
    draft_model_runner = kwargs.get("draft_model_runner")
    if draft_model_runner is None and len(args) >= 4:
        draft_model_runner = args[3]
    if _needs_mhc_hidden_states(draft_model_runner):
        forward_batch.return_hidden_states_before_norm = True
    return result


def around_eagle_prepare_for_verify(original: Callable[..., Any], *args, **kwargs):
    # See around_prepare_for_draft: return arity differs across versions.
    result = original(*args, **kwargs)
    forward_batch = result[0]
    target_worker = kwargs.get("target_worker")
    if target_worker is None and len(args) >= 4:
        target_worker = args[3]
    model_runner = getattr(target_worker, "model_runner", None)
    if _needs_mhc_hidden_states(model_runner):
        forward_batch.return_hidden_states_before_norm = True
    return result


def around_tbo_filter_batch(
    original: Callable[..., Any], cls: type, batch: Any, *args, **kwargs
):
    child_batch = original(cls, batch, *args, **kwargs)
    if is_dcu() and getattr(batch, "return_hidden_states_before_norm", False):
        child_batch.return_hidden_states_before_norm = True
    return child_batch


def patch_broken_torchaudio_availability() -> None:
    try:
        import torchaudio  # noqa: F401
    except (ImportError, OSError):
        try:
            import transformers.utils as transformers_utils
            import transformers.utils.import_utils as import_utils
        except ImportError:
            return
        import_utils.is_torchaudio_available = lambda: False
        transformers_utils.is_torchaudio_available = lambda: False


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    patch_broken_torchaudio_availability()
    HookRegistry.register(
        "sglang.srt.arg_groups.speculative_hook._handle_dspark",
        around_handle_dspark,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.speculative.dspark_components.dspark_draft_sampler."
        "DsparkDraftSampler.stage_sampling_params",
        around_inference_only,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.speculative.dspark_components.dspark_worker_v2."
        "DSparkWorkerV2.init_cuda_graphs",
        around_inference_only,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.kernels.ops.speculative.dspark.dspark_draft_model."
        "_dequant_supported",
        around_dspark_dequant_supported,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.speculative.dspark_components.dspark_planner."
        "DSparkVerifyPlanner.__init__",
        before_dspark_verify_planner_init,
        HookType.BEFORE,
    )
    HookRegistry.register(
        "sglang.srt.speculative.eagle_worker_common.prepare_for_draft",
        around_prepare_for_draft,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.speculative.eagle_worker_common.eagle_prepare_for_verify",
        around_eagle_prepare_for_verify,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.batch_overlap.two_batch_overlap."
        "TboForwardBatchPreparer.filter_batch",
        around_tbo_filter_batch,
        HookType.AROUND,
    )
