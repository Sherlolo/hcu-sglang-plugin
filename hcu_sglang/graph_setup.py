from __future__ import annotations

from typing import Any, Optional
from sglang.srt.model_executor.model_runner_components import cuda_graph_setup as N
from sglang.srt.utils import get_bool_env_var
from hcu_sglang.utils.detection import is_hcu


def has_hybrid_graph_handlers(
    *,
    attention_layers: list[Any],
    moe_layers: list[Any],
    start_layer: int,
    end_layer: int,
) -> bool:
    """Accept pure MoE layers while rejecting gaps in the local decoder range."""
    return (
        0 <= start_layer < end_layer
        and end_layer <= len(attention_layers)
        and end_layer <= len(moe_layers)
        and all(
            attention_layers[i] is not None or moe_layers[i] is not None
            for i in range(start_layer, end_layer)
        )
    )


def index_attention_layers_by_global_id(
    attention_layers: list[Any],
    mha_companion_layers: list[Any],
    layer_model=None,
) -> tuple[list[Any], list[Any]]:
    """Pad PP-local attention metadata so global layer_id remains a valid index.

    Models that re-execute layers pre-expand these lists into position-indexed
    lookup tables (the same layer at several positions); such tables are
    returned unchanged.
    """
    if len(attention_layers) != len(mha_companion_layers):
        raise ValueError("attention and MHA companion metadata must be parallel")
    populated = [layer for layer in attention_layers if layer is not None]
    if not populated or any(not hasattr(layer, "layer_id") for layer in populated):
        if layer_model is not None:
            return (
                N._align_pipeline_layers(attention_layers, layer_model),
                N._align_pipeline_layers(mha_companion_layers, layer_model),
            )
        return attention_layers, mha_companion_layers
    max_layer_id = max(int(layer.layer_id) for layer in populated)
    # Keep trailing pure MoE layers represented as well. Their graph handler
    # lives in moe_layers, and shrinking this list loses decoder alignment.
    indexed_size = max(
        max_layer_id + 1, len(layer_model.layers) if layer_model is not None else 0
    )
    indexed_attention = [None] * indexed_size
    indexed_companions = [None] * indexed_size
    has_reused_layers = False
    for attention, companion in zip(attention_layers, mha_companion_layers):
        if attention is None:
            if companion is not None:
                raise ValueError("MHA companion has no primary attention layer")
            continue
        layer_id = int(attention.layer_id)
        if layer_id < 0:
            raise ValueError(f"invalid or duplicate attention layer_id: {layer_id}")
        if indexed_attention[layer_id] is not None:
            if (
                indexed_attention[layer_id] is not attention
                or indexed_companions[layer_id] is not companion
            ):
                raise ValueError(f"invalid or duplicate attention layer_id: {layer_id}")
            has_reused_layers = True
            continue
        indexed_attention[layer_id] = attention
        indexed_companions[layer_id] = companion
    if has_reused_layers:
        return attention_layers, mha_companion_layers
    return indexed_attention, indexed_companions


def release_hcu_flashmla_decode_h16_capture_cache(model_runner) -> bool:
    """Release unused allocator fragments after an H16 graph capture."""

    if not is_hcu() or not get_bool_env_var("SGLANG_HCU_FLASHMLA_DECODE_H16"):
        return False

    attn_backend = getattr(model_runner, "attn_backend", None)
    if not getattr(attn_backend, "_hcu_flashmla_decode_h16_enabled", False):
        return False

    N.current_platform.synchronize()
    N.current_platform.empty_cache()
    N.log_info_on_rank0(
        N.logger,
        "HCU H16 FlashMLA decode released unused allocator cache after "
        "CUDA graph capture.",
    )
    return True


def refresh_deep_gemm_layout_memory_budget(
    model_runner: ModelRunner, *, only_if_initialized: bool = False
) -> None:
    """Set the all-rank budget before capture, then refresh after startup."""

    if (
        model_runner.device != "cuda"
        or N.envs.SGLANG_DEEPGEMM_STANDARD_LAYOUT.get().lower() != "auto"
    ):
        return

    if only_if_initialized:
        if not N._deep_gemm_layout_memory_budget_initialized:
            return
        # Scale joiners refresh with all expanded ranks during recapture.
        if N.get_exec().moe.ep_join_mode == "scale":
            return
        # Target and draft share the budget. Its pre-capture initialization
        # already used a world-wide collective, so this guard is rank-uniform
        # and also covers a draft-only DeepGEMM backend outside draft context.
    else:
        if model_runner.is_draft_worker:
            moe_runner_backend = (
                N.get_spec().speculative_moe_runner_backend
                or N.get_exec().moe.moe_runner_backend
            )
            moe_a2a_backend = (
                N.get_spec().speculative_moe_a2a_backend or N.get_exec().moe.moe_a2a_backend
            )
        else:
            moe_runner_backend = N.get_exec().moe.moe_runner_backend
            moe_a2a_backend = N.get_exec().moe.moe_a2a_backend

        uses_deep_gemm_moe_runner = moe_runner_backend == "deep_gemm"
        if moe_runner_backend == "auto" and model_runner.model_config.quantization in (
            "fp8",
            "mxfp8",
        ):
            from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend
            from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod

            uses_deep_gemm_moe_runner = (
                Fp8MoEMethod.is_deepgemm_moe_runner_backend_enabled(
                    MoeRunnerBackend(moe_runner_backend),
                    MoeA2ABackend(moe_a2a_backend),
                )
            )
        if not uses_deep_gemm_moe_runner:
            return

    from sglang.srt.layers.moe.moe_runner.deep_gemm import (
        set_masked_standard_layout_memory_budget,
    )

    parallel = N.get_parallel()
    if parallel.pp_size > 1:
        # The speculative draft runner exists only on the last PP stage, so a
        # world-group all-reduce here would deadlock against the collectives
        # the other stages enter. Every rank of this TP group runs this branch.
        memory_group = parallel.tp_group
        distributed = memory_group.world_size > 1
    else:
        memory_group = parallel.world_group
        distributed = parallel.launch_world_size > 1
    available_memory_gb = N.get_available_gpu_memory(
        model_runner.device,
        model_runner.gpu_id,
        distributed=distributed,
        cpu_group=memory_group.cpu_group,
    )
    budget_bytes = set_masked_standard_layout_memory_budget(
        int(available_memory_gb * (1 << 30))
    )
    N._deep_gemm_layout_memory_budget_initialized = True
    N.logger.info(
        "DeepGEMM masked layout budget: %.2f GiB from %.2f GiB free.",
        budget_bytes / (1 << 30),
        available_memory_gb,
    )


def capture_cuda_graphs(
    *,
    model_runner: ModelRunner,
    capture_decode_cuda_graph: bool = True,
    finalize: bool = True,
    defer_distributed_setup: bool = False,
) -> N.CudaGraphsCapture:
    """Capture cuda graphs. Requires init_attention_backends() to have run.

    Spec draft runners pass capture_decode_cuda_graph=False
    because they capture their own decode-style graphs separately.

    """

    model_runner.graph_shared_output = N.GraphSharedOutput.create_for_model_runner(
        model_runner
    )

    # The eager (no-cuda-graph) phase runner, built AFTER the attention
    # backend so its __init__ can warm up kernels (run-once) and allocate the
    # fixed-max static buffer — both before the cuda-graph runners, so that
    # buffer is canonical in the shared pool and the cg runners coalesce onto
    # it. Always built: it serves both the fully-disabled case (decode/prefill
    # runners point at it) and the eager fallback when a cg runner can't run a
    # batch.
    eager_runner = N.EagerRunner(model_runner, run_warmup=not defer_distributed_setup)

    if not defer_distributed_setup:
        refresh_deep_gemm_layout_memory_budget(model_runner)

    # cuda-graph capture: prefill before decode, so both coalesce onto the
    # eager buffer allocated above. (capture_prefill_graph routes prefill
    # to the eager runner when the prefill graph is disabled.)
    prefill = capture_prefill_graph(
        model_runner=model_runner, eager_runner=eager_runner
    )

    decode_phase = "draft_decode" if model_runner.is_draft_worker else "decode"
    decode = N.GraphCapture(
        runner=None,
        memory_phase=decode_phase,
        memory_usage_gb=0,
        capture_time=0,
    )
    if capture_decode_cuda_graph:
        if N.current_platform.capabilities.graph_capture:
            decode = N.capture_decode_graph(model_runner=model_runner)
    else:
        decode = N.GraphCapture(
            runner=eager_runner,
            memory_phase=decode_phase,
            memory_usage_gb=0,
            capture_time=0,
        )

    if finalize:
        N.finalize_cuda_graph_capture(model_runner)

    if capture_decode_cuda_graph:
        release_hcu_flashmla_decode_h16_capture_cache(model_runner)

    return N.CudaGraphsCapture(eager_runner=eager_runner, prefill=prefill, decode=decode)


def capture_prefill_graph(
    *,
    model_runner: ModelRunner,
    eager_runner: N.EagerRunner,
    force_for_draft_worker: bool = False,
) -> N.GraphCapture:
    """Initialize a prefill graph and return its startup resource usage."""

    memory_phase = "draft_prefill" if model_runner.is_draft_worker else "prefill"
    role = "draft" if model_runner.is_draft_worker else "target"

    def result(
        runner: Optional[BaseRunner],
        memory_usage_gb: float = 0,
        capture_time: float = 0,
    ) -> N.GraphCapture:
        return N.GraphCapture(
            runner=runner,
            memory_phase=memory_phase,
            memory_usage_gb=memory_usage_gb,
            capture_time=capture_time,
        )

    if N.check_cuda_graph_backend(N.Phase.PREFILL, N.Backend.DISABLED):
        N.logger.info(
            "Disable prefill CUDA graph because cuda_graph_config "
            "resolved prefill.backend='disabled' (e.g. via "
            "--cuda-graph-backend-prefill=disabled or auto-disable rules)."
        )
        # Prefill cuda graph disabled: route eager prefill through the
        # EagerRunner (its can_run_graph returns False, so _forward_raw's
        # extend branch falls through to the eager path).
        if not model_runner.is_draft_worker:
            return result(eager_runner)
        return result(None)

    # Draft models skip here during __init__; the eagle worker calls
    # this method explicitly (force_for_draft_worker=True) after
    # init_lm_head so graphs capture the final embedding weights.
    if model_runner.is_draft_worker and not force_for_draft_worker:
        return result(None)

    # Skip prefill CG for EAGLE target on tc_piecewise when the fixed server
    # capture ceiling is below FULL. EAGLE target prefill requests FULL, so a
    # NULL or LAST graph is dead; capturing it can perturb FP4/TRTLLM-MoE
    # state and corrupt decode replay (see #28386 and #28870). BCG and FullCG
    # capture FULL for EAGLE targets in PrefillCudaGraphRunner.__init__, so
    # they do not need this skip.
    if (
        model_runner.spec_algorithm.is_eagle()
        and not model_runner.is_draft_worker
        and N.get_server_return_hidden_states_mode() < N.CaptureHiddenMode.FULL
        and N.check_cuda_graph_backend(N.Phase.PREFILL, N.Backend.TC_PIECEWISE)
    ):
        N.logger.info(
            "Disable prefill CUDA graph for EAGLE target on tc_piecewise "
            "to avoid FP4/MoE decode-replay corruption (#28386)."
        )
        return result(eager_runner)

    if (
        model_runner.lora_manager is not None
        and not model_runner.lora_manager.supports_prefill_cuda_graph
    ):
        N.logger.warning(
            "Disable prefill CUDA graph because the current LoRA "
            "configuration does not support it (unsupported LoRA backend, "
            "MoE LoRA without full or breakable capture, or DP attention)."
        )
        return result(eager_runner)

    # Resolve the decoder once. Some VLM wrappers (for example Kimi-VL)
    # expose it as ``language_model`` rather than ``model``.
    try:
        language_model = N.resolve_language_model(model_runner.model)
    except AttributeError:
        N.logger.warning(
            "Disable prefill CUDA graph because the model is not a language model"
        )
        return result(None)

    # Disable prefill CUDA graph for non capture size
    if not N.get_exec().graph.cuda_graph_config.prefill.bs:
        N.logger.warning("Disable prefill CUDA graph because the capture size is not set")
        return result(None)

    prefill_config = N.get_exec().graph.cuda_graph_config.prefill
    prefill_backend = prefill_config.backend
    parallel = N.get_parallel()
    if (
        prefill_backend == N.Backend.BREAKABLE
        and parallel.enable_prefill_cp
        and parallel.pp_size > 1
    ):
        N.logger.warning(
            "Disable prefill CUDA graph because pipeline parallelism combined "
            "with prefill context parallelism is not validated."
        )
        return result(eager_runner)
    context_length = model_runner.model_config.context_len
    if prefill_backend == N.Backend.FULL:
        max_capture_requests = prefill_config.full_prefill_max_req
        if max_capture_requests is None:
            max_capture_requests = max(N.get_schedule().chunked_prefill_size // 512, 1)
        max_capture_requests = min(
            max_capture_requests, model_runner.req_to_token_pool.size
        )
        # Resolve Full's fixed request-axis shape once, just like bs below.
        prefill_config.full_prefill_max_req = max_capture_requests
    else:
        max_capture_requests = model_runner.req_to_token_pool.size
    # The capture dummy batch has at most max_capture_requests rows, and
    # each row can contain at most context_length tokens. Their product is
    # therefore the largest aggregate-token bucket capture can represent.
    max_capture_tokens = max_capture_requests * context_length
    capture_num_tokens = sorted(
        num_tokens
        for num_tokens in prefill_config.bs
        if num_tokens <= max_capture_tokens
    )
    # Resolve the context- and request-capacity-bounded buckets once before
    # constructing the runner so every backend consumes the same config.
    prefill_config.bs = capture_num_tokens
    if not capture_num_tokens:
        N.logger.warning(
            "Disable prefill CUDA graph capture because no configured "
            "capture size fits backend=%s with max_capture_tokens=%s "
            "(max_capture_requests=%s, context_length=%s, request-pool size=%s).",
            prefill_backend,
            max_capture_tokens,
            max_capture_requests,
            context_length,
            model_runner.req_to_token_pool.size,
        )
        return eager_runner

    # Collect attention layers and moe layers from the model. Keep a VLM
    # wrapper that exposes ``language_model`` unchanged: assigning it to
    # ``model`` would register a duplicate module alias and duplicate the
    # model's state-dict namespace.
    if hasattr(model_runner.model, "model"):
        model_runner.model.model = language_model

    # Find the module that owns the decoder `layers`. Models wrap it at
    # varying depths: a direct text model exposes `.layers`, a CausalLM
    # wraps it as `.model.layers`, and some multimodal models add another
    # level (e.g. DeepSeek-OCR: OCR wrapper -> Deepseek*ForCausalLM ->
    # text model -> `.layers`). Descend the `.model` chain until we find it.
    layer_model = language_model
    while not hasattr(layer_model, "layers") and hasattr(layer_model, "model"):
        layer_model = layer_model.model

    if not hasattr(layer_model, "layers"):
        N.log_info_on_rank0(
            N.logger,
            f"Disable {role} prefill CUDA graph because the {role} model does "
            "not have a 'layers' attribute",
        )
        return result(None)

    (
        model_runner.attention_layers,
        model_runner.moe_layers,
        model_runner.moe_fusions,
        model_runner.dsa_indexers,
        model_runner.mha_companion_layers,
    ) = model_runner.get_cuda_graph_layers(layer_model)
    (
        model_runner.attention_layers,
        model_runner.mha_companion_layers,
    ) = index_attention_layers_by_global_id(
        model_runner.attention_layers,
        model_runner.mha_companion_layers,
        layer_model,
    )

    supports_hybrid_graph = (
        getattr(layer_model, "supports_hybrid_prefill_graph", False)
        and prefill_backend == N.Backend.BREAKABLE
        and has_hybrid_graph_handlers(
            attention_layers=model_runner.attention_layers,
            moe_layers=model_runner.moe_layers,
            start_layer=model_runner.layer_info.start_layer,
            end_layer=model_runner.layer_info.end_layer,
        )
    )
    if not supports_hybrid_graph and not N.has_standard_gqa_for_all_local_layers(
        attention_layer_count=sum(
            layer is not None for layer in model_runner.attention_layers
        ),
        start_layer=model_runner.layer_info.start_layer,
        end_layer=model_runner.layer_info.end_layer,
    ):
        # TODO(yuwei): support Non-Standard GQA
        N.log_info_on_rank0(
            N.logger,
            "Disable prefill CUDA graph because some layers do not apply Standard GQA",
        )
        return result(None)

    tic = N.time.perf_counter()
    before_mem = N.get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    capture_name = f"{role} prefill"
    N.logger.info(
        f"Capture {capture_name} CUDA graph begin. "
        f"backend={prefill_backend}, num_tokens={capture_num_tokens}, "
        f"avail mem={before_mem:.2f} GB"
    )

    prefill_runner = N.PrefillCudaGraphRunner(model_runner)

    after_mem = N.get_available_gpu_memory(model_runner.device, model_runner.gpu_id)
    mem_usage = before_mem - after_mem
    capture_time = N.time.perf_counter() - tic
    N.logger.info(
        f"Capture {capture_name} CUDA graph end. "
        f"elapsed={capture_time:.2f} s, "
        f"mem usage={mem_usage:.2f} GB, avail mem={after_mem:.2f} GB."
    )
    return result(prefill_runner, mem_usage, capture_time)


def around_index_attention_layers_by_global_id(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return index_attention_layers_by_global_id(*args, **kwargs)


def around_refresh_deep_gemm_layout_memory_budget(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return refresh_deep_gemm_layout_memory_budget(*args, **kwargs)


def around_capture_cuda_graphs(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return capture_cuda_graphs(*args, **kwargs)


def around_capture_prefill_graph(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return capture_prefill_graph(*args, **kwargs)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.model_executor.model_runner_components.cuda_graph_setup.index_attention_layers_by_global_id", around_index_attention_layers_by_global_id, HookType.AROUND)
    HookRegistry.register("sglang.srt.model_executor.model_runner_components.cuda_graph_setup.refresh_deep_gemm_layout_memory_budget", around_refresh_deep_gemm_layout_memory_budget, HookType.AROUND)
    HookRegistry.register("sglang.srt.model_executor.model_runner_components.cuda_graph_setup.capture_cuda_graphs", around_capture_cuda_graphs, HookType.AROUND)
    HookRegistry.register("sglang.srt.model_executor.model_runner_components.cuda_graph_setup.capture_prefill_graph", around_capture_prefill_graph, HookType.AROUND)
