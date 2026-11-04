from __future__ import annotations
from sglang.srt.arg_groups import deepseek_v4_hook as N
from hcu_sglang.utils.detection import is_hcu

_is_hcu = is_hcu()


def _default_dsv4_attention_backend(server_args: ServerArgs) -> str:
    from sglang.srt.utils.common import is_sm100_supported

    cfg = N.resolving_view(server_args)
    layout = N.envs.SGLANG_DSV4_KV_LAYOUT
    if (
        N.model_config_of(server_args).hf_config.model_type != "deepseek_v41"
        or cfg.device != "cuda"
        or not is_sm100_supported()
        or cfg.kv_cache_dtype != "fp8_e4m3"
        or (layout.is_set() and layout.get().lower() not in ("auto", "v41"))
    ):
        return "flashmla"
    return "flashmla_mega"


def apply_deepseek_v4_defaults(server_args: ServerArgs, model_arch: str) -> None:
    """Apply DeepSeek V4 environment defaults, request limits, and validation."""
    cfg = N.resolving_view(server_args)

    # FlashMLA sparse prefill (SGLANG_OPT_FLASHMLA_SPARSE_PREFILL, default on)
    # currently returns incorrect output for DeepSeek-V4-Flash on ROCm/HIP
    # (MI355X), which breaks the disaggregation nightly. Keep the previous
    # (dense prefill) behavior on ROCm until the sparse kernel is validated
    # there;
    if N.get_platform().is_hip:
        explicit_sparse_prefill = (
            N.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.is_set()
            and N.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()
        )
        if _is_hcu and explicit_sparse_prefill:
            N.logger.warning(
                "Keeping explicitly enabled SGLANG_OPT_FLASHMLA_SPARSE_PREFILL "
                f"on HCU for {model_arch}."
            )
        else:
            N.logger.warning(
                "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL on ROCm/HIP "
                f"for {model_arch}; HCU can explicitly enable it."
            )
            N.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)

    # The kv-cache dtype default moved to the resolution pipeline
    # (arg_groups/overrides.py: _deepseek_v4_kv_cache_dtype), invoked here at
    # its legacy slot.

    N.run_post_process_pass(server_args, N._deepseek_v4_kv_cache_dtype)

    if cfg.dsv4_attn_backend == "auto":
        N.declare_resolution(
            server_args,
            "apply_deepseek_v4_defaults",
            dsv4_attn_backend=_default_dsv4_attention_backend(server_args),
        )

    if cfg.dsv4_attn_backend == "flashmla_mega":
        from sglang.srt.utils.common import is_sm100_supported

        assert (
            N.model_config_of(server_args).hf_config.model_type == "deepseek_v41"
            and cfg.device == "cuda"
            and is_sm100_supported()
        ), "--dsv4-attn-backend flashmla_mega requires DeepSeek-V4.1 on SM100/SM103."

    if cfg.dsv4_attn_backend == "trtllm":
        from sglang.srt.utils.common import is_sm100_supported

        assert cfg.device == "cuda" and is_sm100_supported(), (
            "--dsv4-attn-backend trtllm requires an SM100/SM103 (Blackwell) GPU."
        )
        # The resolution pipeline materializes "auto" as fp8_e4m3 on CUDA.
        assert cfg.kv_cache_dtype in ("auto", "fp8_e4m3"), (
            "--dsv4-attn-backend trtllm requires kv_cache_dtype=fp8_e4m3, "
            f"got {cfg.kv_cache_dtype}."
        )
        assert not cfg.enable_hisparse, (
            "--dsv4-attn-backend trtllm does not support enable_hisparse."
        )
        assert not (
            cfg.attn_cp_size > 1 or cfg.dcp_size > 1 or cfg.enable_prefill_cp
        ), (
            "--dsv4-attn-backend trtllm does not support context parallelism "
            "(prefill CP, attention CP, or decode CP)."
        )
        # The PD handshake rejects peers whose KV layouts differ, but the
        # uniform-FP8 layout has not been run under PD (#37838).
        assert cfg.disaggregation_mode == "null", (
            "--dsv4-attn-backend trtllm does not support PD disaggregation yet "
            "(untested; see https://github.com/sgl-project/sglang/issues/37838)."
        )
        # The trtllm-gen semaphore buffer is sized from the prefill chunk
        # bound; with chunking disabled a single long request has no bound.
        assert cfg.chunked_prefill_size is not None and cfg.chunked_prefill_size > 0, (
            "--dsv4-attn-backend trtllm requires chunked prefill "
            "(--chunked-prefill-size > 0)."
        )
        N.logger.info(
            "DeepSeek V4 attention: trtllm backend enabled "
            "(uniform-FP8 KV pool, decode + sparse prefill)."
        )

    if cfg.max_running_requests is None:
        N.declare_resolution(
            server_args,
            "apply_deepseek_v4_defaults",
            max_running_requests=256,
        )
        N.logger.warning(
            f"Setting max_running_requests to {cfg.max_running_requests} for {model_arch}."
        )

    if cfg.speculative_algorithm is not None:
        assert cfg.speculative_algorithm in (
            "EAGLE",
            "DSPARK",
        ), (
            f"Only EAGLE and DSPARK speculative algorithms are supported for {model_arch}"
        )
        if cfg.speculative_algorithm == "EAGLE":
            assert cfg.speculative_eagle_topk == 1, (
                f"Only EAGLE speculative algorithm with topk == 1 is supported for {model_arch}"
            )


def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:
    """Validate DeepSeek V4 context-parallel configuration."""
    cfg = N.resolving_view(server_args)
    if not cfg.enable_prefill_cp:
        return

    if cfg.cp_strategy not in ("interleave", "zigzag"):
        raise ValueError(
            f"DeepSeekV4 only supports interleave/zigzag CP strategy, got {cfg.cp_strategy}"
        )

    if cfg.cp_strategy == "zigzag" and not N.is_npu():
        raise ValueError(
            "DeepSeekV4 zigzag CP requires the NPU backend; CUDA/HIP backends "
            "reindex with interleave order."
        )
    if N.get_platform().is_hip and not _is_hcu:
        prefill_backend, decode_backend = N.attention_backends_of(
            N.resolved_view(server_args)
        )
        if (prefill_backend, decode_backend) != ("dsv4", "dsv4"):
            raise ValueError(
                "DeepSeekV4 prefill CP on HIP requires the dsv4 attention "
                f"backend for both phases, got prefill={prefill_backend!r}, "
                f"decode={decode_backend!r}."
            )
        unsupported = (
            ("multiple nodes", cfg.nnodes > 1),
            (
                "DeepSeek-V4.1",
                N.model_config_of(server_args).hf_config.model_type == "deepseek_v41",
            ),
            (
                "decoder SWA bounded replay",
                cfg.enable_decoder_swa_bounded_replay,
            ),
            ("--enable-two-batch-overlap", cfg.enable_two_batch_overlap),
        )
        for feature, enabled in unsupported:
            if enabled:
                raise ValueError(
                    f"DeepSeekV4 prefill CP on HIP does not support {feature} yet."
                )

    attn_dp_size = N.context_parallel_attn_dp_size(cfg, "DeepSeek-V4 context parallelism")
    N.declare_resolution(
        server_args,
        "validate_deepseek_v4_cp",
        attn_dp_size=attn_dp_size,
        dp_size=1,
    )
    N.declare_resolution(
        server_args,
        "validate_deepseek_v4_cp",
        moe_dense_tp_size=1,
    )
    N.declare_resolution(
        server_args,
        "validate_deepseek_v4_cp",
        attn_cp_size=cfg.tp_size // attn_dp_size,
    )
    if not N.is_npu():
        assert attn_dp_size == 1, (
            "For round-robin split mode, dp attention is not supported."
        )
        assert cfg.tp_size <= 8, (
            "Context parallel only supports single machine (tp_size <= 8). Cross-machine CP has precision issues."
        )
    supported_a2a_backends = ("none", "deepep", "megamoe", "mori")
    if cfg.moe_a2a_backend not in supported_a2a_backends:
        raise ValueError(
            f"DeepSeekV4 CP supports moe_a2a_backend in {supported_a2a_backends}, "
            f"got {cfg.moe_a2a_backend!r}."
        )
    explicit_hcu_sparse_cp = (
        _is_hcu
        and N.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.is_set()
        and N.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()
    )
    if explicit_hcu_sparse_cp:
        N.logger.warning(
            "Keeping explicitly enabled HCU sparse prefill with DeepSeekV4 CP."
        )
    elif N.model_config_of(server_args).hf_config.model_type != "deepseek_v41":
        # The generic CP-aware sparse chunk cache is validated on V4.1 only.
        N.logger.warning(
            "Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL because DeepSeekV4 "
            "context parallelism is enabled."
        )
        N.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)
    N.logger.warning(
        f"Enable Context Parallel for DeepSeekV4, "
        f"strategy={cfg.cp_strategy}, "
        f"attn_dp_size={attn_dp_size}, moe_dense_tp_size={cfg.moe_dense_tp_size}, "
        f"attn_cp_size={cfg.attn_cp_size}, ep_size={cfg.ep_size}, tp_size={cfg.tp_size}"
    )


def around_defaults(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return apply_deepseek_v4_defaults(*args, **kwargs)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.arg_groups.deepseek_v4_hook.apply_deepseek_v4_defaults", around_defaults, HookType.AROUND)
