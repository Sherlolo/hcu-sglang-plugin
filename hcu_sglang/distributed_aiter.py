# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/custom_all_reduce.py


from __future__ import annotations

import os
from functools import partial
from typing import Any, Optional

from sglang.srt.distributed.device_communicators import custom_all_reduce as S
from hcu_sglang.utils.detection import is_dcu


def _aiter_enable_register_for_capturing(
    tms_cudagraph: bool, hf_config: Optional[Any] = None
) -> bool:
    """Legacy IPC opt-in, with model-specific HCU copy-in safeguards."""
    # On HCU, registering the AITER all-reduce input directly into the decode
    # graph replays the capture warmup instead of the live residual. Copy-in
    # uses the pre-registered workspace and matches eager execution.
    #
    # DeepSeek-V3 / V3.1 uses model_type deepseek_v3. GLM-5.2 DSA shares TopK
    # every four layers; its raw head_dim=192 is normalized to 64 by HF config
    # loading, so identify that checkpoint by the indexer topology instead.
    # Other models retain the legacy memory-saver opt-in.
    model_type = getattr(hf_config, "model_type", None)
    if is_dcu() and model_type == "deepseek_v3":
        return False
    if (
        is_dcu()
        and model_type == "glm_moe_dsa"
        and getattr(hf_config, "index_topk_freq", None) == 4
        and getattr(hf_config, "index_skip_topk_offset", None) == 3
    ):
        return False
    return tms_cudagraph


def _aiter_max_size_bytes() -> Optional[int]:
    """Translate AITER_AR_MAX_SIZE_MB into the constructor's byte limit."""
    raw = os.environ.get("AITER_AR_MAX_SIZE_MB")
    if raw is None or raw.strip() == "":
        return None
    try:
        size_mb = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"AITER_AR_MAX_SIZE_MB must be an integer, got {raw!r}"
        ) from exc
    if size_mb <= 0:
        raise ValueError(f"AITER_AR_MAX_SIZE_MB must be positive, got {size_mb}")
    return size_mb * 1024 * 1024


def dispatch_custom_allreduce(
    group: S.ProcessGroup,
    device: S.torch.device,
    backend: str = "auto",
):
    """Return the CustomAllreduce class (or partial) to use, per backend.

    Backends (see aiter custom_allreduce doc, sections 5 and 6.4):
        - ``off``    : return ``None``; caller must not construct CA.
        - ``native`` : force SGLang's own CA (JIT V2 on CUDA where eligible),
                       ignoring aiter.
        - ``aiter``  : force aiter CA on HIP/HCU. Raises on non-HIP or when the
                       aiter package cannot be imported. Strict-Fabric
                       semantics (re-raise vs. fall back) are enforced at the
                       ``GroupCoordinator`` construction site.
        - ``auto``   : HIP + aiter importable -> aiter (with the same transport
                       rules); otherwise -> native.

    Under aiter, ``enable_register_for_capturing`` is driven by transport
    (doc section 4.3): under ``fabric`` / ``auto`` it must be ``False`` because
    aiter Fabric cannot register arbitrary graph allocations; under ``ipc`` we
    preserve the historical memory-saver-driven behaviour.

    On CUDA the JIT-compiled v2 implementation is used by default when the
    group satisfies the NVLink-clique check. Set
    SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0 to fall back to the legacy
    CustomAllreduce. Multi-node v2 is admitted only for a single NVLink clique
    (see can_use_custom_all_reduce_v2); other cross-node groups fall back to
    NCCL.
    """
    backend = (backend or "auto").lower()
    if backend not in ("auto", "native", "aiter", "off"):
        S.logger.warning(
            "[AR] Unknown custom_all_reduce_backend=%r, coercing to 'auto'.",
            backend,
        )
        backend = "auto"

    if backend == "off":
        return None

    def _native():
        if S._is_cuda and S.envs.SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2.get():
            from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
                CustomAllReduceV2,
                can_use_custom_all_reduce_v2,
            )

            if can_use_custom_all_reduce_v2(group=group, device=device):
                S.logger.debug("[AR] Using CustomAllReduceV2 (JIT-compiled)")
                return CustomAllReduceV2
        return S.CustomAllreduce

    def _aiter_partial():
        from aiter.dist.device_communicators.custom_all_reduce import (
            CustomAllreduce as AiterCustomAllreduce,
        )

        class GraphSafeAiterCustomAllreduce(AiterCustomAllreduce):
            """Run real collectives during graph warmup, before HIP capture."""

            def _needs_real_graph_warmup(self) -> bool:
                return (
                    S.get_bool_env_var(
                        "SGLANG_AITER_AR_REAL_GRAPH_WARMUP", default="true"
                    )
                    and self._IS_CAPTURING
                    and not S.torch.cuda.is_current_stream_capturing()
                    and not S.is_in_tc_piecewise_cuda_graph()
                )

            def custom_all_reduce(
                self,
                input: S.torch.Tensor,
                use_new: bool = True,
                open_fp8_quant: bool = False,
            ):
                if self._needs_real_graph_warmup():
                    if self.disabled or not self.should_custom_ar(input):
                        return None
                    return self.all_reduce(
                        input,
                        use_new=use_new,
                        open_fp8_quant=open_fp8_quant,
                        registered_input=False,
                    )
                return super().custom_all_reduce(
                    input,
                    use_new=use_new,
                    open_fp8_quant=open_fp8_quant,
                )

            def custom_fused_ar_rms(
                self,
                input: S.torch.Tensor,
                residual_inp: S.torch.Tensor,
                weight: S.torch.Tensor,
                eps: float,
                use_1stage: bool = False,
            ):
                if self._needs_real_graph_warmup():
                    if self.disabled or not self.should_custom_ar(input):
                        return None
                    return self.fused_ar_rms(
                        input,
                        residual_inp,
                        w=weight,
                        eps=eps,
                        registered=False,
                        use_1stage=use_1stage,
                    )
                return super().custom_fused_ar_rms(
                    input,
                    residual_inp,
                    weight,
                    eps,
                    use_1stage,
                )

        transport = os.environ.get("AITER_AR_TRANSPORT", "ipc").lower()
        if transport == "ipc":
            hf_config = None
            if is_dcu():
                from sglang.srt.arg_groups.model_override_base import model_config_of
                from sglang.srt.runtime_context import get_server_args

                hf_config = model_config_of(get_server_args()).hf_config
            enable_reg = _aiter_enable_register_for_capturing(
                S.envs.SGLANG_MEMORY_SAVER_CUDA_GRAPH.get(), hf_config
            )
        elif transport in ("fabric", "auto"):
            # Fabric cannot register arbitrary graph allocations; auto may
            # resolve to Fabric, so also force copy-in mode.
            enable_reg = False
        else:
            S.logger.warning(
                "[AR] Unknown AITER_AR_TRANSPORT=%r; treating as copy-in.",
                transport,
            )
            enable_reg = False
        S.logger.info(
            "[AR] Using AiterCustomAllreduce (%s, transport=%s, "
            "enable_register_for_capturing=%s)",
            "HCU" if is_dcu() else "AMD",
            transport,
            enable_reg,
        )
        constructor_kwargs = {"enable_register_for_capturing": enable_reg}
        max_size = _aiter_max_size_bytes()
        if max_size is not None:
            constructor_kwargs["max_size"] = max_size
            S.logger.info("[AR] AITER max workspace size: %d MiB", max_size >> 20)
        return partial(GraphSafeAiterCustomAllreduce, **constructor_kwargs)

    if backend == "native":
        return _native()

    if backend == "aiter":
        if not S._is_hip:
            raise RuntimeError(
                "custom_all_reduce_backend=aiter requires HIP/ROCm; "
                "current platform does not support aiter CA."
            )
        # aiter has no deterministic_all_reduce path; refuse rather than
        # silently degrade correctness.
        if S._use_amd_deterministic_impl():
            raise RuntimeError(
                "custom_all_reduce_backend=aiter is incompatible with "
                "deterministic inference (SGLANG_USE_1STAGE_ALLREDUCE / "
                "SGLANG_ENABLE_DETERMINISTIC_INFERENCE). Use 'native' or "
                "disable deterministic inference."
            )
        return _aiter_partial()

    # backend == "auto"
    if S._is_cuda or S._is_musa:
        return _native()

    assert S._is_hip

    if S.envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
        if S.envs.SGLANG_USE_1STAGE_ALLREDUCE.get():
            S.logger.debug(
                "[AR] All-reduce: 1-stage kernel (SGLANG_USE_1STAGE_ALLREDUCE=1)"
            )
        else:
            S.logger.debug("[AR] All-reduce: default (SGLANG_USE_1STAGE_ALLREDUCE=0)")
    elif S.envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get():
        S.logger.debug(
            "[AR] All-reduce: 1-stage kernel (deterministic inference enabled)"
        )
    else:
        S.logger.debug("[AR] All-reduce: default")

    # On AMD with 1-stage AR, use sglang's CustomAllreduce
    # (AiterCustomAllreduce doesn't have deterministic_all_reduce method).
    if S._use_amd_deterministic_impl():
        return S.CustomAllreduce

    try:
        return _aiter_partial()
    except ImportError as e:
        S.logger.warning(
            "[AR] Aiter custom all-reduce not available; "
            "falling back to sglang CustomAllreduce. Details: %s",
            e,
        )
        return S.CustomAllreduce
