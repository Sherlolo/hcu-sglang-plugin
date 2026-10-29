"""Temporary Triton fallback for the unsupported DSpark AITER MoE shape."""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

_FALLBACK_ENV = "SGLANG_HCU_DSPARK_AITER_MOE_FALLBACK"
_D_SPARK_GEOMETRY = (512, 4096, 4096, 257, 7)
_warned = False


def is_missing_aiter_backend_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "aiter" in message and "no suitable backend found" in message


def _fallback_enabled() -> bool:
    return os.getenv(_FALLBACK_ENV, "1").lower() not in {"0", "false", "off", "no"}


def _first_weight(weight: Any) -> Any:
    return weight[0] if isinstance(weight, tuple) else weight


def _topk_ids(dispatch_output: Any) -> Any:
    topk_output = dispatch_output.topk_output
    if hasattr(topk_output, "topk_ids"):
        return topk_output.topk_ids
    return topk_output[1]


def _geometry(quant_method: Any, layer: Any, dispatch_output: Any) -> tuple[int, ...]:
    hidden_states = dispatch_output.hidden_states
    topk_ids = _topk_ids(dispatch_output)
    w13_weight = _first_weight(layer.w13_weight)
    w2_weight = _first_weight(layer.w2_weight)
    num_experts = getattr(quant_method.moe_runner_config, "num_experts", None)
    if num_experts is None:
        num_experts = w13_weight.shape[0]
    return (
        w13_weight.shape[1],
        w2_weight.shape[1],
        hidden_states.shape[1],
        num_experts,
        topk_ids.shape[1],
    )


def try_run_dspark_triton_fallback(
    quant_method: Any,
    layer: Any,
    dispatch_output: Any,
) -> Optional[Any]:
    if not _fallback_enabled() or getattr(layer, "_moec_shuffled", False):
        return None

    geometry = _geometry(quant_method, layer, dispatch_output)
    if geometry != _D_SPARK_GEOMETRY:
        return None

    from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

    global _warned
    if not _warned:
        logger.warning(
            "AITER has no FP8 MoE backend for the DSpark geometry "
            "N1=%d, N2=%d, K=%d, E=%d, topk=%d; using Triton. "
            "Set %s=0 to disable this temporary fallback.",
            *geometry,
            _FALLBACK_ENV,
        )
        _warned = True

    triton_runner = MoeRunner(MoeRunnerBackend.TRITON, quant_method.moe_runner_config)
    quant_info = TritonMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        use_fp8_w8a8=True,
        per_channel_quant=True,
        w13_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        a13_scale=layer.w13_input_scale,
        a2_scale=layer.w2_input_scale,
    )
    return triton_runner.run(dispatch_output, quant_info)
