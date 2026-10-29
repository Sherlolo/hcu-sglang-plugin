"""REPLACE hook implementation for DCU fused top-k softmax."""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

import torch

from hcu_sglang.utils.module_patch import get_original_attr


logger = logging.getLogger(__name__)

_TOPK_MODULE = "sglang.srt.layers.moe.topk"


def register() -> None:
    target = "sglang.srt.layers.moe.topk.fused_topk"
    logger.info("[hcu_sglang] register REPLACE hook: %s", target)
    HookRegistry.register(target, replacement_fused_topk, HookType.REPLACE)


def replacement_fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias=None,
    scoring_func: str = "softmax",
):
    """Replacement for sglang.srt.layers.moe.topk.fused_topk."""
    if scoring_func != "softmax":
        original = get_original_attr(_TOPK_MODULE, "fused_topk")
        return original(
            hidden_states,
            gating_output,
            topk,
            renormalize,
            correction_bias=correction_bias,
            scoring_func=scoring_func,
        )

    try:
        from lightop import op
    except Exception as exc:
        logger.warning(
            "[hcu_sglang] lightop topk_softmax is unavailable, fallback: %s",
            exc,
        )
        original = get_original_attr(_TOPK_MODULE, "fused_topk")
        return original(
            hidden_states,
            gating_output,
            topk,
            renormalize,
            correction_bias=correction_bias,
            scoring_func=scoring_func,
        )

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    num_tokens = hidden_states.shape[0]
    topk_weights = torch.empty(
        num_tokens, topk, dtype=torch.float32, device=hidden_states.device
    )
    topk_ids = torch.empty(
        num_tokens, topk, dtype=torch.int32, device=hidden_states.device
    )

    op.topk_softmax(
        topk_weights,
        topk_ids,
        None,
        gating_output,
        renormalize,
    )
    return topk_weights, topk_ids
