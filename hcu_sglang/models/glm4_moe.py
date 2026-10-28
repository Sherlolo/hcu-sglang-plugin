"""GLM4-MoE HCU compatibility hooks."""

from __future__ import annotations

from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_dcu


class _MissingQuantConfig:
    weight_block_size = None

    @classmethod
    def get_name(cls) -> str:
        return "unquantized"


def after_mlp_init(result: Any, mlp, *args, **kwargs) -> Any:
    del args, kwargs
    if not is_dcu():
        return result
    for projection_name in ("gate_up_proj", "down_proj"):
        projection = getattr(mlp, projection_name, None)
        weight = getattr(projection, "weight", None)
        quant_method = getattr(projection, "quant_method", None)
        if (
            quant_method is not None
            and weight is not None
            and weight.dtype == torch.float8_e4m3fn
            and not hasattr(quant_method, "quant_config")
        ):
            quant_method.quant_config = _MissingQuantConfig()
    return result


def around_forward_normal(
    original: Callable[..., Any], moe, hidden_states
) -> Any:
    if not is_dcu():
        return original(moe, hidden_states)

    routed_scaling_factor = moe.routed_scaling_factor
    moe.routed_scaling_factor = 1.0
    try:
        return original(moe, hidden_states)
    finally:
        moe.routed_scaling_factor = routed_scaling_factor


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.models.glm4_moe.Glm4MoeMLP.__init__",
        after_mlp_init,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.models.glm4_moe.Glm4MoeSparseMoeBlock.forward_normal",
        around_forward_normal,
        HookType.AROUND,
    )
