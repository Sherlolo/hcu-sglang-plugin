"""GLM4-MoE HCU compatibility hooks."""

from __future__ import annotations

from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_hcu


def after_mlp_init(result: Any, mlp, *args, **kwargs) -> Any:
    return result


def around_forward_normal(original: Callable[..., Any], moe, hidden_states) -> Any:
    if not is_hcu():
        return original(moe, hidden_states)
    from hcu_sglang.models.glm4_moe_source import forward_normal
    return forward_normal(moe, hidden_states)


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
