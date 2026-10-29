"""MiniMax-H3 HCU backend selection and QK-Norm compatibility."""

from __future__ import annotations

import logging
from typing import Any, Callable

from hcu_sglang.utils.detection import is_dcu

logger = logging.getLogger(__name__)


def after_adjust_attention_backend(result: Any, server_args: Any) -> Any:
    if not is_dcu() or server_args.attention_backend != "fa":
        return result

    from sglang.multimodal_gen.runtime.server_args.server_args import Backend

    if (
        server_args.backend == Backend.DIFFUSERS
        or type(server_args.pipeline_config).__name__ != "MiniMaxH3PipelineConfig"
    ):
        return result

    defaults = {"text_encoder": "torch_sdpa", "transformer": "fa"}
    for component, backend in defaults.items():
        if component not in server_args.component_attention_backends:
            server_args.component_attention_backends[component] = backend
            logger.info(
                "Automatically set MiniMax-H3 HCU %s attention backend to %s",
                component,
                backend,
            )
    return result


def around_apply_qk_norm(
    original: Callable[..., Any], q, k, q_norm, k_norm, head_dim: int
):
    if not is_dcu():
        return original(q, k, q_norm, k_norm, head_dim)
    return q_norm(q), k_norm(k)


def after_post_load_weights(result: Any, model: Any) -> Any:
    if is_dcu():
        model._resolve_attention_backend_once()
    return result


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.multimodal_gen.runtime.server_args.server_args."
        "ServerArgs._adjust_attention_backend",
        after_adjust_attention_backend,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3._apply_qk_norm",
        around_apply_qk_norm,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.models.dits.minimax_h3."
        "MiniMaxH3DiTModel.post_load_weights",
        after_post_load_weights,
        HookType.AFTER,
    )
