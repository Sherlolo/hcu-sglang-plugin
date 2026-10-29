"""Wan HCU sparse-linear-attention configuration compatibility."""

from __future__ import annotations

import contextvars
from typing import Any, Callable

from hcu_sglang.utils.detection import is_dcu


_local_blocks = contextvars.ContextVar("wan_sla_local_blocks", default=0)
_skip_linear = contextvars.ContextVar("wan_sla_skip_linear", default=True)


def _config_value(config: Any, keys: tuple[str, ...]) -> Any:
    if not config:
        return None
    for key in keys:
        try:
            if key in config:
                return config[key]
        except TypeError:
            pass
        if hasattr(config, key):
            return getattr(config, key)
    return None


def resolve_attention_type(value: str, backend: str | None) -> str:
    if value != "original" or backend is None:
        return value
    return {"sla_attn": "sla", "sage_sla_attn": "sagesla"}.get(
        backend.lower(), value
    )


def resolve_topk(value: float, attention_type: str, config: Any) -> float:
    if attention_type not in ("sla", "sagesla"):
        return value
    raw = _config_value(config, ("sla_topk", "topk"))
    if raw is None:
        return value
    parsed = float(raw)
    if not 0 < parsed <= 1:
        raise ValueError(f"sla_topk must be in (0, 1], got {parsed}")
    return parsed


def resolve_local_blocks(value: int, attention_type: str, config: Any) -> int:
    if attention_type not in ("sla", "sagesla"):
        return value
    raw = _config_value(config, ("sla_local_blocks", "local_blocks"))
    if raw is None:
        return value
    parsed = int(raw)
    if parsed < 0:
        raise ValueError(
            f"sla_local_blocks must be a non-negative integer, got {parsed}"
        )
    return parsed


def resolve_skip_linear(value: bool, attention_type: str, config: Any) -> bool:
    if attention_type not in ("sla", "sagesla"):
        return value
    raw = _config_value(
        config, ("sla_skip_linear", "skip_linear_branch", "skip_linear")
    )
    if raw is None or isinstance(raw, bool):
        return value if raw is None else raw
    if isinstance(raw, int) and raw in (0, 1):
        return bool(raw)
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in ("1", "true", "yes", "on"):
            return True
        if normalized in ("0", "false", "no", "off"):
            return False
    raise ValueError(f"sla_skip_linear must be a boolean, got {raw!r}")


def around_wan_model_init(
    original: Callable[..., None], model, config, hf_config, quant_config=None
) -> None:
    if not is_dcu():
        return original(model, config, hf_config, quant_config)

    from sglang.multimodal_gen.runtime.server_args.server_args import (
        get_global_server_args,
    )

    server_args = get_global_server_args()
    attention_type = resolve_attention_type(
        config.attention_type, server_args.attention_backend
    )
    topk = resolve_topk(
        config.sla_topk, attention_type, server_args.attention_backend_config
    )
    local_blocks = resolve_local_blocks(
        getattr(config, "sla_local_blocks", 0),
        attention_type,
        server_args.attention_backend_config,
    )
    skip_linear = resolve_skip_linear(
        getattr(config, "sla_skip_linear_branch", True),
        attention_type,
        server_args.attention_backend_config,
    )

    original_type, original_topk = config.attention_type, config.sla_topk
    type_token = _local_blocks.set(local_blocks)
    skip_token = _skip_linear.set(skip_linear)
    config.attention_type, config.sla_topk = attention_type, topk
    try:
        return original(model, config, hf_config, quant_config)
    finally:
        config.attention_type, config.sla_topk = original_type, original_topk
        _local_blocks.reset(type_token)
        _skip_linear.reset(skip_token)


def after_minimal_a2a_init(result: Any, attention_op, *args, **kwargs) -> Any:
    local_attn = getattr(attention_op, "local_attn", None)
    if is_dcu() and local_attn is not None:
        from hcu_sglang.multimodal_gen.rocm_sparse_linear_attn import (
            RocmSparseLinearAttentionImpl,
        )

        if isinstance(local_attn, RocmSparseLinearAttentionImpl):
            local_attn.local_blocks = _local_blocks.get()
            local_attn.skip_linear_branch = _skip_linear.get()
    return result


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.multimodal_gen.runtime.models.dits.wanvideo."
        "WanTransformer3DModel.__init__",
        around_wan_model_init,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.layers.attention.turbo_layer."
        "MinimalA2AAttnOp.__init__",
        after_minimal_a2a_init,
        HookType.AFTER,
    )
