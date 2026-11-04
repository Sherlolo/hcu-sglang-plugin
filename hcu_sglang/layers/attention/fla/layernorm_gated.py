"""REPLACE hook implementation for HCU optimized gated layer norm."""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

import torch
import triton

from sglang.srt.utils import cdiv, device_context
from hcu_sglang.utils.module_patch import get_original_attr


logger = logging.getLogger(__name__)

try:
    import sglang.kernels.ops.attention.fla.layernorm_gated  # noqa: F401

    _LAYERNORM_MODULE = "sglang.kernels.ops.attention.fla.layernorm_gated"
except ImportError:
    _LAYERNORM_MODULE = "sglang.kernels.ops.attention.fla.layernorm_gated"


def register() -> None:
    target = f"{_LAYERNORM_MODULE}._layer_norm_fwd"
    logger.info("[hcu_sglang] register REPLACE hook: %s", target)
    HookRegistry.register(target, replacement_layer_norm_fwd, HookType.REPLACE)


def replacement_layer_norm_fwd(*args, **kwargs):
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        return get_original_attr(_LAYERNORM_MODULE, "_layer_norm_fwd")(*args, **kwargs)
    from hcu_sglang.layers.attention.fla import layernorm_gated_source
    return layernorm_gated_source._layer_norm_fwd(*args, **kwargs)
