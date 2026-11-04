"""REPLACE hook implementations for AITER linear-attention paths."""

import logging

from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from typing import Optional, Tuple

import torch
import triton

try:
    from sglang.kernels.ops.attention.fla.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )
except ImportError:
    from sglang.kernels.ops.attention.fla.index import (
        prepare_chunk_indices,
        prepare_chunk_offsets,
    )

CHUNK_SIZE = 64
from hcu_sglang.utils.module_patch import get_original_attr


logger = logging.getLogger(__name__)

try:
    import sglang.kernels.ops.attention.fla.chunk_delta_h  # noqa: F401

    _FLA_MODULE_ROOT = "sglang.kernels.ops.attention.fla"
except ImportError:
    _FLA_MODULE_ROOT = "sglang.srt.layers.attention.fla"

_CHUNK_DELTA_H_MODULE = f"{_FLA_MODULE_ROOT}.chunk_delta_h"
_CHUNK_O_MODULE = f"{_FLA_MODULE_ROOT}.chunk_o"
_FUSED_RECURRENT_MODULE = f"{_FLA_MODULE_ROOT}.fused_recurrent"
_GDN_TRITON_MODULE = "sglang.srt.layers.attention.linear.kernels.gdn_triton"


def register() -> None:
    hooks = (
        (
            f"{_CHUNK_DELTA_H_MODULE}.chunk_gated_delta_rule_fwd_h",
            replacement_chunk_gated_delta_rule_fwd_h,
        ),
        (
            f"{_CHUNK_O_MODULE}.chunk_fwd_o",
            replacement_chunk_fwd_o,
        ),
        (
            f"{_FUSED_RECURRENT_MODULE}.fused_recurrent_gated_delta_rule_packed_decode",
            replacement_fused_recurrent_gated_delta_rule_packed_decode,
        ),
        (
            "sglang.srt.layers.attention.linear.kernels.gdn_triton.TritonGDNKernel.target_verify",
            replacement_triton_gdn_target_verify,
        ),
    )
    for target, hook in hooks:
        logger.info("[hcu_sglang] register REPLACE hook: %s", target)
        HookRegistry.register(target, hook, HookType.REPLACE)


def replacement_chunk_gated_delta_rule_fwd_h(*args, **kwargs):
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        return get_original_attr(_CHUNK_DELTA_H_MODULE, "chunk_gated_delta_rule_fwd_h")(*args, **kwargs)
    from hcu_sglang.layers.attention.fla import chunk_delta_h_source
    return chunk_delta_h_source.chunk_gated_delta_rule_fwd_h(*args, **kwargs)


def replacement_chunk_fwd_o(*args, **kwargs):
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        return get_original_attr(_CHUNK_O_MODULE, "chunk_fwd_o")(*args, **kwargs)
    from hcu_sglang.layers.attention.fla import chunk_o_source
    return chunk_o_source.chunk_fwd_o(*args, **kwargs)


def replacement_fused_recurrent_gated_delta_rule_packed_decode(*args, **kwargs):
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        return get_original_attr(_FUSED_RECURRENT_MODULE, "fused_recurrent_gated_delta_rule_packed_decode")(*args, **kwargs)
    from hcu_sglang.layers.attention.fla import fused_recurrent_source
    return fused_recurrent_source.fused_recurrent_gated_delta_rule_packed_decode(*args, **kwargs)


def replacement_triton_gdn_target_verify(*args, **kwargs):
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        return get_original_attr(_GDN_TRITON_MODULE, "TritonGDNKernel.target_verify")(*args, **kwargs)
    from hcu_sglang.layers.attention.fla.gdn_verify_source import target_verify
    return target_verify(*args, **kwargs)
