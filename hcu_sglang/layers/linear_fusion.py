"""Normalization handoff for the current community linear interface.

The HCU fork's INT8 prequantized projection is incompatible with a channel-FP8
W8A8Fp8LinearMethod. Preserve its configured quantizer instead of sending INT8
buffers to an FP8 method, while keeping residual-stream updates identical.
"""
from __future__ import annotations

import torch
from hcu_sglang.utils.detection import is_dcu


def normalize_projection_input(hidden, weight, residual=None, eps=1e-6, update=True):
    if residual is not None:
        if residual.shape != hidden.shape:
            raise ValueError("RMS handoff requires matching hidden/residual shapes")
        combined = hidden + residual
        if update:
            residual.copy_(combined)
    else:
        combined = hidden
    value = combined.float()
    value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
    return (value * weight.float()).to(hidden.dtype)


def around_column_forward(original, layer, input_, rms_weight=None,
                          residual=None, update_hd=True):
    if rms_weight is None:
        return original(layer, input_)
    if not is_dcu():
        raise ValueError("HCU normalization handoff requires a supported HCU device")
    normalized = normalize_projection_input(
        input_, rms_weight, residual, getattr(layer, "eps", 1e-6), update_hd
    )
    return original(layer, normalized)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.layers.linear.ColumnParallelLinear.forward",
                          around_column_forward, HookType.AROUND)
