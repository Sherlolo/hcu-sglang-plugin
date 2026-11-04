from __future__ import annotations

import torch
from hcu_sglang.utils.detection import is_hcu


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


def register_hooks():
    from . import row_fusion, current_linear
    row_fusion.register_hooks()
    current_linear.register_hooks()
