"""Module-level helpers patched onto sglang.srt.models.deepseek_v2."""

from __future__ import annotations

import logging

import torch

from hcu_sglang.models.deepseek.deepseek_v2 import constants as C

logger = logging.getLogger(__name__)


def ds_bmm_wrapper(
    q: torch.Tensor, w: torch.Tensor, scale: float, dtype: torch.dtype
) -> torch.Tensor:
    if abs(scale - 1) < 1e-6:
        return torch.bmm(q.to(dtype).transpose(0, 1), w.to(dtype))
    return torch.bmm(q.to(dtype).transpose(0, 1), w.to(dtype) * scale)


def add_forward_absorb_core_attention_backend(backend_name: str) -> None:
    if backend_name not in C.FORWARD_ABSORB_CORE_ATTENTION_BACKENDS:
        C.FORWARD_ABSORB_CORE_ATTENTION_BACKENDS.append(backend_name)
        logger.info(
            "Added %s to FORWARD_ABSORB_CORE_ATTENTION_BACKENDS.",
            backend_name,
        )
