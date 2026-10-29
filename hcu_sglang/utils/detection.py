"""DCU hardware detection (plugin-internal)."""

from __future__ import annotations

import logging
from functools import lru_cache

import torch


logger = logging.getLogger(__name__)

_SUPPORTED_ARCHS = ("gfx936", "gfx938", "gfx928")
_FP8_ARCHS = ("gfx938",)


@lru_cache(maxsize=1)
def is_dcu() -> bool:
    if not getattr(torch.version, "hip", None):
        return False
    try:
        props = torch.cuda.get_device_properties(0)
        gcn_arch = getattr(props, "gcnArchName", "")
        return any(gfx in gcn_arch for gfx in _SUPPORTED_ARCHS)
    except Exception as e:
        logger.warning("DCU detection failed: %s", e)
        return False


@lru_cache(maxsize=1)
def is_dcu_native_fp8_supported() -> bool:
    if not is_dcu():
        return False
    try:
        gcn_arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
        return any(gfx in gcn_arch for gfx in _FP8_ARCHS)
    except Exception as e:
        logger.warning("DCU native FP8 detection failed: %s", e)
        return False


def get_gcn_arch(device_id: int = 0) -> str:
    try:
        props = torch.cuda.get_device_properties(device_id)
        return getattr(props, "gcnArchName", "")
    except Exception:
        return ""
