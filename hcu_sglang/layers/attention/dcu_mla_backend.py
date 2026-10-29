"""Backward-compatible aliases for the renamed HCU MLA backend."""

from hcu_sglang.layers.attention.hcu_mla_backend import (
    HCUMLABackend as DCUMLABackend,
    HCUMLAMultiStepDraftBackend as DCUMLAMultiStepDraftBackend,
)

__all__ = ["DCUMLABackend", "DCUMLAMultiStepDraftBackend"]
