"""Registration glue for the out-of-tree HCU MLA backend."""

from __future__ import annotations

from typing import Any


def create_hcu_mla_backend(runner: Any):
    if not runner.use_mla_backend:
        raise ValueError("hcu_mla backend can only be used with MLA models.")

    from hcu_sglang.layers.attention.hcu_mla_backend import HCUMLABackend

    return HCUMLABackend(runner)


def _create_hcu_mla_decode_backend(factory: Any):
    from hcu_sglang.layers.attention.hcu_mla_backend import (
        HCUMLAMultiStepDraftBackend,
    )

    return (
        "hcu_mla",
        HCUMLAMultiStepDraftBackend(
            factory.draft_model_runner,
            factory.topk,
            factory.speculative_num_steps,
        ),
    )


def _create_hcu_mla_prefill_backend(factory: Any):
    from hcu_sglang.layers.attention.hcu_mla_backend import HCUMLABackend

    return (
        "hcu_mla",
        HCUMLABackend(factory.draft_model_runner, skip_prefill=False),
    )


def add_hcu_mla_to_draft_backend_map(
    factory: Any,
    backend_name: str,
    backend_map: dict,
    error_template: str,
    stamps_children: bool = False,
):
    del error_template, stamps_children
    creator = (
        _create_hcu_mla_decode_backend
        if backend_name == "decode_attention_backend"
        else _create_hcu_mla_prefill_backend
    )
    backend_map.setdefault("hcu_mla", lambda: creator(factory))


def register() -> None:
    from . import dsv4_registry
    dsv4_registry.register()
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    ATTENTION_BACKENDS["hcu_mla"] = create_hcu_mla_backend
    HookRegistry.register(
        "sglang.srt.speculative.draft_utils.DraftBackendFactory._create_backend",
        add_hcu_mla_to_draft_backend_map,
        HookType.BEFORE,
    )
