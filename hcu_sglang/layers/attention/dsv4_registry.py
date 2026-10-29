from __future__ import annotations

import logging
from hcu_sglang.utils.detection import is_dcu

logger = logging.getLogger(__name__)
_original_dsv4_backend = None


def around_create_dsv4_backend(original, runner):
    if not is_dcu():
        return original(runner)
    from sglang.srt.layers.attention.deepseek_v4_trtllm_backend import create_deepseek_v4_attn_backend
    backend = create_deepseek_v4_attn_backend(runner)
    logger.info("HCU dsv4 backend: %s.%s", type(backend).__module__, type(backend).__name__)
    return backend


def create_hcu_dsv4_backend(runner):
    return around_create_dsv4_backend(_original_dsv4_backend, runner)


def around_create_dsv4_decode_backend(original, factory):
    if not is_dcu():
        return original(factory)
    from sglang.srt.layers.attention.deepseek_v4_trtllm_backend import create_deepseek_v4_multistep_backend
    return "dsv4", create_deepseek_v4_multistep_backend(factory.draft_model_runner, topk=factory.topk, speculative_num_steps=factory.speculative_num_steps)


def around_create_dsv4_prefill_backend(original, factory):
    if not is_dcu():
        return original(factory)
    from sglang.srt.layers.attention.deepseek_v4_trtllm_backend import create_deepseek_v4_attn_backend
    return "dsv4", create_deepseek_v4_attn_backend(factory.draft_model_runner, skip_prefill=False)


def register():
    if not is_dcu():
        return
    global _original_dsv4_backend
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    if ATTENTION_BACKENDS["dsv4"] is not create_hcu_dsv4_backend:
        _original_dsv4_backend = ATTENTION_BACKENDS["dsv4"]
    ATTENTION_BACKENDS["dsv4"] = create_hcu_dsv4_backend
    HookRegistry.register("sglang.srt.layers.attention.attention_registry.create_dsv4_backend", around_create_dsv4_backend, HookType.AROUND)
    HookRegistry.register("sglang.srt.speculative.draft_utils.DraftBackendFactory._create_dsv4_decode_backend", around_create_dsv4_decode_backend, HookType.AROUND)
    HookRegistry.register("sglang.srt.speculative.draft_utils.DraftBackendFactory._create_dsv4_prefill_backend", around_create_dsv4_prefill_backend, HookType.AROUND)
