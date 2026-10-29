"""Fork-compatible multimodal configuration behavior."""

from __future__ import annotations

from types import MappingProxyType
from types import MethodType
from typing import Any

import torch


def supports_ltx_dynamic_batching(config) -> bool:
    del config
    return True


def after_wan_full_vae_path(result: Any, config, *args, **kwargs) -> Any:
    del args, kwargs
    config.vae_config.use_parallel_encode = False
    config.vae_config.use_parallel_decode = False
    return result


def patch_ltx_seed_metadata() -> None:
    from hcu_sglang.compat.aiter import install_aiter_compat

    install_aiter_compat()
    from sglang.multimodal_gen.configs.sample.ltx_2 import (
        LTX2SamplingParams,
        LTX23SamplingParams,
    )

    for sampling_params_cls in (LTX2SamplingParams, LTX23SamplingParams):
        seed_field = sampling_params_cls.__dataclass_fields__["seed"]
        metadata = dict(seed_field.metadata)
        metadata["batch_sig_exclude"] = True
        seed_field.metadata = MappingProxyType(metadata)


def around_ltx_attention_forward(original, attention, *args, **kwargs):
    attention_impl = attention.attn
    original_forward = attention_impl.forward

    def contiguous_forward(module, *forward_args, **forward_kwargs):
        del module
        output = original_forward(*forward_args, **forward_kwargs)
        if isinstance(output, torch.Tensor):
            return output.contiguous()
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return (output[0].contiguous(), *output[1:])
        return output

    attention_impl.forward = MethodType(contiguous_forward, attention_impl)
    try:
        return original(attention, *args, **kwargs)
    finally:
        attention_impl.forward = original_forward


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    patch_ltx_seed_metadata()
    HookRegistry.register(
        "sglang.multimodal_gen.configs.pipeline_configs.ltx_2."
        "LTX2PipelineConfig.supports_dynamic_batching",
        supports_ltx_dynamic_batching,
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.multimodal_gen.runtime.models.dits.ltx_2.LTX2Attention.forward",
        around_ltx_attention_forward,
        HookType.AROUND,
    )
    for class_name in (
        "WanI2V480PConfig",
        "Wan2_2_TI2V_5B_Config",
        "Wan2_2_T2V_A14B_Config",
    ):
        HookRegistry.register(
            "sglang.multimodal_gen.configs.pipeline_configs.wan."
            f"{class_name}.__post_init__",
            after_wan_full_vae_path,
            HookType.AFTER,
        )
