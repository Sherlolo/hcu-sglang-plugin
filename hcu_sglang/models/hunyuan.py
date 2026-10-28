"""Install plugin-owned Hunyuan V3 model implementations on HCU."""

from __future__ import annotations


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from hcu_sglang.models.hunyuan_v3 import HYV3ForCausalLM
    from hcu_sglang.models.hunyuan_v3_nextn import HYV3ForCausalLMNextN

    HookRegistry.register(
        "sglang.srt.models.hunyuan_v3.HYV3ForCausalLM",
        HYV3ForCausalLM,
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.srt.models.hunyuan_v3_nextn.HYV3ForCausalLMNextN",
        HYV3ForCausalLMNextN,
        HookType.REPLACE,
    )
