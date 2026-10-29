from hcu_sglang.utils.detection import is_dcu


def register_hooks():
    if not is_dcu():
        return 0
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from hcu_sglang.models.deepseek.current_v4.execution import core, mhc

    HookRegistry.register("sglang.srt.models.deepseek_v4.DeepseekV4DecoderLayer", core.DeepseekV4PlatformDecoderLayer, HookType.REPLACE)
    for name in ("_mix_stats_impl", "post"):
        HookRegistry.register("sglang.srt.models.deepseek_v4_mhc." + name, getattr(mhc, name), HookType.REPLACE)
    return 3
