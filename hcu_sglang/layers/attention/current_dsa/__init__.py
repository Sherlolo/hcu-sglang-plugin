from hcu_sglang.utils.detection import is_dcu


def register_hooks():
    if not is_dcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from hcu_sglang.layers.attention.current_dsa import backend

    for name in ("DeepseekSparseAttnBackend", "DeepseekSparseAttnMultiStepBackend"):
        HookRegistry.register("sglang.srt.layers.attention.dsa_backend." + name, getattr(backend, name), HookType.REPLACE)
