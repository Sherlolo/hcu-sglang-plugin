from hcu_sglang.utils.detection import is_dcu


def register_hooks():
    if not is_dcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from hcu_sglang.layers.attention.current_dsa import backend, rotation, indexer

    for name in ("DeepseekSparseAttnBackend", "DeepseekSparseAttnMultiStepBackend"):
        HookRegistry.register("sglang.srt.layers.attention.dsa_backend." + name, getattr(backend, name), HookType.REPLACE)

    HookRegistry.register("sglang.srt.layers.attention.dsa.dsa_indexer.rotate_activation", rotation.rotate_activation, HookType.REPLACE)

    for target in ("sglang.srt.layers.attention.dsa.dsa_indexer.Indexer", "sglang.srt.models.deepseek_v2.Indexer"):
        HookRegistry.register(target, indexer.Indexer, HookType.REPLACE)
