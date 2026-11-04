from hcu_sglang.utils.detection import is_hcu


def _around(implementation):
    def hook(original, *args, **kwargs):
        if not is_hcu():
            return original(*args, **kwargs)
        return implementation(*args, **kwargs)
    return hook


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from hcu_sglang.layers.moe import topk_source as T, hash_topk_source as H
    targets = [
        ("sglang.srt.layers.moe.topk.fused_topk", T.fused_topk),
        ("sglang.srt.layers.moe.topk.grouped_topk_gpu", T.grouped_topk_gpu),
        ("sglang.srt.layers.moe.topk._biased_grouped_topk_postprocess", T._biased_grouped_topk_postprocess),
        ("sglang.srt.layers.moe.topk.biased_grouped_topk_gpu", T.biased_grouped_topk_gpu),
        ("sglang.srt.layers.moe.topk._post_process_topk_ids", T._post_process_topk_ids),
        ("sglang.srt.layers.moe.topk.select_experts", T.select_experts),
        ("sglang.srt.layers.moe.topk.precomputed_topk_postprocess_is_noop", T.precomputed_topk_postprocess_is_noop),
        ("sglang.srt.layers.moe.topk.TopK.__init__", T.TopK.__init__),
        ("sglang.srt.layers.moe.topk.TopK.forward_cuda", T.TopK.forward_cuda),
        ("sglang.srt.layers.moe.topk.TopK.forward_npu", T.TopK.forward_npu),
        ("sglang.srt.layers.moe.hash_topk.HashTopK.__init__", H.HashTopK.__init__),
        ("sglang.srt.layers.moe.hash_topk.HashTopK.forward", H.HashTopK.forward),
    ]
    for target, implementation in targets:
        HookRegistry.register(target, _around(implementation), HookType.AROUND)
