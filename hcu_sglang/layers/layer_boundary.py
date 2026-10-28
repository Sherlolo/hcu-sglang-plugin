"""Per-call normalization handoff without mutating shared stage plans."""
from __future__ import annotations

from copy import copy


class HcuSkipNorm:
    def __call__(self, hidden_states, residual=None, post_residual_addition=None, **kwargs):
        if post_residual_addition is not None:
            raise NotImplementedError("Deferred HCU normalization requires a separate post-residual addition handoff")
        return hidden_states if residual is None else (hidden_states, residual)


def around_prepare(original, stage, hidden_states, forward_batch, *, cache=None, **call):
    skip_norm = call.pop("hcu_skip_layernorm", False)
    if not skip_norm:
        return original(stage, hidden_states, forward_batch, cache=cache, **call)
    # Consume the HCU option before public prepare dispatches to the strict
    # attention reader. Keep the reusable plan immutable across forwards.
    invocation = copy(stage)
    invocation.plan = copy(stage.plan)
    invocation.plan.norm = HcuSkipNorm()
    return original(invocation, hidden_states, forward_batch, cache=cache, **call)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.layers.layer_boundary.stage.StageBoundary.prepare",
                          around_prepare, HookType.AROUND)
