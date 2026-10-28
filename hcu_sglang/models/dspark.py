"""HCU compatibility for DeepSeek V4 DSpark checkpoint scale names."""

from __future__ import annotations

from hcu_sglang.utils.detection import is_dcu


def before_load_weights(model, weights):
    if not is_dcu():
        return None
    parameter_names = {name for name, _ in model.named_parameters()}
    model._dspark_scale_suffix = (
        "weight_scale"
        if any(name.endswith(".weight_scale") for name in parameter_names)
        else "weight_scale_inv"
    )
    return None


def after_remap(mapped, model, name):
    del name
    if not is_dcu() or mapped is None:
        return mapped
    suffix = getattr(model, "_dspark_scale_suffix", "weight_scale_inv")
    if suffix != "weight_scale_inv":
        mapped = mapped.replace(".weight_scale_inv", f".{suffix}")
    return mapped


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    target = "sglang.srt.models.deepseek_v4_dspark.DeepseekV4ForCausalLMDSpark"
    HookRegistry.register(
        f"{target}.load_weights",
        before_load_weights,
        HookType.BEFORE,
    )
    HookRegistry.register(
        f"{target}._remap_dspark_weight_name",
        after_remap,
        HookType.AFTER,
    )
