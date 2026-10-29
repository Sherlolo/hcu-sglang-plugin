"""HCU AITER routing for official W8A8 MoE quantization methods."""

from __future__ import annotations

from typing import Any, Callable

from hcu_sglang.utils.detection import is_dcu


def _use_aiter_runner(quant_method) -> bool:
    runner = getattr(quant_method, "runner", None)
    backend = getattr(runner, "runner_backend", None)
    return is_dcu() and backend is not None and backend.is_aiter()


def around_create_moe_runner(
    original: Callable[..., Any], quant_method, layer, moe_runner_config
):
    if not is_dcu():
        return original(quant_method, layer, moe_runner_config)

    from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend, get_moe_runner_backend

    if not get_moe_runner_backend().is_aiter():
        return original(quant_method, layer, moe_runner_config)
    quant_method.moe_runner_config = moe_runner_config
    quant_method.runner = MoeRunner(MoeRunnerBackend.AITER, moe_runner_config)
    return None


def around_apply_int8(original, quant_method, layer, dispatch_output):
    if not _use_aiter_runner(quant_method):
        return original(quant_method, layer, dispatch_output)

    from hcu_sglang.layers.moe.aiter_w4a8 import (
        get_aiter_w8a8_int8_quant_info,
    )

    return quant_method.runner.run(
        dispatch_output, get_aiter_w8a8_int8_quant_info(layer)
    )


def around_apply_fp8(original, quant_method, layer, dispatch_output):
    if not _use_aiter_runner(quant_method):
        return original(quant_method, layer, dispatch_output)

    from hcu_sglang.layers.moe.aiter_w4a8 import (
        get_aiter_w8a8_fp8_quant_info,
    )

    return quant_method.runner.run(
        dispatch_output, get_aiter_w8a8_fp8_quant_info(layer)
    )


def around_apply_compressed_fp8(original, quant_method, layer, dispatch_output):
    if not _use_aiter_runner(quant_method):
        return original(quant_method, layer, dispatch_output)

    from compressed_tensors.quantization import QuantizationStrategy

    if quant_method.weight_quant.strategy != QuantizationStrategy.CHANNEL:
        return original(quant_method, layer, dispatch_output)

    from hcu_sglang.layers.moe.aiter_w4a8 import (
        get_aiter_w8a8_fp8_quant_info,
    )

    return quant_method.runner.run(
        dispatch_output, get_aiter_w8a8_fp8_quant_info(layer)
    )


def around_get_compressed_moe_scheme(
    original: Callable[..., Any], compressed_config, layer, layer_name=None
):
    try:
        return original(compressed_config, layer, layer_name)
    except NotImplementedError as exc:
        if not is_dcu() or "W8A8Int8 Fused MoE" not in str(exc):
            raise
        original_error = exc

    compressed_config._add_fused_moe_to_target_scheme_map()
    unfused_names = [
        layer_name + projection
        for projection in (".0.gate_proj", ".0.up_proj", ".0.down_proj")
    ]
    scheme_dicts = [
        compressed_config.get_scheme_dict(layer, name) for name in unfused_names
    ]
    scheme_dict = scheme_dicts[0] if scheme_dicts else None
    if scheme_dict is None or not all(item == scheme_dict for item in scheme_dicts):
        raise original_error
    weight_quant = scheme_dict.get("weights")
    input_quant = scheme_dict.get("input_activations")
    if not compressed_config._is_dynamic_token_w8a8(weight_quant, input_quant):
        raise original_error

    from hcu_sglang.layers.quantization.compressed_w8a8_int8_moe import (
        CompressedTensorsW8A8Int8MoE,
    )

    return CompressedTensorsW8A8Int8MoE(weight_quant, input_quant)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for module, class_name, apply_hook in (
        (
            "sglang.srt.layers.quantization.w8a8_int8",
            "W8A8Int8MoEMethod",
            around_apply_int8,
        ),
        (
            "sglang.srt.layers.quantization.w8a8_fp8",
            "W8A8FP8MoEMethod",
            around_apply_fp8,
        ),
    ):
        HookRegistry.register(
            f"{module}.{class_name}.create_moe_runner",
            around_create_moe_runner,
            HookType.AROUND,
        )
        HookRegistry.register(
            f"{module}.{class_name}.apply",
            apply_hook,
            HookType.AROUND,
        )
    HookRegistry.register(
        "sglang.srt.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a8_fp8_moe."
        "CompressedTensorsW8A8Fp8MoE.apply_weights",
        around_apply_compressed_fp8,
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.layers.quantization.compressed_tensors.compressed_tensors."
        "CompressedTensorsConfig.get_moe_scheme",
        around_get_compressed_moe_scheme,
        HookType.AROUND,
    )
