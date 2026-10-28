"""Route W8A8 MoE quantization methods through the LightOp runner."""

from __future__ import annotations

from typing import Any, Callable

from hcu_sglang.utils.detection import is_dcu

LIGHTOP_BACKEND_NAME = "lightop"


def _backend_is_lightop(backend: Any) -> bool:
    return getattr(backend, "value", None) == LIGHTOP_BACKEND_NAME


def _requested_lightop() -> bool:
    if not is_dcu():
        return False
    from sglang.srt.layers.moe import get_moe_runner_backend

    return _backend_is_lightop(get_moe_runner_backend())


def _uses_lightop_runner(quant_method: Any) -> bool:
    runner = getattr(quant_method, "runner", None)
    return is_dcu() and _backend_is_lightop(getattr(runner, "runner_backend", None))


def around_create_moe_runner(
    original: Callable[..., Any], quant_method, layer, moe_runner_config
) -> None:
    if not _requested_lightop():
        return original(quant_method, layer, moe_runner_config)

    from hcu_sglang.layers.moe.lightop_runner import register_lightop_backend

    register_lightop_backend()
    from sglang.srt.layers.moe import MoeRunner, MoeRunnerBackend

    quant_method.moe_runner_config = moe_runner_config
    quant_method.runner = MoeRunner(MoeRunnerBackend.LIGHTOP, moe_runner_config)


def _process_lightop_weights(original, quant_method, layer, *, use_fp8_w8a8: bool):
    if not _requested_lightop():
        return original(quant_method, layer)

    from torch.nn.parameter import Parameter

    from hcu_sglang.layers.moe.lightop_runner import (
        process_weights_after_loading_lightop,
    )

    process_weights_after_loading_lightop(layer, use_fp8_w8a8=use_fp8_w8a8)
    layer.w13_weight_scale = Parameter(
        layer.w13_weight_scale.data,
        requires_grad=False,
    )
    layer.w2_weight_scale = Parameter(
        layer.w2_weight_scale.data,
        requires_grad=False,
    )


def around_process_weights_int8(original, quant_method, layer):
    return _process_lightop_weights(
        original,
        quant_method,
        layer,
        use_fp8_w8a8=False,
    )


def around_process_weights_fp8(original, quant_method, layer):
    return _process_lightop_weights(
        original,
        quant_method,
        layer,
        use_fp8_w8a8=True,
    )


def _apply_lightop(
    original,
    quant_method,
    layer,
    dispatch_output,
    *,
    use_fp8_w8a8: bool,
):
    if not _uses_lightop_runner(quant_method):
        return original(quant_method, layer, dispatch_output)

    from hcu_sglang.layers.moe.lightop_runner import get_lightop_quant_info

    return quant_method.runner.run(
        dispatch_output,
        get_lightop_quant_info(layer, use_fp8_w8a8=use_fp8_w8a8),
    )


def around_apply_int8(original, quant_method, layer, dispatch_output):
    return _apply_lightop(
        original,
        quant_method,
        layer,
        dispatch_output,
        use_fp8_w8a8=False,
    )


def around_apply_fp8(original, quant_method, layer, dispatch_output):
    return _apply_lightop(
        original,
        quant_method,
        layer,
        dispatch_output,
        use_fp8_w8a8=True,
    )


def around_compressed_process_weights(original, quant_method, layer):
    result = original(quant_method, layer)
    if not _requested_lightop():
        return result

    from compressed_tensors.quantization import QuantizationStrategy

    if quant_method.weight_quant.strategy != QuantizationStrategy.CHANNEL:
        raise ValueError("LightOp compressed FP8 MoE requires channelwise weights")
    from hcu_sglang.layers.moe.lightop_runner import (
        process_weights_after_loading_lightop,
    )

    process_weights_after_loading_lightop(layer, use_fp8_w8a8=True)
    return result


def around_compressed_apply(original, quant_method, layer, dispatch_output):
    if not _uses_lightop_runner(quant_method):
        try:
            return original(quant_method, layer, dispatch_output)
        except RuntimeError as error:
            from hcu_sglang.layers.moe.dspark_aiter_fallback import (
                is_missing_aiter_backend_error,
                try_run_dspark_triton_fallback,
            )

            if not is_dcu() or not is_missing_aiter_backend_error(error):
                raise
            fallback_output = try_run_dspark_triton_fallback(
                quant_method,
                layer,
                dispatch_output,
            )
            if fallback_output is None:
                raise
            return fallback_output

    from compressed_tensors.quantization import QuantizationStrategy

    if quant_method.weight_quant.strategy != QuantizationStrategy.CHANNEL:
        return original(quant_method, layer, dispatch_output)
    from hcu_sglang.layers.moe.lightop_runner import get_lightop_quant_info

    return quant_method.runner.run(
        dispatch_output,
        get_lightop_quant_info(layer, use_fp8_w8a8=True),
    )


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    hook_specs = (
        (
            "sglang.srt.layers.quantization.w8a8_int8.W8A8Int8MoEMethod",
            around_process_weights_int8,
            around_apply_int8,
        ),
        (
            "sglang.srt.layers.quantization.w8a8_fp8.W8A8FP8MoEMethod",
            around_process_weights_fp8,
            around_apply_fp8,
        ),
    )
    for target, process_hook, apply_hook in hook_specs:
        HookRegistry.register(
            f"{target}.create_moe_runner",
            around_create_moe_runner,
            HookType.AROUND,
        )
        HookRegistry.register(
            f"{target}.process_weights_after_loading",
            process_hook,
            HookType.AROUND,
        )
        HookRegistry.register(
            f"{target}.apply",
            apply_hook,
            HookType.AROUND,
        )

    compressed_target = (
        "sglang.srt.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w8a8_fp8_moe.CompressedTensorsW8A8Fp8MoE"
    )
    HookRegistry.register(
        f"{compressed_target}.create_moe_runner",
        around_create_moe_runner,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{compressed_target}.process_weights_after_loading",
        around_compressed_process_weights,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{compressed_target}.apply_weights",
        around_compressed_apply,
        HookType.AROUND,
    )
