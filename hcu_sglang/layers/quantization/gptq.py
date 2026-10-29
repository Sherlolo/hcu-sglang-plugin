"""HCU GPTQ Triton integration for compressed-tensors WNA16."""

from __future__ import annotations

import os

import torch

from hcu_sglang.utils.detection import is_dcu


def _uses_triton(scheme) -> bool:
    return is_dcu() and getattr(scheme, "use_triton_gemm", False)


def after_init(result, scheme, *args, **kwargs):
    if is_dcu():
        scheme.use_triton_gemm = (
            os.getenv("SGLANG_GPTQ_MARLIN_USE_TRITON", "0") == "1"
        )
    return result


def around_process_weights(original, scheme, layer):
    if not _uses_triton(scheme):
        return original(scheme, layer)

    scheme.w_q_name = "weight_packed"
    scheme.w_s_name = "weight_scale"
    scheme.w_zp_name = "weight_zero_point"
    scheme.w_gidx_name = "weight_g_idx"
    device = layer.weight_packed.device
    layer.weight_packed = torch.nn.Parameter(
        layer.weight_packed.data, requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        layer.weight_scale.data, requires_grad=False
    )
    if not scheme.symmetric:
        layer.weight_zero_point = torch.nn.Parameter(
            layer.weight_zero_point.data, requires_grad=False
        )
    if scheme.has_g_idx:
        layer.weight_g_idx = torch.nn.Parameter(
            layer.weight_g_idx.data, requires_grad=False
        )
    layer.g_idx_sort_indices = torch.empty((0,), dtype=torch.int32, device=device)
    scheme.workspace = torch.empty((0,), dtype=torch.int32, device=device)


def around_apply_weights(original, scheme, layer, x, bias):
    if not _uses_triton(scheme):
        return original(scheme, layer, x, bias)

    from hcu_sglang.layers.quantization.gptq_triton import gptq_gemm_triton

    weight = getattr(layer, scheme.w_q_name)
    scale = getattr(layer, scheme.w_s_name)
    zero_point = getattr(layer, scheme.w_zp_name or "", None)
    group_index = getattr(layer, scheme.w_gidx_name or "", None)
    output = gptq_gemm_triton(
        x,
        weight,
        scale,
        scheme.kernel_config.weight_type,
        zero_point,
        group_index,
    )
    return output if bias is None else output + bias


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    target = (
        "sglang.srt.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_wNa16.CompressedTensorsWNA16"
    )
    HookRegistry.register(f"{target}.__init__", after_init, HookType.AFTER)
    HookRegistry.register(
        f"{target}.process_weights_after_loading",
        around_process_weights,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{target}.apply_weights",
        around_apply_weights,
        HookType.AROUND,
    )
