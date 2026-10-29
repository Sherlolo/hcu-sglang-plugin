"""HCU LightOp Marlin W16A16 MoE compatibility."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.nn.parameter import Parameter

from hcu_sglang.utils.detection import is_dcu


def _enabled() -> bool:
    from sglang.srt.layers.moe import get_moe_a2a_backend
    from sglang.srt.utils import get_bool_env_var

    return (
        is_dcu()
        and get_bool_env_var("SGLANG_USE_MARLIN_W16A16_MOE")
        and not get_bool_env_var("SGLANG_ROCM_USE_AITER_MOE")
        and not get_moe_a2a_backend().is_deepep()
    )


def _weight_permutation(device: torch.device) -> torch.Tensor:
    permutation = []
    for thread_id in range(64):
        for column_offset in range(2):
            column = (thread_id % 16) * 2 + column_offset
            for row_offset in range(4):
                row = (thread_id // 16) * 4 + row_offset
                permutation.append(row * 32 + column)
    return torch.tensor(permutation, dtype=torch.long, device=device)


def pack_w16a16_marlin_weight(weight: torch.Tensor) -> torch.Tensor:
    transposed = weight.T.contiguous()
    size_k, size_n = transposed.shape
    if size_k % 16 or size_n % 32:
        raise RuntimeError(
            "W16A16 Marlin packing requires K divisible by 16 and N by 32, "
            f"got weight shape={tuple(weight.shape)}"
        )
    packed = transposed.reshape(size_k // 16, 16, size_n // 32, 32)
    packed = packed.permute(0, 2, 1, 3).reshape(size_k // 16, size_n * 16)
    permutation = _weight_permutation(weight.device)
    return packed.reshape(-1, permutation.numel())[:, permutation].reshape(packed.shape)


def _copy_parameter_attrs(source: torch.nn.Parameter, target: torch.nn.Parameter):
    for name, value in getattr(source, "__dict__", {}).items():
        setattr(target, name, value)


def around_process_weights(
    original: Callable[..., Any], method: Any, layer: Any
) -> Any:
    result = original(method, layer)
    if not _enabled() or getattr(layer, "_marlin_w16a16_moe_packed", False):
        return result
    if getattr(layer, "use_nn_moe", False):
        return result

    original_w13 = layer.w13_weight
    original_w2 = layer.w2_weight
    intermediate_size = original_w2.shape[2]
    with torch.no_grad():
        packed_w13 = torch.stack(
            [pack_w16a16_marlin_weight(weight) for weight in original_w13]
        )
        packed_w2 = torch.stack(
            [pack_w16a16_marlin_weight(weight) for weight in original_w2]
        )
    layer.w13_weight = Parameter(packed_w13, requires_grad=False)
    layer.w2_weight = Parameter(packed_w2, requires_grad=False)
    _copy_parameter_attrs(original_w13, layer.w13_weight)
    _copy_parameter_attrs(original_w2, layer.w2_weight)
    layer.w13_weight.marlin_w16a16_packed = True
    layer.w13_weight.N = intermediate_size
    layer.w2_weight.marlin_w16a16_packed = True
    layer._marlin_w16a16_moe_packed = True
    return result


def fused_marlin_w16a16(
    hidden_states: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    num_experts: int,
    intermediate_size: int,
    routed_scaling_factor: float,
    inplace: bool,
) -> torch.Tensor:
    from lightop.activation import fuse_silu_and_mul
    from lightop.moe import (
        get_moe_cuda_marlin_config_w16a16,
        moe_gemm_marlin_w16a16,
    )
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )

    num_tokens, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    properties = torch.cuda.get_device_properties(hidden_states.device)
    device_name = getattr(properties, "gcnArchName", properties.name).split(":")[0]
    config_up, config_down, _ = get_moe_cuda_marlin_config_w16a16(
        w13_weight.shape[0],
        num_tokens,
        2 * intermediate_size,
        hidden_size,
        hidden_size,
        intermediate_size,
        top_k,
        device_name,
        properties.multi_processor_count,
        hidden_states.dtype,
    )
    sorted_ids, expert_ids, post_padded = moe_align_block_size(
        topk_ids,
        config_up["BLOCK_SIZE_M"],
        num_experts,
    )
    gate_up = torch.empty(
        (num_tokens * top_k, 2 * intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    activated = torch.empty(
        (num_tokens * top_k, intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    expert_output = torch.empty(
        (num_tokens * top_k, hidden_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    moe_gemm_marlin_w16a16(
        hidden_states,
        w13_weight,
        gate_up,
        None,
        sorted_ids,
        expert_ids,
        post_padded,
        top_k,
        config_up,
    )
    fuse_silu_and_mul(gate_up, activated)
    moe_gemm_marlin_w16a16(
        activated,
        w2_weight,
        expert_output,
        topk_weights,
        sorted_ids,
        expert_ids,
        post_padded,
        1,
        config_down,
    )
    output = expert_output.view(num_tokens, top_k, hidden_size).sum(dim=1)
    output.mul_(routed_scaling_factor)
    if inplace:
        hidden_states.copy_(output)
        return hidden_states
    return output


def around_forward_cuda(
    original: Callable[..., Any], method: Any, layer: Any, dispatch_output: Any
) -> Any:
    if not _enabled() or not getattr(layer, "_marlin_w16a16_moe_packed", False):
        return original(method, layer, dispatch_output)

    from hcu_sglang.compat.aiter import install_aiter_compat

    install_aiter_compat()
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    topk_weights, topk_ids, _ = dispatch_output.topk_output
    output = fused_marlin_w16a16(
        hidden_states=dispatch_output.hidden_states,
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        num_experts=method.moe_runner_config.num_experts,
        intermediate_size=layer.w13_weight.N,
        routed_scaling_factor=method.moe_runner_config.routed_scaling_factor or 1.0,
        inplace=method.moe_runner_config.inplace,
    )
    return StandardCombineInput(hidden_states=output)


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    target = "sglang.srt.layers.quantization.unquant.UnquantizedFusedMoEMethod"
    HookRegistry.register(
        f"{target}.process_weights_after_loading",
        around_process_weights,
        HookType.AROUND,
    )
    HookRegistry.register(
        f"{target}.forward_cuda",
        around_forward_cuda,
        HookType.AROUND,
    )
