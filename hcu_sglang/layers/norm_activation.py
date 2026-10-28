"""DCU norm and activation adaptations."""

from __future__ import annotations

from typing import Any, Callable

import torch

from hcu_sglang.utils.detection import is_dcu


def after_silu_and_mul_init(result: Any, layer, *args, **kwargs) -> Any:
    if is_dcu() and getattr(layer._forward_method, "__name__", "") == "forward_aiter":
        layer._forward_method = layer.forward_cuda
    return result


def around_silu_and_mul_forward_cuda(
    original: Callable[..., torch.Tensor], layer, x: torch.Tensor
) -> torch.Tensor:
    if not is_dcu():
        return original(layer, x)

    from lightop import fuse_silu_and_mul

    output_shape = x.shape[:-1] + (x.shape[-1] // 2,)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    fuse_silu_and_mul(x, output)
    return output


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    target = "sglang.srt.layers.activation.SiluAndMul"
    HookRegistry.register(
        f"{target}.__init__",
        after_silu_and_mul_init,
        HookType.AFTER,
    )
    HookRegistry.register(
        f"{target}.forward_cuda",
        around_silu_and_mul_forward_cuda,
        HookType.AROUND,
    )
