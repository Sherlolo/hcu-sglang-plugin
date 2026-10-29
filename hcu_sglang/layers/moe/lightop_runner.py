"""LightOp MoE runner integration for HCU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
from torch.nn.parameter import Parameter

from hcu_sglang.utils.detection import is_dcu

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

LIGHTOP_BACKEND_NAME = "lightop"


def register_lightop_backend():
    from sglang.srt.layers.moe.utils import MoeRunnerBackend

    existing = MoeRunnerBackend.__members__.get("LIGHTOP")
    if existing is not None:
        if not hasattr(MoeRunnerBackend, "is_lightop"):
            type.__setattr__(
                MoeRunnerBackend,
                "is_lightop",
                lambda self: self is existing,
            )
        return existing

    member = object.__new__(MoeRunnerBackend)
    member._name_ = "LIGHTOP"
    member._value_ = LIGHTOP_BACKEND_NAME
    MoeRunnerBackend._member_names_.append("LIGHTOP")
    MoeRunnerBackend._member_map_["LIGHTOP"] = member
    MoeRunnerBackend._value2member_map_[LIGHTOP_BACKEND_NAME] = member
    type.__setattr__(MoeRunnerBackend, "LIGHTOP", member)
    type.__setattr__(
        MoeRunnerBackend,
        "is_lightop",
        lambda self: self is member,
    )
    return member


def register_lightop_cli_choice() -> None:
    from sglang.srt import server_args

    if LIGHTOP_BACKEND_NAME not in server_args.MOE_RUNNER_BACKEND_CHOICES:
        server_args.add_moe_runner_backend_choices([LIGHTOP_BACKEND_NAME])


def _is_lightop_backend(backend: Any) -> bool:
    return getattr(backend, "value", None) == LIGHTOP_BACKEND_NAME


def around_moe_runner_init(
    original: Callable[..., Any],
    runner,
    runner_backend,
    config,
    lora_enabled: bool = False,
) -> None:
    if not _is_lightop_backend(runner_backend):
        return original(runner, runner_backend, config, lora_enabled)
    if not is_dcu():
        raise ValueError("The lightop MoE runner is only available on DCU/HCU.")

    from sglang.srt.layers.moe.moe_runner.base import FusedOpPool
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    runner.runner_backend = runner_backend
    runner.config = config
    runner.lora_enabled = lora_enabled
    runner.runner_core = LightOpRunnerCore(config)
    runner.fused_func = None
    if not lora_enabled:
        runner.fused_func = FusedOpPool.get_fused_func(
            get_moe_a2a_backend().value,
            LIGHTOP_BACKEND_NAME,
        )
    runner.down_gemm_overlap_args = None
    runner.meta_overlap_args = None


def weight8bit_nt_kpack2_marlin1(
    weight: torch.Tensor,
    k_tile: int = 16,
    k_tile1: int = 4,
    n_tile: int = 16,
    n_tile1: int = 16,
) -> torch.Tensor:
    if weight.element_size() != 1:
        raise ValueError("LightOp Marlin weights must use an 8-bit dtype")
    if weight.dim() == 2:
        size_n, size_k = weight.shape
        if size_n % (n_tile * n_tile1) or size_k % (k_tile * k_tile1):
            raise ValueError("Weight shape is not aligned for LightOp Marlin")
        return (
            weight.reshape(
                size_n // (n_tile * n_tile1),
                n_tile1,
                n_tile,
                size_k // (k_tile * k_tile1),
                k_tile1,
                k_tile,
            )
            .permute(0, 3, 1, 4, 2, 5)
            .contiguous()
        )
    if weight.dim() == 3:
        num_experts, size_n, size_k = weight.shape
        if size_n % (n_tile * n_tile1) or size_k % (k_tile * k_tile1):
            raise ValueError("Weight shape is not aligned for LightOp Marlin")
        return (
            weight.reshape(
                num_experts,
                size_n // (n_tile * n_tile1),
                n_tile1,
                n_tile,
                size_k // (k_tile * k_tile1),
                k_tile1,
                k_tile,
            )
            .permute(0, 1, 4, 2, 5, 3, 6)
            .contiguous()
        )
    raise ValueError("LightOp Marlin weights must be rank 2 or rank 3")


@dataclass
class LightOpRunnerInput:
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor

    @property
    def runner_backend(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        return MoeRunnerBackend.LIGHTOP


@dataclass
class LightOpRunnerOutput:
    hidden_states: torch.Tensor

    @property
    def runner_backend(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        return MoeRunnerBackend.LIGHTOP


@dataclass
class LightOpMoeQuantInfo:
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    use_fp8_w8a8: bool = False
    use_int8_w8a8: bool = True
    origin_w13_shape: Optional[torch.Size] = None
    origin_w2_shape: Optional[torch.Size] = None
    deepep_w13_weight: Optional[torch.Tensor] = None
    deepep_w2_weight: Optional[torch.Tensor] = None


def _is_moe_prefill_or_normal() -> bool:
    from sglang.srt.runtime_context import get_server_args

    server_args = get_server_args()
    return (
        server_args.disaggregation_mode == "prefill"
        or server_args.deepep_mode == "normal"
    )


def get_lightop_marlin_weight(
    weight: torch.Tensor, *, use_fp8_w8a8: bool = False
) -> torch.Tensor:
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    if get_moe_a2a_backend().is_deepep():
        if use_fp8_w8a8 and _is_moe_prefill_or_normal():
            from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
                weight8bit_nt_kpack2_marlin,
            )

            return weight8bit_nt_kpack2_marlin(weight)
        return weight8bit_nt_kpack2_marlin1(weight)

    if use_fp8_w8a8:
        from hcu_sglang.layers.quantization.lightop_int8_marlin import (
            w8a8_2_marlin_weight,
        )

        return w8a8_2_marlin_weight(weight)

    from hcu_sglang.layers.quantization.lightop_int8_marlin import (
        get_w8a8_int8_marlin_weights,
    )

    return get_w8a8_int8_marlin_weights(weight)


def process_weights_after_loading_lightop(
    layer: torch.nn.Module, *, use_fp8_w8a8: bool = False
) -> None:
    origin_w13_shape = layer.w13_weight.shape
    origin_w2_shape = layer.w2_weight.shape
    from sglang.srt.layers.moe import get_moe_a2a_backend

    if get_moe_a2a_backend().is_deepep():
        layer._lightop_deepep_w13_weight = layer.w13_weight.detach()
        layer._lightop_deepep_w2_weight = layer.w2_weight.detach()
    if use_fp8_w8a8:
        if (
            layer.w13_weight.dim() != 3
            or layer.w2_weight.dim() != 3
            or layer.w13_weight.size(0) != layer.w2_weight.size(0)
        ):
            raise RuntimeError("Unexpected MoE weight shapes")
        two_n, hidden_size = layer.w13_weight.shape[1:]
        if layer.w2_weight.size(1) != hidden_size:
            raise RuntimeError("Unexpected MoE w2 layout")
        intermediate_size = layer.w2_weight.size(2)
        if two_n != 2 * intermediate_size:
            raise RuntimeError("Unexpected MoE hidden dimensions")
        if hidden_size % 32 or intermediate_size % 16 or two_n % 32:
            raise RuntimeError("LightOp Marlin packing requires aligned weights")

    layer.w13_weight = Parameter(
        torch.stack(
            [
                get_lightop_marlin_weight(weight, use_fp8_w8a8=use_fp8_w8a8)
                for weight in layer.w13_weight
            ],
            dim=0,
        ),
        requires_grad=False,
    )
    layer.w2_weight = Parameter(
        torch.stack(
            [
                get_lightop_marlin_weight(weight, use_fp8_w8a8=use_fp8_w8a8)
                for weight in layer.w2_weight
            ],
            dim=0,
        ),
        requires_grad=False,
    )
    layer._lightop_origin_w13_shape = origin_w13_shape
    layer._lightop_origin_w2_shape = origin_w2_shape


def get_lightop_quant_info(
    layer: torch.nn.Module, *, use_fp8_w8a8: bool = False
) -> LightOpMoeQuantInfo:
    return LightOpMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        w13_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        a13_scale=layer.w13_input_scale,
        a2_scale=layer.w2_input_scale,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=not use_fp8_w8a8,
        origin_w13_shape=getattr(layer, "_lightop_origin_w13_shape", None),
        origin_w2_shape=getattr(layer, "_lightop_origin_w2_shape", None),
        deepep_w13_weight=getattr(layer, "_lightop_deepep_w13_weight", None),
        deepep_w2_weight=getattr(layer, "_lightop_deepep_w2_weight", None),
    )


class LightOpRunnerCore:
    def __init__(self, config: MoeRunnerConfig):
        self.config = config

    def run(
        self,
        runner_input: LightOpRunnerInput,
        quant_info: LightOpMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> LightOpRunnerOutput:
        if hooks is not None:
            raise NotImplementedError("LightOp MoE does not support LoRA hooks")
        routed_scaling_factor = self.config.routed_scaling_factor or 1.0

        if quant_info.use_fp8_w8a8:
            if (
                quant_info.origin_w13_shape is None
                or quant_info.origin_w2_shape is None
            ):
                raise RuntimeError("Missing original FP8 MoE weight shapes")
            from hcu_sglang.layers.moe.lightop_fp8 import fused_moe_fp8_w8a8

            output = fused_moe_fp8_w8a8(
                hidden_states=runner_input.hidden_states,
                w1=quant_info.w13_weight,
                w2=quant_info.w2_weight,
                w1_scale=quant_info.w13_scale,
                w2_scale=quant_info.w2_scale,
                topk_weights=runner_input.topk_weights,
                topk_ids=runner_input.topk_ids,
                global_num_experts=self.config.num_experts,
                inplace=self.config.inplace,
                origin_w1_shape=quant_info.origin_w13_shape,
                origin_w2_shape=quant_info.origin_w2_shape,
                routed_scaling_factor=routed_scaling_factor,
            )
        else:
            output = torch.ops.sglang.fused_experts_impl_int8_marlin(
                runner_input.hidden_states,
                quant_info.w13_weight,
                quant_info.w2_weight,
                topk_weights=runner_input.topk_weights,
                topk_ids=runner_input.topk_ids,
                inplace=self.config.inplace,
                activation=self.config.activation,
                apply_router_weight_on_input=self.config.apply_router_weight_on_input,
                use_fp8_w8a8=False,
                use_int8_w8a8=quant_info.use_int8_w8a8,
                per_channel_quant=True,
                global_num_experts=self.config.num_experts,
                w1_scale=quant_info.w13_scale,
                w2_scale=quant_info.w2_scale,
                a1_scale=quant_info.a13_scale,
                a2_scale=quant_info.a2_scale,
                use_nn_moe=False,
                routed_scaling_factor=float(routed_scaling_factor),
            )
        return LightOpRunnerOutput(hidden_states=output)

    @property
    def runner_backend(self):
        from sglang.srt.layers.moe.utils import MoeRunnerBackend

        return MoeRunnerBackend.LIGHTOP


def pre_permute_standard_to_lightop(
    dispatch_output: StandardDispatchOutput,
    quant_info: LightOpMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> LightOpRunnerInput:
    from sglang.srt.layers.moe.topk import apply_topk_weights_cpu

    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    hidden_states, topk_weights = apply_topk_weights_cpu(
        runner_config.apply_router_weight_on_input,
        topk_weights,
        hidden_states,
    )
    return LightOpRunnerInput(hidden_states, topk_weights, topk_ids)


def post_permute_lightop_to_standard(
    runner_output: LightOpRunnerOutput,
    quant_info: LightOpMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(hidden_states=runner_output.hidden_states)


def register_permute_methods() -> None:
    from sglang.srt.layers.moe.moe_runner.base import PermuteMethodPool

    pre_key = ("standard", LIGHTOP_BACKEND_NAME)
    if pre_key not in PermuteMethodPool._pre_permute_methods:
        PermuteMethodPool.register_pre_permute(
            *pre_key,
            pre_permute_standard_to_lightop,
        )
    post_key = (LIGHTOP_BACKEND_NAME, "standard")
    if post_key not in PermuteMethodPool._post_permute_methods:
        PermuteMethodPool.register_post_permute(
            *post_key,
            post_permute_lightop_to_standard,
        )


def register_hooks() -> None:
    register_lightop_backend()
    register_lightop_cli_choice()
    register_permute_methods()
    from hcu_sglang.layers.moe.lightop_deepep import register_fused_func

    register_fused_func()

    from hcu_sglang.layers.quantization import lightop_int8_marlin  # noqa: F401
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    target = "sglang.srt.layers.moe.moe_runner.runner.MoeRunner.__init__"
    registered = HookRegistry._hooks.get(target, ())
    if not any(hook is around_moe_runner_init for _, hook, _ in registered):
        HookRegistry.register(target, around_moe_runner_init, HookType.AROUND)
