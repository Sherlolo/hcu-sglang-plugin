# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/quantization/fp8.py

from __future__ import annotations
import torch
from sglang.srt.layers.quantization import fp8 as P
from hcu_sglang.utils.detection import is_hcu

_is_hcu = is_hcu()
_is_shuffle_moe_mxfp4 = P._is_gfx95_supported and not _is_hcu
SGLANG_USE_AITER_FP8_ASM_MOE = False


def asm_shuffle_weight_b8(weight, which):
    from aiter.ops.shuffle import asm_shuffle_weight_b8 as provider
    return provider(weight, which)

def create_fp8_moe_weight_(
    layer: P.Module,
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_partition: int,
    block_quant: bool,
    quant_config,
    use_mxfp8: bool,
    is_checkpoint_fp8_serialized: bool,
    is_fp4_expert: bool,
    params_dtype: torch.dtype,
    with_bias: bool = False,
    fp4_scale_dtype: P.Optional[torch.dtype] = None,
    **extra_weight_attrs,
):
    """
        Registers weights into `layer`. This static method can be reused by other quantization methods that require loading FP8 checkpoints first (e.g. requantization to other formats as MXFP4).
        """
    from sglang.srt.layers.moe.fused_moe_triton import FusedMoeWeightScaleSupported

    if is_checkpoint_fp8_serialized:
        params_dtype = torch.uint32 if P._use_hip_int4 else torch.float8_e4m3fn

    tp_size = layer.moe_tp_size
    w13_num_shards = 2 if layer.moe_runner_config.is_gated else 1

    w13_up_dim, w2_up_dim, weight_padded = P.get_moe_weight_sizes(
        intermediate_size_per_partition,
        is_aiter_moe=P._use_aiter,
        is_concat=layer.moe_runner_config.is_gated,
        is_packed=False,
    )

    if block_quant:
        block_n, block_k = (
            quant_config.weight_block_size[0],
            quant_config.weight_block_size[1],
        )

        padding_size = P.get_moe_padding_size(P._use_aiter)
        if not (P._use_aiter and padding_size == block_n == block_k):
            # NOTE(HandH1998): To ensure proper alignment of the block-wise quantization scales, the output_size of the weights for both the gate and up layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1:
                # Required by row parallel
                if intermediate_size_per_partition % block_k != 0:
                    raise ValueError(
                        f"The input_size of down's weight = "
                        f"{intermediate_size_per_partition} is not divisible by "
                        f"weight quantization block_k = {block_k}."
                    )

    # WEIGHTS
    if is_fp4_expert:
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 2,
                dtype=torch.int8,
            ),
            requires_grad=False,
        )
    elif P._is_hip and P._use_hip_int4:
        # INT4 MoE weight - INT32 packed
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size // 8,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // 8,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
    else:
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_up_dim,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                w2_up_dim,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )

    extra_weight_attrs.update(
        {"weight_padded": weight_padded},
    )

    layer.register_parameter("w13_weight", w13_weight)
    P.set_weight_attrs(w13_weight, extra_weight_attrs)

    layer.register_parameter("w2_weight", w2_weight)
    P.set_weight_attrs(w2_weight, extra_weight_attrs)

    # BIAS (optional, e.g. GPT-OSS)
    if with_bias:
        w13_weight_bias = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_bias", w13_weight_bias)
        P.set_weight_attrs(w13_weight_bias, extra_weight_attrs)

        w2_weight_bias = torch.nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_bias", w2_weight_bias)
        P.set_weight_attrs(w2_weight_bias, extra_weight_attrs)

    # WEIGHT_SCALES
    if is_fp4_expert:
        fp4_block_k = 32
        if fp4_scale_dtype is None:
            fp4_scale_dtype = (
                torch.float8_e8m0fnu if P._use_aiter or P.is_xpu() else torch.float32
            )
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size // fp4_block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // fp4_block_k,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)
    elif block_quant:
        scale_dtype = torch.uint8 if use_mxfp8 else torch.float32
        scale_init = torch.zeros if scale_dtype == torch.uint8 else torch.ones
        w13_weight_scale = torch.nn.Parameter(
            scale_init(
                num_experts,
                w13_num_shards
                * ((intermediate_size_per_partition + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            scale_init(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        # w13_weight and w2_weight are always requanted together
        w13_weight_scale.format_ue8m0 = use_mxfp8
        w2_weight_scale.format_ue8m0 = use_mxfp8
        layer.register_parameter("w13_weight_scale_inv", w13_weight_scale)
        layer.register_parameter("w2_weight_scale_inv", w2_weight_scale)

        assert quant_config.activation_scheme == "dynamic"

    else:
        # One scale per w13 shard; a gated layer combines its two into a
        # single scale after weight loading.
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, w13_num_shards, dtype=torch.float32),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        if P._is_hip:  # _use_aiter: TODO: add check back after triton kernel
            # ROCm - using column scaling, duplicate scaling numbers in case per tensor scaling
            w13_weight_scale1 = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale1 = torch.nn.Parameter(
                torch.ones(num_experts, hidden_size, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale1", w13_weight_scale1)
            layer.register_parameter("w2_weight_scale1", w2_weight_scale1)

    # Add the quantization method used (per tensor/grouped/channel)
    # to ensure the weight scales are loaded in properly
    extra_weight_attrs.update(
        {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
        if block_quant
        else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
    )

    # If loading fp8 checkpoint, pass the weight loaders.
    # If loading an fp16 checkpoint, do not (we will quantize in
    #   process_weights_after_loading()
    if quant_config.is_checkpoint_fp8_serialized:
        P.set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        P.set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        if P._is_hip and P._use_hip_int4:
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
            )
            P.set_weight_attrs(w13_weight_scale1, extra_weight_attrs)
            P.set_weight_attrs(w2_weight_scale1, extra_weight_attrs)

    # INPUT_SCALES
    if quant_config.activation_scheme == "static":
        if not quant_config.is_checkpoint_fp8_serialized:
            raise ValueError(
                "Found static activation scheme for checkpoint that "
                "was not serialized fp8."
            )

        w13_input_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w13_input_scale", w13_input_scale)
        P.set_weight_attrs(w13_input_scale, extra_weight_attrs)

        w2_input_scale = torch.nn.Parameter(
            torch.ones(num_experts, dtype=torch.float32), requires_grad=False
        )
        layer.register_parameter("w2_input_scale", w2_input_scale)
        P.set_weight_attrs(w2_input_scale, extra_weight_attrs)

    else:
        layer.w13_input_scale = None
        layer.w2_input_scale = None

def maybe_get_hip_aiter_quant_info(
    self,
    layer: torch.nn.Module,
    no_combine: bool = False,
) -> P.Optional[AiterMoeQuantInfo]:
    # HCU keeps its validated native/ASM AITER dispatch path. The official
    # runner-core quant-info path below is for generic ROCm.
    if _is_hcu or not (P._use_aiter or P._use_hip_int4):
        return None
    assert not no_combine, f"{no_combine=} is not supported."

    from sglang.srt.layers.moe.moe_runner.aiter import (
        AiterMoeQuantInfo,
        AiterQuantType,
    )

    w13_weight = layer.w13_weight
    w2_weight = layer.w2_weight

    if self.use_mxfp8:
        gemm1_alpha = self.moe_runner_config.gemm1_alpha
        if gemm1_alpha != 1.702:
            raise NotImplementedError(
                f"AITER MXFP8 MoE only supports swiglu-oai "
                f"alpha=1.702, got {gemm1_alpha=}."
            )
        from aiter import ActivationType
        from aiter.ops.flydsl.moe_common import GateMode

        return AiterMoeQuantInfo(
            w13_weight=w13_weight,
            w2_weight=w2_weight,
            quant_type=AiterQuantType.PER_1X32,
            w13_scale=layer.w13_weight_scale_inv,
            w2_scale=layer.w2_weight_scale_inv,
            expert_mask=layer.dispatcher.expert_mask_gpu if P._use_aiter else None,
            swiglu_limit=self.moe_runner_config.swiglu_limit
            or self.moe_runner_config.gemm1_clamp_limit
            or 0.0,
            hidden_pad=getattr(layer, "hidden_pad", 0),
            intermediate_pad=getattr(layer, "intermediate_pad", 0),
            fused_moe_kwargs={
                "activation": ActivationType.Swiglu,
                "gate_mode": GateMode.INTERLEAVE.value,
            },
        )

    if self.block_quant:
        quant_type = (
            AiterQuantType.PER_1X32
            if self.is_fp4_expert
            else AiterQuantType.PER_128X128
        )
        if self.is_fp4_expert:
            fp4_weight_dtype = P._require_fp4_dtype()
            w13_weight = w13_weight.view(fp4_weight_dtype)
            w2_weight = w2_weight.view(fp4_weight_dtype)
            if getattr(layer.w13_weight, "is_shuffled", False):
                w13_weight.is_shuffled = True
                w2_weight.is_shuffled = True
        w13_scale = layer.w13_weight_scale_inv
        w2_scale = layer.w2_weight_scale_inv
    else:
        quant_type = AiterQuantType.PER_TOKEN
        w13_scale = layer.w13_weight_scale1
        w2_scale = layer.w2_weight_scale1

    fused_moe_kwargs = None
    gate_up_interleaved = getattr(layer, "_aiter_gate_up_interleaved", None)
    if (
        gate_up_interleaved is not None
        and (self.moe_runner_config.swiglu_limit or 0.0) > 0
    ):
        from aiter.ops.flydsl.moe_common import GateMode

        fused_moe_kwargs = {
            "gate_mode": (
                GateMode.INTERLEAVE.value
                if gate_up_interleaved
                else GateMode.SEPARATED.value
            )
        }

    return AiterMoeQuantInfo(
        w13_weight=w13_weight,
        w2_weight=w2_weight,
        quant_type=quant_type,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
        expert_mask=layer.dispatcher.expert_mask_gpu if P._use_aiter else None,
        swiglu_limit=self.moe_runner_config.swiglu_limit or 0.0,
        hidden_pad=getattr(layer, "hidden_pad", 0),
        intermediate_pad=getattr(layer, "intermediate_pad", 0),
        fused_moe_kwargs=fused_moe_kwargs,
    )

def process_weights_after_loading(self, layer: P.Module) -> None:
    if _is_hcu and P.get_moe_a2a_backend().is_megamoe():
        from sglang.srt.layers.moe.mega_moe import (
            build_hcu_w8a8_mega_moe_experts_weights,
            get_hcu_mega_moe_runtime,
        )

        runtime = get_hcu_mega_moe_runtime()
        if runtime == "megamoe" and (
            self.block_quant
            or self.is_fp4_expert
            or not self.quant_config.is_checkpoint_fp8_serialized
        ):
            raise ValueError(
                "standalone megamoe requires an FP8-serialized, "
                "channelwise W8A8 checkpoint; blockwise, per-tensor, "
                "non-serialized, and FP4 expert formats are unsupported"
            )
        if not self.block_quant and not self.is_fp4_expert:
            if not self.quant_config.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "HCU W8A8 MegaMoE requires an FP8-serialized checkpoint "
                    "with channelwise expert weight scales"
                )
            build_hcu_w8a8_mega_moe_experts_weights(layer)
            return

    if P._is_hip and P._use_hip_int4:
        self.process_weights_hip_int4(layer)

    elif self.block_quant:
        # Block quant doesn't need to process weights after loading
        self.process_weights_after_loading_block_quant(layer)

    # If checkpoint is fp16 or bfloat16, quantize in place.
    elif not self.quant_config.is_checkpoint_fp8_serialized:
        # If ROCm, fp8_dtype will be float8_e4m3fnuz (MI300x HW)
        w13_weight = torch.empty_like(layer.w13_weight.data, dtype=P.fp8_dtype)
        w2_weight = torch.empty_like(layer.w2_weight.data, dtype=P.fp8_dtype)

        # Re-initialize w13_scale because we directly quantize
        # merged w13 weights and generate a single scaling factor.
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                layer.num_local_experts,
                dtype=torch.float32,
                device=w13_weight.device,
            ),
            requires_grad=False,
        )
        for expert in range(layer.num_local_experts):
            w13_weight[expert, :, :], layer.w13_weight_scale[expert] = (
                P.scaled_fp8_quant(layer.w13_weight.data[expert, :, :])
            )
            w2_weight[expert, :, :], layer.w2_weight_scale[expert] = (
                P.scaled_fp8_quant(layer.w2_weight.data[expert, :, :])
            )
        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)

        if P._is_hip:
            self.process_weights_hip_scale_padding(layer)

    # If checkpoint is fp8, we need to handle that the
    # MoE kernels require single activation scale and single weight
    # scale for w13 per expert.
    else:
        # Fp8 moe kernels require a single activation scale.
        # We take the max of all the scales in case they differ.
        if self.quant_config.activation_scheme == "static":
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )
            if not P.all_close_1d(layer.w13_input_scale) or not P.all_close_1d(
                layer.w2_input_scale
            ):
                P.print_warning_once(
                    "Found input_scales that are not equal for "
                    "fp8 MoE layer. Using the maximum across experts "
                    "for each layer. "
                )
            layer.w13_input_scale = torch.nn.Parameter(
                layer.w13_input_scale.max(), requires_grad=False
            )
            layer.w2_input_scale = torch.nn.Parameter(
                layer.w2_input_scale.max(), requires_grad=False
            )

        # If ROCm, normalize the weights and scales to e4m3fnuz
        if P._is_fp8_fnuz:
            # Normalize the weights and scales
            w13_weight, w13_weight_scale, w13_input_scale = (
                P.normalize_e4m3fn_to_e4m3fnuz(
                    layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                )
            )
            w2_weight, w2_weight_scale, w2_input_scale = (
                P.normalize_e4m3fn_to_e4m3fnuz(
                    layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
                )
            )
            # Reset the parameter
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(
                w13_weight_scale, requires_grad=False
            )
            if w13_input_scale is not None:
                layer.w13_input_scale = torch.nn.Parameter(
                    w13_input_scale, requires_grad=False
                )
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_weight_scale, requires_grad=False
            )
            if w2_input_scale is not None:
                layer.w2_input_scale = torch.nn.Parameter(
                    w2_input_scale, requires_grad=False
                )
        # Fp8 moe kernel needs single weight scale for w13 per expert.
        # We take the max then dequant and requant each expert.
        assert layer.w13_weight_scale is not None
        w13_num_shards = 2 if layer.moe_runner_config.is_gated else 1
        shard_size = layer.intermediate_size_per_partition
        max_w13_scales = layer.w13_weight_scale.max(dim=1).values
        # A single shard already carries one scale per expert; nothing to fuse.
        if w13_num_shards > 1:
            for expert_id in range(layer.num_local_experts):
                start = 0
                for shard_id in range(w13_num_shards):
                    dq_weight = P.per_tensor_dequantize(
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id],
                    )
                    (
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        _,
                    ) = P.scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])
                    start += shard_size

        layer.w13_weight_scale = torch.nn.Parameter(
            max_w13_scales, requires_grad=False
        )

        if P._is_hip:
            self.process_weights_hip_scale_padding(layer)

        # Align FP8 weights to FlashInfer per-tensor kernel layout if enabled
        if (
            P.get_moe_runner_backend().is_flashinfer_trtllm()
            or P.get_moe_runner_backend().is_flashinfer_trtllm_routed()
        ):
            from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
                align_fp8_moe_weights_for_flashinfer_trtllm,
            )

            align_fp8_moe_weights_for_flashinfer_trtllm(layer)

    # The runner backend is global, so it is also true for a borrowed delegate,
    # which has no moe_runner_config and whose kernel ignores these params.
    if (
        P.get_moe_runner_backend().is_flashinfer_trtllm()
        or P.get_moe_runner_backend().is_flashinfer_trtllm_routed()
    ) and self._owns_moe_runner:
        self._prepare_flashinfer_trtllm_activation_params(layer)

    if P.get_moe_runner_backend().is_hpc_ops():
        self._prepare_hpc_ops_weights(layer)

    if hasattr(layer, "dispatcher"):
        layer.dispatcher.set_quant_config(
            {
                "weight_dtype": layer.w13_weight.dtype,
                "use_mxfp8": self.use_mxfp8,
            }
        )

def process_weights_after_loading_block_quant(self, layer: P.Module) -> None:
    # HCU has no aiter MXFP4 MoE kernel: it enters here only for the DSV4
    # FP4 -> FP8 dequant and otherwise keeps the generic path.
    if (
        P._use_aiter
        and self.is_fp4_expert
        and (not _is_hcu or self.dequant_fp4_to_fp8)
    ):
        # aiter MegaMoEv2 builds from the packed FP4 layout, so it has to
        # claim the experts before the native-MXFP4 and dequant arms.
        if (
            P.get_moe_a2a_backend().is_megamoe()
            and P.envs.SGLANG_AMD_USE_FLYDSL_MEGA_MOE.get()
        ):
            if self.dequant_fp4_to_fp8:
                raise ValueError(
                    "SGLANG_DSV4_FP4_DEQUANT strips the packed FP4 layout that "
                    "aiter MegaMoEv2 builds from; unset it, or pick another "
                    "--moe-a2a-backend."
                )
            from sglang.srt.layers.moe.mega_moe import (
                build_mega_moe_experts_weights,
            )

            fp4_weight_dtype = P._require_fp4_dtype()
            layer.w13_weight.data = layer.w13_weight.data.view(fp4_weight_dtype)
            layer.w2_weight.data = layer.w2_weight.data.view(fp4_weight_dtype)
            build_mega_moe_experts_weights(layer)
            return

        # AMD FP4 experts: use aiter's native MXFP4 MoE path.
        elif not self.dequant_fp4_to_fp8:
            gu_intv = P.envs.SGLANG_USE_AITER_MOE_GU_ITLV.get()
            fp4_weight_dtype = P._require_fp4_dtype()

            # DeepSeek V4 MoE is implemented by the FlyDSL kernel, which supports
            # tile_k=128, so we only need to pad dim to 128. This lets DeepSeek-V4-Pro
            # at TP8 skip padding 384 -> 512, reducing routed-expert memory by ~25%.
            # shuffle_scale also supports non-256 shapes since aiter PR#4130.
            fp4_k_align = 128
            E, w13_N, w13_K_packed = layer.w13_weight.shape
            _, w2_N, w2_K_packed = layer.w2_weight.shape
            inter_per_part = w13_N // 2
            padded_inter = (
                (inter_per_part + fp4_k_align - 1) // fp4_k_align * fp4_k_align
            )
            # Record the padding so fused_moe is told the real intermediate size
            # (aiter fused_moe needs intermediate_pad = padded - real; ATOM passes
            # 128, SGLang previously defaulted to 0 -> computed the padded region).
            layer.intermediate_pad = padded_inter - inter_per_part
            layer.hidden_pad = 0
            if padded_inter != inter_per_part:
                pad_amount = padded_inter - inter_per_part
                fp4_block_k = 32

                # Pad w13_weight: (E, 2*inter, K_packed) → (E, 2*padded, K_packed)
                old_w13 = layer.w13_weight.data
                new_w13 = torch.zeros(
                    E,
                    2 * padded_inter,
                    w13_K_packed,
                    dtype=old_w13.dtype,
                    device=old_w13.device,
                )
                new_w13[:, :inter_per_part, :] = old_w13[:, :inter_per_part, :]
                new_w13[:, padded_inter : padded_inter + inter_per_part, :] = (
                    old_w13[:, inter_per_part:, :]
                )
                layer.w13_weight = torch.nn.Parameter(new_w13, requires_grad=False)

                # Pad w2_weight: (E, N, inter_packed) → (E, N, padded_packed)
                old_w2 = layer.w2_weight.data
                new_w2 = torch.zeros(
                    E,
                    w2_N,
                    padded_inter // 2,
                    dtype=old_w2.dtype,
                    device=old_w2.device,
                )
                new_w2[:, :, :w2_K_packed] = old_w2
                layer.w2_weight = torch.nn.Parameter(new_w2, requires_grad=False)

                # Pad w13 scale: (E, 2*inter, K/block_k) → (E, 2*padded, K/block_k)
                old_s13 = layer.w13_weight_scale_inv.data
                _, _, s13_K = old_s13.shape
                new_s13 = torch.zeros(
                    E,
                    2 * padded_inter,
                    s13_K,
                    dtype=old_s13.dtype,
                    device=old_s13.device,
                )
                new_s13[:, :inter_per_part, :] = old_s13[:, :inter_per_part, :]
                new_s13[:, padded_inter : padded_inter + inter_per_part, :] = (
                    old_s13[:, inter_per_part:, :]
                )
                layer.w13_weight_scale_inv = torch.nn.Parameter(
                    new_s13, requires_grad=False
                )

                # Pad w2 scale: (E, N, inter/block_k) → (E, N, padded/block_k)
                old_s2 = layer.w2_weight_scale_inv.data
                new_s2 = torch.zeros(
                    E,
                    w2_N,
                    padded_inter // fp4_block_k,
                    dtype=old_s2.dtype,
                    device=old_s2.device,
                )
                new_s2[:, :, : old_s2.shape[2]] = old_s2
                layer.w2_weight_scale_inv = torch.nn.Parameter(
                    new_s2, requires_grad=False
                )

            for scale_name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
                scale = getattr(layer, scale_name)
                num_experts, num_rows, _ = scale.shape
                is_w13_scale = scale_name == "w13_weight_scale_inv"
                if P._is_gfx1250_supported:
                    scale.data = P.moe_shuffle_scale(
                        scale.contiguous(),
                        experts_cnt=num_experts,
                        is_guinterleave=gu_intv,
                        gate_up=is_w13_scale,
                    )
                else:
                    scale_2d = scale.reshape(-1, scale.shape[-1])
                    scale.data = P.shuffle_scale(
                        scale_2d, num_experts, gu_intv, is_w13_scale
                    )

            layer.w13_weight.data = layer.w13_weight.data.view(fp4_weight_dtype)
            layer.w2_weight.data = layer.w2_weight.data.view(fp4_weight_dtype)

            if P._is_gfx1250_supported:
                is_shuffled = True
                layer.w13_weight.data = P.moe_shuffle_weight(
                    layer.w13_weight,
                    is_guinterleave=gu_intv,
                    gate_up=True,
                )
                layer.w2_weight.data = P.moe_shuffle_weight(
                    layer.w2_weight,
                    is_guinterleave=gu_intv,
                    gate_up=False,
                )
            else:
                is_shuffled = _is_shuffle_moe_mxfp4 or P._use_aiter_a8w4
                if is_shuffled:
                    shuffle_gu_intv = gu_intv and not P._use_aiter_a8w4
                    layer.w13_weight.data = P.shuffle_weight(
                        layer.w13_weight,
                        is_guinterleave=shuffle_gu_intv,
                        gate_up=True,
                    )
                    layer.w2_weight.data = P.shuffle_weight(
                        layer.w2_weight,
                        is_guinterleave=shuffle_gu_intv,
                        gate_up=False,
                    )
            layer.w13_weight.is_shuffled = is_shuffled
            layer.w2_weight.is_shuffled = is_shuffled
            return

        # DSV4 dequant requested: convert to block-FP8 and use the standard
        # block-FP8 MoE path.
        else:
            self._dequantize_aiter_fp4_experts(layer)
            self.weight_block_size = [128, 128]

            # gfx942/gfx950 native FP8 is e4m3fnuz, not e4m3fn.
            if P._is_fp8_fnuz:
                w13_weight, w13_weight_scale, _ = P.normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.w13_weight,
                    weight_scale=layer.w13_weight_scale_inv,
                    input_scale=None,
                )
                w2_weight, w2_weight_scale, _ = P.normalize_e4m3fn_to_e4m3fnuz(
                    weight=layer.w2_weight,
                    weight_scale=layer.w2_weight_scale_inv,
                    input_scale=None,
                )
                layer.w13_weight = P.Parameter(w13_weight, requires_grad=False)
                layer.w13_weight_scale_inv = P.Parameter(
                    w13_weight_scale, requires_grad=False
                )
                layer.w2_weight = P.Parameter(w2_weight, requires_grad=False)
                layer.w2_weight_scale_inv = P.Parameter(
                    w2_weight_scale, requires_grad=False
                )
                layer.w13_input_scale = None
                layer.w2_input_scale = None

            # Only aiter-shuffle when the MoE runner is aiter; the triton runner
            # consumes un-shuffled weights (shuffling the wrong runner corrupts output).
            runner_is_aiter = (
                getattr(self, "runner", None) is not None
                and self.runner.runner_backend.is_aiter()
            )
            if P._use_aiter and runner_is_aiter:
                layer.w13_weight.data = P.shuffle_weight(
                    layer.w13_weight.contiguous(), (16, 16)
                )
                layer.w2_weight.data = P.shuffle_weight(
                    layer.w2_weight.contiguous(), (16, 16)
                )
            return

    if self.convert_mxfp8_to_block:
        # Only aiter-shuffle when the MoE runner is aiter; the triton runner
        # consumes un-shuffled weights (shuffling the wrong runner corrupts output).
        self._convert_mxfp8_moe_to_block_fp8(layer)
        self.use_mxfp8 = False
        self.convert_mxfp8_to_block = False
        self.weight_block_size = [128, 128]
        if P._is_fp8_fnuz:
            w13_weight, w13_weight_scale, _ = P.normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.w13_weight,
                weight_scale=layer.w13_weight_scale_inv,
                input_scale=None,
            )
            w2_weight, w2_weight_scale, _ = P.normalize_e4m3fn_to_e4m3fnuz(
                weight=layer.w2_weight,
                weight_scale=layer.w2_weight_scale_inv,
                input_scale=None,
            )
            layer.w13_weight = P.Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale_inv = P.Parameter(
                w13_weight_scale, requires_grad=False
            )
            layer.w2_weight = P.Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale_inv = P.Parameter(
                w2_weight_scale, requires_grad=False
            )
            layer.w13_input_scale = None
            layer.w2_input_scale = None
        runner_is_aiter = (
            getattr(self, "runner", None) is not None
            and self.runner.runner_backend.is_aiter()
        )
        if P._use_aiter and runner_is_aiter:
            layer.w13_weight.data = P.shuffle_weight(
                layer.w13_weight.contiguous(), (16, 16)
            )
            layer.w2_weight.data = P.shuffle_weight(
                layer.w2_weight.contiguous(), (16, 16)
            )
            layer.w13_weight.is_shuffled = True
            layer.w2_weight.is_shuffled = True
            layer._aiter_gate_up_interleaved = False
        return
    elif self.use_mxfp8 and P.get_moe_a2a_backend().is_flashinfer_megamoe():
        from sglang.srt.layers.moe.flashinfer_megamoe import (
            prepare_mxfp8_moe_weights_for_flashinfer_megamoe,
        )

        prepare_mxfp8_moe_weights_for_flashinfer_megamoe(layer)
        return
    elif self.use_mxfp8:
        self._process_mxfp8_moe_weights(
            layer, quantize=not self.quant_config.is_checkpoint_fp8_serialized
        )
        return

    # If ROCm, normalize the weights and scales to e4m3fnuz
    if P._is_fp8_fnuz:
        # activation_scheme: dynamic
        w13_weight, w13_weight_scale, _ = P.normalize_e4m3fn_to_e4m3fnuz(
            weight=layer.w13_weight,
            weight_scale=layer.w13_weight_scale_inv,
            input_scale=None,
        )
        w2_weight, w2_weight_scale, _ = P.normalize_e4m3fn_to_e4m3fnuz(
            weight=layer.w2_weight,
            weight_scale=layer.w2_weight_scale_inv,
            input_scale=None,
        )
        # Reset the parameter
        layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
        layer.w13_weight_scale_inv = torch.nn.Parameter(
            w13_weight_scale, requires_grad=False
        )
        layer.w13_input_scale = None
        layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
        layer.w2_weight_scale_inv = torch.nn.Parameter(
            w2_weight_scale, requires_grad=False
        )
        layer.w2_input_scale = None
        if P._use_aiter and not _is_hcu:
            layer.w13_weight.data = P.shuffle_weight(
                layer.w13_weight.contiguous(), (16, 16)
            )
            layer.w2_weight.data = P.shuffle_weight(
                layer.w2_weight.contiguous(), (16, 16)
            )
            layer.w13_weight.is_shuffled = True
            layer.w2_weight.is_shuffled = True
            layer._aiter_gate_up_interleaved = False

    if _is_hcu and (P._use_aiter or SGLANG_USE_AITER_FP8_ASM_MOE):
        # HCU aiter ASM MoE kernels consume the b8 weight layout.
        w13_weight = asm_shuffle_weight_b8(layer.w13_weight, 1)
        layer.w13_weight.copy_(w13_weight)
        del w13_weight
        w2_weight = asm_shuffle_weight_b8(layer.w2_weight, 2)
        layer.w2_weight.copy_(w2_weight)
        del w2_weight
    elif P._is_fp8_fnuz and not _is_hcu:
        # Upstream ROCm: the fnuz arm above already normalized and shuffled.
        pass
    elif P._use_aiter:
        # Pre-shuffle weights
        t = P.shuffle_weight(layer.w13_weight, (16, 16))
        layer.w13_weight.copy_(t)
        del t
        t = P.shuffle_weight(layer.w2_weight, (16, 16))
        layer.w2_weight.copy_(t)
        del t
        layer.w13_weight.is_shuffled = True
        layer.w2_weight.is_shuffled = True
        layer._aiter_gate_up_interleaved = False
    elif P._is_cpu:
        assert P._is_cpu_amx_available, (
            "Fp8MoEMethod on CPU requires that CPU has AMX support"
        )
        P._amx_process_weight_after_loading(layer, ["w13_weight", "w2_weight"])
    else:
        # For fp8 moe run with deepgemm, the expert weights and scales need be requantized to ue8m0
        from sglang.srt.layers import deep_gemm_wrapper

        # Check if MoE will actually use DeepGEMM runner
        will_use_deepgemm = self.is_deepgemm_moe_runner_backend_enabled()

        if self.is_fp4_expert and self.dequant_fp4_to_fp8:
            for weight_param, scale_param in [
                (layer.w13_weight, layer.w13_weight_scale_inv),
                (layer.w2_weight, layer.w2_weight_scale_inv),
            ]:
                num_experts = weight_param.shape[0]
                new_weights = []
                new_scales = []
                for e in range(num_experts):
                    w, s = P.cast_e2m1fn_to_e4m3fn(
                        weight_param.data[e], scale_param.data[e]
                    )
                    new_weights.append(w)
                    new_scales.append(s)
                weight_param.data = torch.stack(new_weights)
                scale_param.data = torch.stack(new_scales).float()
                scale_param.format_ue8m0 = False
            self.is_fp4_expert = False
            P.logger.warning_once("Dequantized FP4 expert weights to FP8.")

        if self.is_fp4_expert:
            if P._is_npu:
                self._process_npu_fp4_expert_weights(layer)
                return

            if P.get_moe_runner_backend().is_marlin():
                layer.w13_weight.data = layer.w13_weight.data.view(torch.int8)
                layer.w2_weight.data = layer.w2_weight.data.view(torch.int8)
                return

            # HCU keeps the packed FP4 experts as int8 bytes.
            fp4_weight_dtype = (
                P._require_fp4_dtype() if P._use_aiter and not _is_hcu else torch.int8
            )
            layer.w13_weight.data = layer.w13_weight.data.view(fp4_weight_dtype)
            layer.w2_weight.data = layer.w2_weight.data.view(fp4_weight_dtype)

            if P.get_moe_a2a_backend().is_megamoe():
                from sglang.srt.layers.moe.mega_moe import (
                    build_mega_moe_experts_weights,
                )

                build_mega_moe_experts_weights(layer)
                return

            if P.get_moe_a2a_backend().is_flashinfer_megamoe():
                from sglang.srt.layers.moe.flashinfer_megamoe import (
                    prepare_fp4_moe_weights_for_flashinfer_megamoe,
                )

                prepare_fp4_moe_weights_for_flashinfer_megamoe(layer)
                return

            if deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0 and will_use_deepgemm:
                from deep_gemm import transform_sf_into_required_layout

                for scale_param, weight_param in [
                    (layer.w13_weight_scale_inv, layer.w13_weight),
                    (layer.w2_weight_scale_inv, layer.w2_weight),
                ]:
                    num_experts, n, _ = scale_param.data.shape
                    k = weight_param.shape[2] * 2
                    scale_param.data = transform_sf_into_required_layout(
                        scale_param.data,
                        mn=n,
                        k=k,
                        recipe=(1, 32),
                        num_groups=num_experts,
                        disable_ue8m0_cast=False,
                    )
                layer.w13_weight_scale_inv.format_ue8m0 = True
                layer.w2_weight_scale_inv.format_ue8m0 = True

        if P.get_moe_a2a_backend().is_megamoe() and P.get_platform().is_sm90:
            from sglang.srt.layers.moe.mega_moe_sm90 import (
                build_sm90_mega_moe_experts_weights,
            )

            assert not self.is_fp4_expert
            build_sm90_mega_moe_experts_weights(layer)
            return

        if not self.is_fp4_expert:
            weight_block_size = self.quant_config.weight_block_size
            for weight, weight_scale in (
                (layer.w13_weight, layer.w13_weight_scale_inv),
                (layer.w2_weight, layer.w2_weight_scale_inv),
            ):
                P.requant_block_scale_ue8m0_for_deepgemm(
                    weight,
                    weight_scale,
                    weight_block_size,
                    use_deepgemm_runner=will_use_deepgemm,
                    output_dtype=torch.bfloat16,
                    weight_shape=weight.shape[-2:],
                )

def around_create_weights(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return create_fp8_moe_weight_(*args, **kwargs)


def around_quant_info(original, self, layer, no_combine=False):
    if not is_hcu():
        return original(self, layer, no_combine)
    return maybe_get_hip_aiter_quant_info(self, layer, no_combine)


def around_process_weights(original, self, layer):
    if not is_hcu():
        return original(self, layer)
    return process_weights_after_loading(self, layer)


def around_process_block_weights(original, self, layer):
    if not is_hcu():
        return original(self, layer)
    return process_weights_after_loading_block_quant(self, layer)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    for name, callback in (
        ("create_fp8_moe_weight_", around_create_weights),
        ("maybe_get_hip_aiter_quant_info", around_quant_info),
        ("process_weights_after_loading", around_process_weights),
        ("process_weights_after_loading_block_quant", around_process_block_weights),
    ):
        HookRegistry.register(
            "sglang.srt.layers.quantization.fp8.Fp8MoEMethod." + name,
            callback, HookType.AROUND,
        )
