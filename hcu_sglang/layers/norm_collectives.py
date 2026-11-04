# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/communication_op.py
from __future__ import annotations

from typing import Optional, Tuple, Union
import torch
from sglang.srt.distributed import parallel_state as P
from sglang.srt.layers import layernorm as N
from hcu_sglang.utils.detection import is_hcu


def tensor_model_parallel_fused_allreduce_rmsnorm(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
    group: str = "tp",
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fused TP all-reduce + RMSNorm.

    Policy and backend selection are owned by GroupCoordinator:
    it may dispatch to communicator-native fused APIs, custom fused kernels,
    or return None so callers can run generic fallback paths.
    """
    coordinator = {
        "tp": P.get_tp_group,
        "attn_tp": P.get_attn_tp_group,
        "moe_tp": P.get_moe_tp_group,
    }.get(group)
    if coordinator is None:
        raise ValueError(f"Unsupported fused all-reduce group: {group}")
    return coordinator().fused_allreduce_rmsnorm(input_, residual_inp_, weight_, eps)


def tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
    group_size: int = 128,
    emit_bf16: bool = False,
    transpose_scale: bool = False,
    group: str = "tp",
) -> Optional[Tuple[torch.Tensor, ...]]:
    """Fused TP all-reduce + RMSNorm + per-group FP8 quant (ROCm/aiter).

    Returns ``(fp8_output, residual_out, per_group_scale)`` by default, or
    ``(fp8_output, residual_out, per_group_scale, bf16_output)`` when
    ``emit_bf16=True`` (kernel writes both fp8 and the pre-quantization bf16
    normed output — no extra kernel). ``None`` when the backend cannot
    service the request (non-AMD, custom AR disabled, shape unsupported).
    Callers MUST handle ``None`` by falling back to the separate
    fused-AR-RMSNorm + per-group-quant path.

    ``transpose_scale=True`` asks the kernel to emit the per-group scale in the
    column-major layout the gfx95 bpreshuffle GEMM consumes, so callers can skip
    the post-kernel ``materialize_bpreshuffle_fp8_scale`` transpose.
    """
    coordinator = {
        "tp": P.get_tp_group,
        "attn_tp": P.get_attn_tp_group,
        "moe_tp": P.get_moe_tp_group,
    }.get(group)
    if coordinator is None:
        raise ValueError(f"Unsupported fused all-reduce group: {group}")
    return coordinator().fused_allreduce_rmsnorm_quant_per_group(
        input_,
        residual_inp_,
        weight_,
        eps,
        group_size,
        emit_bf16=emit_bf16,
        transpose_scale=transpose_scale,
    )


def _forward_with_allreduce_fusion(
    norm_module,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    post_residual_addition: Optional[torch.Tensor],
    weight: torch.Tensor,
    use_attn_tp_group: bool = True,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Shared allreduce-fused RMSNorm logic usable by any norm."""
    if residual is not None:
        from sglang.srt.distributed import (
            attention_tensor_model_parallel_all_reduce,
            moe_tensor_model_parallel_all_reduce,
            tensor_model_parallel_fused_allreduce_rmsnorm,
        )
        from sglang.srt.layers.flashinfer_comm_fusion import (
            flashinfer_allreduce_residual_rmsnorm,
        )
        from sglang.srt.layers.moe.utils import deferred_post_experts_all_reduce

        if use_attn_tp_group:
            world_size = N.get_parallel().attn_tp_size
            allreduce_group = "attn_tp"
            all_reduce = attention_tensor_model_parallel_all_reduce
        else:
            world_size = N.get_parallel().moe_tp_size
            allreduce_group = "moe_tp"
            all_reduce = moe_tensor_model_parallel_all_reduce

        if world_size > 1:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition

            # The AITER communicator is independently selectable from the
            # global AITER compute backend. MiniMax keeps SGLANG_USE_AITER=0
            # for its LightOp dense FP8 path while still using AITER custom AR.
            use_aiter_ar = N._is_hip and N.get_exec().comm.enable_aiter_allreduce_fusion
            if use_aiter_ar:
                fused_result = tensor_model_parallel_fused_allreduce_rmsnorm(
                    x,
                    residual,
                    weight,
                    norm_module.variance_epsilon,
                    group=allreduce_group,
                )
                if fused_result is not None:
                    return fused_result
            elif not N._is_hip:
                fused_result = flashinfer_allreduce_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual,
                    weight=weight,
                    eps=norm_module.variance_epsilon,
                    max_token_num=max(x.shape[0], 2048),
                    use_attn_tp_group=use_attn_tp_group,
                )
                if fused_result[0] is not None:
                    return fused_result
                # The kernel declined: all-reduce over the group its workspace
                # is built on, then add and norm into a new residual tensor, as
                # the kernel does.
                if use_attn_tp_group:
                    x = attention_tensor_model_parallel_all_reduce(x)
                else:
                    x = deferred_post_experts_all_reduce(x)
                if post_residual_addition is None:
                    residual = residual.clone()
                return norm_module.forward(x, residual, None)

            # For AITER route, preserve correctness when fused path is unavailable.
            if use_aiter_ar:
                x = all_reduce(x)
                return norm_module.forward(x, residual, None)

    return norm_module.forward(x, residual, post_residual_addition)


def _forward_with_allreduce_fusion_quant_per_group(
    norm_module,
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    group_size: int = 128,
    use_attn_tp_group: bool = True,
    keep_bf16: bool = False,
):
    """Fused AR + RMSNorm + per-group FP8 quant with graceful staged fallback.

    The single-kernel quantized backend dispatch is ROCm + aiter + gfx95-only.
    Other HIP/aiter runs can still use the 2-kernel fallback below to preserve
    the existing tuple handoff behavior.

    The helper returns one of:

    * ``((fp8, scale), residual)``                       when keep_bf16=False
    * ``((bf16, fp8, scale), residual)``                 when keep_bf16=True
    * ``None``                                           when no fusion is possible

    Fallback chain (best → worst):

    1. Fully-fused AR+RMSNorm+per-group-quant  (aiter single kernel).
    2. Fused AR+RMSNorm followed by a separate per-1x128 quant
       (two kernels, still saves the 3-kernel unfused baseline path).
    3. ``None`` so the caller can run the generic unfused path.

    ``keep_bf16`` is required for GDN-style layers that have one FP8 projection
    (``in_proj_qkvz``) **and** one bf16 projection (``in_proj_ba``) on the
    same normed output; without the bf16 we would have to dequantize which is
    lossy. Standard attention layers (single FP8 ``qkv_proj``) use
    ``keep_bf16=False``.
    """
    if residual is None or not N._use_aiter:
        return None

    from sglang.srt.distributed import (
        tensor_model_parallel_fused_allreduce_rmsnorm,
        tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group,
    )
    from sglang.srt.layers.quantization.fp8_utils import (
        _use_aiter_bpreshuffle_gfx95 as use_bpreshuffle,
    )
    from sglang.srt.layers.quantization.fp8_utils import (
        materialize_bpreshuffle_fp8_scale,
    )

    if use_attn_tp_group:
        world_size = N.get_parallel().attn_tp_size
        allreduce_group = "attn_tp"
    else:
        world_size = N.get_parallel().moe_tp_size
        allreduce_group = "moe_tp"
    if world_size <= 1:
        return None

    # ``transpose_scale=use_bpreshuffle`` asks the fused kernel to emit the
    # per-group scale directly in the column-major layout the gfx95 bpreshuffle
    # GEMM consumes (identical to ``materialize_bpreshuffle_fp8_scale``), so the
    # fused-success paths below need no post-kernel transpose. The separate
    # per-group-quant fallbacks still materialize: ``per_1x128_quant``'s own
    # ``transpose_scale`` byte-shuffles the scale into a *different* arrangement,
    # not the column-major layout this GEMM expects.
    if not keep_bf16:
        result = tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group(
            x,
            residual,
            weight,
            norm_module.variance_epsilon,
            group_size,
            transpose_scale=use_bpreshuffle,
            group=allreduce_group,
        )
        if result is not None:
            fp8_out, residual_out, scale_out = result
            return (fp8_out, scale_out), residual_out

        # Fallback: fused AR+RMSNorm then separate per-group quant.
        fused_result = tensor_model_parallel_fused_allreduce_rmsnorm(
            x,
            residual,
            weight,
            norm_module.variance_epsilon,
            group=allreduce_group,
        )
        if fused_result is None:
            return None
        bf16_out, residual_out = fused_result
        per_1x128_quant, fp8_dtype = N._get_aiter_per_group_quant()
        fp8_out, scale_out = per_1x128_quant(
            bf16_out,
            quant_dtype=fp8_dtype,
            transpose_scale=False,
        )
        if use_bpreshuffle:
            scale_out = materialize_bpreshuffle_fp8_scale(scale_out)
        return (fp8_out, scale_out), residual_out

    # keep_bf16=True: GDN path — need both an unquantized bf16 normed output
    # (for in_proj_ba) AND (fp8, scale) (for in_proj_qkvz). Preferred path:
    # use the fully-fused AR+RMSNorm+per-group-quant kernel with the optional
    # bf16 side-output, so we avoid the separate per-group quant launch
    # entirely. Fallback: fused AR+RMSNorm + separate per-group quant.
    result = tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group(
        x,
        residual,
        weight,
        norm_module.variance_epsilon,
        group_size,
        emit_bf16=True,
        transpose_scale=use_bpreshuffle,
        group=allreduce_group,
    )
    if result is not None and len(result) == 4:
        fp8_out, residual_out, scale_out, bf16_out = result
        return (bf16_out, fp8_out, scale_out), residual_out

    fused_result = tensor_model_parallel_fused_allreduce_rmsnorm(
        x,
        residual,
        weight,
        norm_module.variance_epsilon,
        group=allreduce_group,
    )
    if fused_result is None:
        return None
    bf16_out, residual_out = fused_result
    per_1x128_quant, fp8_dtype = N._get_aiter_per_group_quant()
    fp8_out, scale_out = per_1x128_quant(
        bf16_out,
        quant_dtype=fp8_dtype,
        transpose_scale=False,
    )
    if use_bpreshuffle:
        scale_out = materialize_bpreshuffle_fp8_scale(scale_out)
    return (bf16_out, fp8_out, scale_out), residual_out


def around_fused_norm(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return tensor_model_parallel_fused_allreduce_rmsnorm(*args, **kwargs)


def around_fused_norm_quant(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group(*args, **kwargs)


def around_norm_forward(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return _forward_with_allreduce_fusion(*args, **kwargs)


def around_norm_quant_forward(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return _forward_with_allreduce_fusion_quant_per_group(*args, **kwargs)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    for target, wrapper in (
        ("sglang.srt.distributed.communication_op.tensor_model_parallel_fused_allreduce_rmsnorm", around_fused_norm),
        ("sglang.srt.distributed.communication_op.tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group", around_fused_norm_quant),
        ("sglang.srt.layers.layernorm._forward_with_allreduce_fusion", around_norm_forward),
        ("sglang.srt.layers.layernorm._forward_with_allreduce_fusion_quant_per_group", around_norm_quant_forward),
    ):
        HookRegistry.register(target, wrapper, HookType.AROUND)
