# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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

from __future__ import annotations
import torch
from torch import nn
from sglang.srt.models import deepseek_v2 as N
from hcu_sglang.utils.detection import is_hcu as is_hcu
from hcu_sglang.models.deepseek.deepseek_nextn.methods.model import enable_cp_v2
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from hcu_sglang.models.deepseek.current_v4.hcu_backend.hcu_cp_utils import mla_use_prefill_cp

_is_hcu = is_hcu()


def __init__(
    self,
    config,
    quant_config,
    prefix: str = "",
    is_hash_moe: bool = False,
    is_deepseek_v4: bool = False,
    vl_correction_bias: bool = False,
):
    super(N.MoEGate, self).__init__()
    self.is_deepseek_v4 = is_deepseek_v4
    is_glm_next = getattr(config, "model_type", "") in (
        "glm5_next",
        "glm5v_next",
        "glm5next_text",
        "glm5_next_text",
    )
    router_fp32 = (
        (
            N.envs.SGLANG_MOE_ROUTER_USE_CONFIG_DTYPE.get()
            and getattr(config, "router_dtype", None) in ("float32", "fp32")
        )
        if is_glm_next
        else getattr(config, "router_fp32", False)
    )
    self.weight = nn.Parameter(
        torch.empty(
            (config.n_routed_experts, config.hidden_size),
            dtype=torch.float32
            if router_fp32
            else (torch.bfloat16 if is_glm_next else torch.get_default_dtype()),
        )
    )
    if config.topk_method == "noaux_tc" and not is_hash_moe:
        correction_bias_dtype = torch.float32
        # GLM-5.2's bias sits at an offset where its spread is only a few bf16 ULPs
        # wide, so bf16 collapses it and reorders top-k routing. HF stores it fp32.
        if quant_config is not None and not N.is_glm_moe_dsa(config):
            if N._use_aiter and quant_config.get_name() in (
                "fp8",
                "compressed_tensors",
                "quark",
            ):
                correction_bias_dtype = torch.bfloat16
        correction_bias = torch.empty(
            (config.n_routed_experts), dtype=correction_bias_dtype
        )
        if quant_config is not None and quant_config.get_name() == "expert_pack":
            correction_bias.zero_()
        self.e_score_correction_bias = nn.Parameter(correction_bias)
    else:
        self.e_score_correction_bias = None
    self.e_score_correction_bias_vl = None
    if vl_correction_bias:
        self.e_score_correction_bias_vl = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32),
            requires_grad=False,
        )
    if N._is_cpu and N._is_cpu_amx_available:
        self.quant_method = N.PackWeightMethod(weight_names=["weight"])
    self.tiny_router_gemm_max_tokens = N.tiny_router_gemm_max_tokens(
        num_experts=config.n_routed_experts,
        hidden_size=config.hidden_size,
        weight_dtype=self.weight.dtype,
    )
    # Rows up to which the ROCm split-K router serves the gate (-1: never)
    self.rocm_router_max_tokens = (
        N._hip_moe.router_max_tokens(self, config, is_hash_moe) if N._is_hip else -1
    )


def forward(
    self,
    hidden_states,
    gemm_output_zero_allocator: BumpAllocator = None,
    forward_batch: ForwardBatch = None,
):
    if self.weight.dtype == torch.float32:
        return N.F.linear(hidden_states.float(), self.weight)
    hidden_states = hidden_states.to(self.weight.dtype)

    if N.use_intel_amx_backend(self):
        return torch.ops.sgl_kernel.weight_packed_linear(
            hidden_states,
            self.weight,
            None,  # bias
            True,  # is_vnni
        )

    if N.get_exec().deterministic.enable_deterministic_inference:
        if N._is_cuda or N._is_hip:
            return torch.mm(hidden_states, self.weight.t(), out_dtype=torch.float32)
        return N.F.linear(hidden_states.float(), self.weight.float(), None)

    if (
        not enable_cp_v2()
        and not self.is_deepseek_v4
        and forward_batch is not None
        and (dsa_use_prefill_cp(forward_batch) or mla_use_prefill_cp(forward_batch))
    ):
        return N.F.linear(hidden_states, self.weight, None)

    if hidden_states.shape[0] <= self.tiny_router_gemm_max_tokens:
        logits = N.tiny_gemm_bf16(
            hidden_states,
            self.weight,
            out_dtype=torch.float32,
            max_m=self.tiny_router_gemm_max_tokens,
        )
    elif (
        _is_hcu
        and self.is_deepseek_v4
        and N.envs.SGLANG_OPT_BF16_FP32_GEMM_ALGO.get() == "auto"
    ):
        # Route auto before the broad AITER branch so HCU DSV4 router
        # logits use the per-shape selector.
        from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32

        logits = linear_bf16_fp32(hidden_states, self.weight)
    elif N._use_aiter:
        logits = N.aiter_dsv3_router_gemm(hidden_states, self.weight)
    elif _is_hcu and self.is_deepseek_v4:
        # HCU Hash-MoE requires FP32 router logits.
        from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32

        logits = linear_bf16_fp32(hidden_states, self.weight)
    elif not N._is_cuda:
        logits = N.F.linear(hidden_states, self.weight, None)
    else:
        # cuBLAS bf16 x bf16 -> fp32 GEMM (torch.mm's out_dtype kwarg is CUDA-only)
        from sglang.kernels.ops.gemm.bf16_fp32 import linear_bf16_fp32

        logits = linear_bf16_fp32(hidden_states, self.weight)

    return logits


def around_init(original, self, *args, **kwargs):
    if not is_hcu():
        return original(self, *args, **kwargs)
    return __init__(self, *args, **kwargs)


def around_forward(original, self, *args, **kwargs):
    if not is_hcu():
        return original(self, *args, **kwargs)
    return forward(self, *args, **kwargs)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    for name, hook in [("__init__", around_init), ("forward", around_forward)]:
        HookRegistry.register("sglang.srt.models.deepseek_v2.MoEGate." + name, hook, HookType.AROUND)
