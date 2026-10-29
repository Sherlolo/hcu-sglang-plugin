# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# Copyright 2024 SGLang Team
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


from __future__ import annotations

import logging
from typing import Optional

import torch
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.utils import get_bool_env_var
from hcu_sglang.utils.detection import is_dcu

logger = logging.getLogger(__name__)
_use_fused_topk_softmax = get_bool_env_var("SGLANG_USE_FUSED_TOPK_SOFTMAX")


def register() -> None:
    target = "sglang.srt.layers.moe.topk.fused_topk"
    logger.info("[hcu_sglang] register AROUND hook: %s", target)
    HookRegistry.register(target, replacement_fused_topk, HookType.AROUND)


def replacement_fused_topk(
    original_fn,
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    correction_bias: Optional[torch.Tensor] = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: Optional[float] = None,
    apply_routed_scaling_factor_on_output: Optional[bool] = False,
    num_fused_shared_experts: int = 0,
    packed_out: Optional[torch.Tensor] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
):
    """Replacement for sglang.srt.layers.moe.topk.fused_topk."""
    from sglang.srt.layers.moe import topk as source

    if is_dcu() and scoring_func == "softmax" and _use_fused_topk_softmax:
        assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
        use_rocm_triton = source._use_rocm_triton_softmax_topk(
            hidden_states, gating_output, topk, correction_bias,
            num_fused_shared_experts, packed_out,
        )
        if not (source._use_aiter and not use_rocm_triton):
            from lightop import op

            num_tokens = hidden_states.shape[0]
            topk_weights = torch.empty(
                num_tokens, topk, dtype=torch.float32, device=hidden_states.device
            )
            topk_ids = torch.empty(
                num_tokens, topk, dtype=torch.int32, device=hidden_states.device
            )
            op.topk_softmax(topk_weights, topk_ids, None, gating_output, renormalize)
            return topk_weights, topk_ids

    return original_fn(
        hidden_states, gating_output, topk, renormalize,
        correction_bias=correction_bias,
        scoring_func=scoring_func,
        routed_scaling_factor=routed_scaling_factor,
        apply_routed_scaling_factor_on_output=apply_routed_scaling_factor_on_output,
        num_fused_shared_experts=num_fused_shared_experts,
        packed_out=packed_out,
        num_token_non_padded=num_token_non_padded,
    )
