# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# Copyright 2025-2026 SGLang Team
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
import sglang.srt.models.glm4_moe as upstream
from hcu_sglang.utils.detection import is_hcu
globals().update({name: value for name, value in vars(upstream).items() if not name.startswith("__")})
_is_hcu = is_hcu()

def forward_normal(
    self,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    if hidden_states.shape[0] > 0:
        shared_output = self._forward_shared_experts(hidden_states)
        # router_logits: (num_tokens, n_experts)
        router_logits = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
    else:
        shared_output = None
        topk_output = self.topk.empty_topk_output(hidden_states.device)

    final_hidden_states = self.experts(hidden_states, topk_output)
    # HCU's Triton MoE fallback applies routed_scaling_factor in
    # moe_sum_reduce_triton; applying it again here would yield factor^2.
    if not _is_cuda and not _is_hcu and not _use_aiter:
        final_hidden_states *= self.routed_scaling_factor
    if shared_output is not None:
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            final_hidden_states_out = torch.empty_like(final_hidden_states)
        torch.add(final_hidden_states, shared_output, out=final_hidden_states_out)
        final_hidden_states = final_hidden_states_out
    return final_hidden_states
