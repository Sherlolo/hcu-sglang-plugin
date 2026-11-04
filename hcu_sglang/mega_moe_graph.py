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
from __future__ import annotations

import torch
from sglang.srt.environ import envs
from sglang.srt.layers.moe import mega_moe as N
from hcu_sglang.utils.detection import is_hcu

_is_hcu = is_hcu()
_HCU_MEGA_MOE_RUNTIME_DEEP_GEMM = "deep_gemm"
_HCU_MEGA_MOE_RUNTIME_MEGAMOE = "megamoe"
_HCU_MEGA_MOE_RUNTIMES = {
    _HCU_MEGA_MOE_RUNTIME_DEEP_GEMM,
    _HCU_MEGA_MOE_RUNTIME_MEGAMOE,
}


def get_hcu_mega_moe_runtime() -> str:
    runtime = envs.SGLANG_HCU_MEGA_MOE_RUNTIME.get().strip().lower()
    if runtime not in _HCU_MEGA_MOE_RUNTIMES:
        raise ValueError(
            "SGLANG_HCU_MEGA_MOE_RUNTIME must be one of "
            f"{sorted(_HCU_MEGA_MOE_RUNTIMES)}, got {runtime!r}"
        )
    return runtime


def _is_standalone_megamoe_runtime() -> bool:
    return _is_hcu and get_hcu_mega_moe_runtime() == _HCU_MEGA_MOE_RUNTIME_MEGAMOE


@torch.inference_mode()
def set_mega_moe_cuda_graph_num_tokens(num_tokens: int) -> None:
    if not _is_standalone_megamoe_runtime():
        return
    for buf in N._MEGA_MOE_SYMM_BUFFER.values():
        graph_num_tokens = getattr(buf, "cuda_graph_num_tokens", None)
        if graph_num_tokens is not None:
            graph_num_tokens.fill_(num_tokens)
