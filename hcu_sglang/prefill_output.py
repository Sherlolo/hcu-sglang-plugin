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

from dataclasses import dataclass
from typing import Optional
import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput as _NativeLogitsProcessorOutput
from hcu_sglang.utils.detection import is_hcu


@dataclass
class LogitsProcessorOutput(_NativeLogitsProcessorOutput):
    # Draft decode VP returns a deterministic top-1 proposal without full logits.
    draft_top1_token_ids: Optional[torch.Tensor] = None
    draft_top1_probs: Optional[torch.Tensor] = None
    # Greedy EAGLE3 fast path.  When populated, the draft worker consumes
    # these ids directly and ``next_token_logits`` is intentionally None.
    draft_topk_index: Optional[torch.Tensor] = None


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.layers.logits_processor.LogitsProcessorOutput", LogitsProcessorOutput, HookType.REPLACE)
