# Copyright 2026 SGLang Team
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
from sglang.srt.mem_cache import prefill_budget as N
from hcu_sglang.utils.detection import is_hcu


def PrefillBudget_check_prefill(
    self,
    *,
    extend_input_len: int,
    total_tokens: int,
    max_new_tokens: int,
    input_tokens: int,
    swa_host_hit_length: int,
    chunk_limit: int | None,
    auxiliary_swa_host_hit_length: int = 0,
) -> tuple[bool, int | None]:
    """Return admission feasibility and a memory bound on the requested chunk."""
    return (
        (True, chunk_limit)
        if total_tokens < self.remaining_total
        else (False, None)
    )


def SWAPrefillBudget_check_prefill(
    self,
    *,
    extend_input_len: int,
    total_tokens: int,
    max_new_tokens: int,
    input_tokens: int,
    swa_host_hit_length: int,
    chunk_limit: int | None,
    auxiliary_swa_host_hit_length: int = 0,
) -> tuple[bool, int | None]:
    if total_tokens >= self.remaining_total:
        return False, None
    extend_input_len = self.ceil_paged_tokens(extend_input_len)
    needed = self.swa_tokens(
        extend_input_len,
        max_new_tokens,
        chunk_limit=chunk_limit,
        swa_host_hit_length=swa_host_hit_length,
    )
    fits = (
        needed <= self.remaining_swa
        if self.req_ring
        else needed < self.remaining_swa
    )
    if fits:
        return True, chunk_limit
    # Only permanent shortfalls may shrink a chunk. Transient pressure waits
    # so a new prefill does not consume running decodes' window headroom.
    cap = 0
    if self.swa_never_fits(
        extend_input_len,
        max_new_tokens,
        chunk_limit=chunk_limit,
        swa_host_hit_length=swa_host_hit_length,
    ):
        cap = self._chunk_cap(max_new_tokens, swa_host_hit_length)
    if chunk_limit is None or cap <= 0:
        return False, None
    return True, min(chunk_limit, cap)


def SharedSWAPrefillBudget_check_prefill(
    self,
    *,
    extend_input_len: int,
    total_tokens: int,
    max_new_tokens: int,
    input_tokens: int,
    swa_host_hit_length: int,
    chunk_limit: int | None,
    auxiliary_swa_host_hit_length: int = 0,
) -> tuple[bool, int | None]:
    needed = self.swa_tokens(
        extend_input_len,
        max_new_tokens,
        chunk_limit=chunk_limit,
        swa_host_hit_length=swa_host_hit_length,
    )
    if self._fits(total_tokens, needed):
        return True, chunk_limit
    if chunk_limit is None or self._fits(
        input_tokens + max_new_tokens + self.page_size,
        self.swa_tokens(input_tokens, max_new_tokens, chunk_limit=chunk_limit),
        empty_pool=True,
    ):
        return False, None
    cap = self._joint_chunk_cap(
        max_chunk_tokens=min(chunk_limit, max(0, extend_input_len - 1)),
        chunk_limit=chunk_limit,
        swa_host_hit_length=swa_host_hit_length,
    )
    return (True, min(chunk_limit, cap)) if cap > 0 else (False, None)


def around_PrefillBudget(original, self, *args, **kwargs):
    if not is_hcu():
        return original(self, *args, **kwargs)
    return PrefillBudget_check_prefill(self, *args, **kwargs)


def around_SWAPrefillBudget(original, self, *args, **kwargs):
    if not is_hcu():
        return original(self, *args, **kwargs)
    return SWAPrefillBudget_check_prefill(self, *args, **kwargs)


def around_SharedSWAPrefillBudget(original, self, *args, **kwargs):
    if not is_hcu():
        return original(self, *args, **kwargs)
    return SharedSWAPrefillBudget_check_prefill(self, *args, **kwargs)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.mem_cache.prefill_budget.PrefillBudget.check_prefill", around_PrefillBudget, HookType.AROUND)
    HookRegistry.register("sglang.srt.mem_cache.prefill_budget.SWAPrefillBudget.check_prefill", around_SWAPrefillBudget, HookType.AROUND)
    HookRegistry.register("sglang.srt.mem_cache.prefill_budget.SharedSWAPrefillBudget.check_prefill", around_SharedSWAPrefillBudget, HookType.AROUND)
