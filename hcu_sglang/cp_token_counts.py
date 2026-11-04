# Copyright 2023-2026 SGLang Team
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
from typing import List
from hcu_sglang.encoder_swa_replay import replay_len, folded_replay_rows
from sglang.srt.layers.cp.utils import get_cp_strategy, ZigzagCPStrategy
from sglang.srt.runtime_context import get_parallel


def enable_cp_v2() -> bool:
    """Return whether the strategy-based generic prefill CP path is available."""
    from sglang.srt.utils import is_musa, is_npu
    from hcu_sglang.utils.detection import is_hcu as is_hcu

    if is_hcu():
        from sglang.srt.environ import envs

        return bool(envs.SGLANG_ENABLE_CP_V2.get())
    return not (is_npu() or is_musa())


def normalize_dp_cp_token_counts(
    global_num_tokens: list[int], global_cp_num_tokens: list[int]
) -> list[int]:
    """Select one collective layout for a synchronized attention-DP forward.

    CP may coexist with an idle DP replica, which participates with zero local
    rows.  It may not coexist with a non-idle replica that is ineligible for CP:
    that replica needs the full-TP layout.  In the latter case disable CP for
    the whole global forward so every rank executes matching collectives.
    """
    if len(global_num_tokens) != len(global_cp_num_tokens):
        raise ValueError(
            "DP/CP token-count width mismatch: "
            f"num_tokens={global_num_tokens}, cp_tokens={global_cp_num_tokens}"
        )
    has_cp_batch = any(int(tokens) > 0 for tokens in global_cp_num_tokens)
    has_non_cp_work = any(
        int(tokens) > 0 and int(cp_tokens) == 0
        for tokens, cp_tokens in zip(global_num_tokens, global_cp_num_tokens)
    )
    if has_cp_batch and has_non_cp_work:
        return [0] * len(global_cp_num_tokens)
    return [int(tokens) for tokens in global_cp_num_tokens]


def get_cp_v2_physical_token_count(
    *, num_tokens: int, extend_seq_lens, cp_size: int
) -> int:
    """Compute this DP replica's CP-local padded rows in the scheduler.

    Zero means that CP-v2 is inactive for the local batch.  This function is
    deliberately CPU-only because it runs before the scheduler's existing DP
    metadata all-gather.
    """
    if (
        not enable_cp_v2()
        or cp_size <= 1
        or extend_seq_lens is None
        or get_parallel().enable_cp_tp_group_sharing
    ):
        return 0
    strategy = get_cp_strategy()
    if not isinstance(strategy, ZigzagCPStrategy):
        return 0

    from types import SimpleNamespace

    extend_seq_lens = [int(length) for length in extend_seq_lens]
    probe = SimpleNamespace(
        forward_mode=None,
        extend_seq_lens_cpu=extend_seq_lens,
    )
    if not strategy.can_apply(int(num_tokens), probe):
        return 0
    return compute_zigzag_cp_physical_token_count(extend_seq_lens, cp_size)


def compute_zigzag_cp_physical_token_count(
    extend_seq_lens: List[int], cp_size: int
) -> int:
    """Return the padded row count held by one zigzag-CP rank.

    The scheduler needs this value before ``ForwardBatch`` construction so it
    can publish CP-local DP-attention buffer sizes to every DP replica.  Keep
    the arithmetic here in lockstep with ``build_metadata`` and
    ``pad_logical_token_to_physical`` without allocating any device tensors.
    """
    if cp_size <= 1:
        return sum(int(length) for length in extend_seq_lens)

    cp_segment_num = cp_size * 2
    per_rank_tokens = [0] * cp_size
    for raw_length in extend_seq_lens:
        length = int(raw_length)
        base, rem = divmod(length, cp_segment_num)
        for rank in range(cp_size):
            per_rank_tokens[rank] += (
                2 * base + int(rank < rem) + int(cp_segment_num - 1 - rank < rem)
            )

    # Zigzag padding aligns the largest logical shard to 2 * cp_size.
    max_tokens = max(per_rank_tokens, default=0)
    return (max_tokens + cp_segment_num - 1) // cp_segment_num * cp_segment_num






def attn_replica_size() -> int:
    """Return the fixed physical width of one attention-DP replica."""
    parallel = get_parallel()
    return parallel.attn_tp_size * parallel.attn_cp_size
