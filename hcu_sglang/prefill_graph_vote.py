# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.

from __future__ import annotations
from typing import Optional
from sglang.srt.managers.scheduler_components import dp_attn as N
from hcu_sglang.utils.detection import is_hcu


def _local_prefill_cuda_graph_vote(
    *,
    local_batch: Optional[ScheduleBatch],
    num_tokens: int,
    prefill_graph_runner,
    coordinated_prefill: bool,
    breakable_prefill: bool,
    spec_algorithm: SpeculativeAlgorithm,
    model_config,
) -> bool:
    """This rank's vote for the prefill graph (min-reduced across dp
    ranks). Extend and mixed batches share the runner's rank-local replay
    policy. A decode batch eligible for the decode->extend conversion votes as
    its 1-token-extend view, so the vote and post-sync conversion agree.
    num_tokens is the rows this rank gathers, encoder replay rows folded in,
    which an extend replay runs."""
    if local_batch is None or local_batch.forward_mode.is_idle():
        return True
    if not coordinated_prefill:
        return False

    mode = local_batch.forward_mode
    if mode in (N.ForwardMode.EXTEND, N.ForwardMode.MIXED):
        input_embeds = local_batch.input_embeds
        replace_embeds = local_batch.replace_embeds
        prefix_lens = local_batch.prefix_lens
        return_logprob = local_batch.return_logprob
    elif (
        mode.is_decode()
        # Conversion replays the breakable graphs only; full's fixed
        # request-slot geometry does not cover converted decode tails.
        and breakable_prefill
        # decode->extend conversion eligibility; needs the captured-graph
        # prefill runner, not the eager fallback.
        and isinstance(prefill_graph_runner, N.PrefillCudaGraphRunner)
        and spec_algorithm.is_none()
        and not local_batch.return_logprob
        # Grammar FSMs advance through the decode result path only.
        and not local_batch.has_grammar
        # A converted batch takes the prefill result path, which commits beam
        # requests per-req rather than through the batch decode fold; member
        # rows also have no req of their own for the reqs-aligned extend lists.
        and all(r.beam_group is None for r in local_batch.reqs)
        # Small-bucket BCG replays amplify the a2a EP logits drift (#30898)
        # into an accuracy loss.
        and N.get_moe_a2a_backend().is_none()
        # The converted view lacks prepare_for_extend's mamba-track fills.
        and not N.uses_ssm_state(model_config)
        # HiSparse decode has its own batch lifecycle and host-offloaded KV.
        and not N.get_memory().enable_hisparse
        and not N.get_exec().overlap.enable_two_batch_overlap
        and N.get_cp_strategy() is None
    ):
        num_tokens = local_batch.batch_size()
        input_embeds = None
        replace_embeds = None
        prefix_lens = None
        return_logprob = False
    else:
        return False

    if prefill_graph_runner is None:
        return True
    if not isinstance(prefill_graph_runner, N.PrefillCudaGraphRunner):
        return False
    backend_vote = prefill_graph_runner._backend_vote_prefill_cuda_graph
    if (
        mode in (N.ForwardMode.EXTEND, N.ForwardMode.MIXED)
        and backend_vote is not None
        and not backend_vote(local_batch)
    ):
        return False
    return prefill_graph_runner.can_replay_locally(
        batch_size=local_batch.batch_size(),
        num_tokens=num_tokens,
        input_embeds=input_embeds,
        replace_embeds=replace_embeds,
        prefix_lens=prefix_lens,
        is_target_verify=mode.is_target_verify(),
        capture_hidden_mode=None,
        return_logprob=return_logprob,
        contains_mm_inputs=any(
            mm_input is not None and mm_input.contains_mm_input()
            for mm_input in local_batch.multimodal_inputs or ()
        ),
        lora_ineligible=prefill_graph_runner.enable_lora,
        is_mixed=mode == N.ForwardMode.MIXED,
        batch_max_context_len=(
            int(local_batch.seq_lens_cpu.max().item())
            if (
                prefill_graph_runner.max_context_size is not None
                or prefill_graph_runner.max_seq_len is not None
            )
            and local_batch.seq_lens_cpu is not None
            and local_batch.seq_lens_cpu.numel() > 0
            else None
        ),
    )


def around_vote(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return _local_prefill_cuda_graph_vote(*args, **kwargs)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.managers.scheduler_components.dp_attn._local_prefill_cuda_graph_vote", around_vote, HookType.AROUND)
