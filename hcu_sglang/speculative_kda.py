from __future__ import annotations

import types
from sglang.srt.speculative import spec_utils as N
from hcu_sglang.utils.detection import is_hcu

def commit_mamba_states_after_verify(
    target_worker: TpModelWorker,
    batch: ScheduleBatch,
    accept_lens: torch.Tensor,
    accept_index: torch.Tensor,
    draft_token_num: int,
    prepared_step_indices: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
) -> None:
    """Commit accepted per-step mamba states into the persistent caches.

    During TARGET_VERIFY, hybrid linear attention backends keep per-step
    states in intermediate caches instead of advancing the persistent
    conv/ssm caches. After acceptance, the state of each request's last
    accepted step is committed back, plus the interval-crossing state used
    for prefix-cache tracking (mamba extra_buffer mode).

    No-op for models without mamba-style state or backends without the
    commit hook.
    """
    model_runner = target_worker.model_runner
    if mambaish_config(model_runner.model_config) is None:
        return

    # ReplaySSM spec-verify path (Part B of #28511): the accepted drafts already
    # live in the per-slot circular ring (written during verify). Instead of
    # scattering an intermediate full SSM state into `temporal`, advance the
    # block-keyed cursors by the accepted count (the ring owns the SSM state; the
    # verify/flush kernel folds it into `temporal` periodically). The CONV state
    # still needs its usual accept-rollback, so we keep the conv-window scatter and
    # skip only the SSM scatter. GDN-only + linear-chain (topk<=1) -- the runtime
    # ring is allocated only then; KDA never allocates the cursors.
    req_pool = model_runner.req_to_token_pool
    mamba_pool = getattr(req_pool, "mamba_pool", None)
    bs = accept_lens.shape[0]
    src_indices_raw = (
        batch.req_pool_indices[:bs] if pp_spec_stable_rows_enabled() else None
    )

    # Fold-every-commit: replay the accepted prefix from the ring into
    # `temporal`; the same fold stores the interval-crossing state to the
    # track slot, so no SSM scatter or force-flush is needed here.
    if (
        mamba_pool is not None
        and getattr(mamba_pool, "replayssm_spec_fold", False)
        and not getattr(mamba_pool, "replayssm_is_kda", False)
    ):
        if batch.forward_mode.is_idle() or accept_index.numel() == 0:
            return
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
            commit_gdn_replayssm_fold_after_verify,
        )

        spec_state = req_pool.get_speculative_mamba2_params_all_layers()
        state_batch_indices = req_pool.get_mamba_indices(batch.req_pool_indices)
        if prepared_step_indices is None:
            last_correct_step_indices, mamba_steps_to_track = (
                _verify_commit_step_indices(
                    batch=batch,
                    accept_index=accept_index,
                    accept_lens=accept_lens,
                    draft_token_num=draft_token_num,
                )
            )
        else:
            last_correct_step_indices, mamba_steps_to_track = prepared_step_indices
        commit_gdn_replayssm_fold_after_verify(
            spec_state=spec_state,
            state_batch_indices=state_batch_indices,
            accept_lens=accept_lens,
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            src_indices_raw=src_indices_raw,
            null_block_id=-1,
        )
        return

    if (
        mamba_pool is not None
        and getattr(mamba_pool, "replayssm_cache_base", None) is not None
        and not getattr(mamba_pool, "replayssm_is_kda", False)
    ):
        if batch.forward_mode.is_idle() or accept_index.numel() == 0:
            return
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
            commit_gdn_replayssm_circular,
            commit_gdn_replayssm_spec,
        )
        from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
            fused_conv_window_scatter_with_mask,
        )

        spec_state = req_pool.get_speculative_mamba2_params_all_layers()
        state_batch_indices = req_pool.get_mamba_indices(batch.req_pool_indices)
        replay_indices = batch.req_pool_indices
        if prepared_step_indices is None:
            last_correct_step_indices, mamba_steps_to_track = (
                _verify_commit_step_indices(
                    batch=batch,
                    accept_index=accept_index,
                    accept_lens=accept_lens,
                    draft_token_num=draft_token_num,
                )
            )
        else:
            last_correct_step_indices, mamba_steps_to_track = prepared_step_indices
        # Advance the per-request circular cursors by the accepted count (incl. the
        # bonus token). max_cache_len = ring length L = replayssm_d.shape[-2].
        commit_gdn_replayssm_spec(
            write_pos=mamba_pool.replayssm_spec_write_pos,
            cache_base=mamba_pool.replayssm_cache_base,
            is_flush=mamba_pool.replayssm_is_flush,
            num_accepted=accept_lens,  # [bs], includes the bonus token
            replay_indices=replay_indices,
            max_cache_len=spec_state.replayssm_d.shape[-2],
            max_spec_len=draft_token_num,
            fold_every_commit=spec_state.temporal.dtype != torch.float32,
            null_block_id=-1,  # SGLang: valid slots >= 0, padding == -1
        )
        # Capacity rows fold all layers in one launch; track rows snapshot the
        # exact crossing state without disturbing the active circular history.
        commit_gdn_replayssm_circular(
            checkpoint_state=spec_state.temporal,
            d_cache=spec_state.replayssm_d,
            k_cache=spec_state.replayssm_k,
            g_cache=spec_state.replayssm_g,
            d_residual_cache=spec_state.replayssm_rawv,
            k_residual_cache=spec_state.replayssm_rawk,
            state_batch_indices=state_batch_indices,
            replay_indices=replay_indices,
            write_pos=mamba_pool.replayssm_spec_write_pos,
            cache_base=mamba_pool.replayssm_cache_base,
            is_flush=mamba_pool.replayssm_is_flush,
            accept_lens=accept_lens,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            null_block_id=-1,
        )
        # Roll back active conv state and snapshot its interval-crossing window.
        fused_conv_window_scatter_with_mask(
            spec_state.conv[0],
            spec_state.intermediate_conv_window[0],
            state_batch_indices,
            last_correct_step_indices,
            src_indices_raw,
        )
        if batch.mamba_track_indices is not None:
            fused_conv_window_scatter_with_mask(
                spec_state.conv[0],
                spec_state.intermediate_conv_window[0],
                batch.mamba_track_indices,
                mamba_steps_to_track,
                src_indices_raw,
            )
        return

    # KDA ReplaySSM (fold-every-commit): KDA keeps its own recurrent verify kernel
    # for the OUTPUT, so we replay the accepted window into the fp32 checkpoint
    # (`temporal`) here on commit -- `temporal` is always the current committed
    # state. The draft window's raw inputs were written to the ring during verify
    # by the KDA backend. Gate on the fold flag + is_kda (the cursor tensors are
    # never allocated under fold, so they cannot serve as the signal).
    if (
        mamba_pool is not None
        and getattr(mamba_pool, "replayssm_spec_fold", False)
        and getattr(mamba_pool, "replayssm_is_kda", False)
    ):
        if batch.forward_mode.is_idle() or accept_index.numel() == 0:
            return
        from hcu_sglang.layers.attention.linear.kda_replayssm import (
            commit_kda_replayssm_after_verify,
        )

        spec_state = req_pool.get_speculative_mamba2_params_all_layers()
        bs = accept_lens.shape[0]
        state_batch_indices = req_pool.translate_mamba_indices(req_pool.get_mamba_indices(batch.req_pool_indices))
        accept_indices_offset = torch.arange(
            0,
            bs * draft_token_num,
            step=draft_token_num,
            dtype=accept_lens.dtype,
            device=accept_lens.device,
        )
        req_idx = torch.arange(bs, dtype=torch.int64, device=accept_lens.device)
        last_correct_step_indices = (
            accept_index[req_idx, (accept_lens - 1).to(torch.int64)]
            - accept_indices_offset
        )
        # extra_buffer: the interval-crossing step whose state must snapshot into
        # the track ping-pong slot (mirrors the regular commit's
        # mamba_steps_to_track); commit_kda_replayssm_spec folds it in one pass, so
        # `temporal` stays current and no device-side force-flush is needed.
        mamba_track_indices = batch.mamba_track_indices
        mamba_steps_to_track = None
        if mamba_track_indices is not None:
            ti = mamba_track_grid(batch.tree_cache.page_size)
            seq_pre = batch.seq_lens
            seq_post = batch.seq_lens + accept_lens
            to_track_mask = seq_pre // ti != seq_post // ti
            tracking_point = seq_post // ti * ti
            to_track_ith = torch.clamp(tracking_point - seq_pre - 1, min=0).to(
                torch.int64
            )
            candidate = accept_index[req_idx, to_track_ith] - accept_indices_offset
            mamba_steps_to_track = torch.where(
                to_track_mask, candidate, torch.full_like(candidate, -1)
            )
        if mamba_track_indices is not None:
            mamba_track_indices = req_pool.translate_mamba_indices(mamba_track_indices)
        commit_kda_replayssm_after_verify(
            spec_state=spec_state,
            use_hcu_kda=mamba_pool.use_hcu_kda,
            state_batch_indices=state_batch_indices,
            accept_lens=accept_lens,  # incl. bonus token
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            src_indices_raw=src_indices_raw,
            null_block_id=-1,  # SGLang: valid slots >= 0, padding == -1
        )
        return

    attn_backend = model_runner.attn_backend

    # `accept_lens` already includes the bonus token (drafts + 1 per req).
    if not batch.forward_mode.is_idle() and accept_index.numel() > 0:
        if prepared_step_indices is None:
            last_correct_step_indices, mamba_steps_to_track = (
                _verify_commit_step_indices(
                    batch=batch,
                    accept_index=accept_index,
                    accept_lens=accept_lens,
                    draft_token_num=draft_token_num,
                )
            )
        else:
            last_correct_step_indices, mamba_steps_to_track = prepared_step_indices

        if hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            attn_backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=last_correct_step_indices,
                mamba_track_indices=batch.mamba_track_indices,
                mamba_steps_to_track=mamba_steps_to_track,
                model=model_runner.model,
                req_pool_indices=batch.req_pool_indices[:bs],
            )

_HOOK = None


def register_hooks():
    global _HOOK
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    target = "sglang.srt.speculative.spec_utils.commit_mamba_states_after_verify"
    if _HOOK is not None and any(hook is _HOOK for _, hook, _ in HookRegistry._hooks.get(target, ())):
        return
    fn = types.FunctionType(commit_mamba_states_after_verify.__code__, vars(N),
                            "commit_mamba_states_after_verify",
                            commit_mamba_states_after_verify.__defaults__)
    fn.__kwdefaults__ = commit_mamba_states_after_verify.__kwdefaults__
    fn.__annotations__ = commit_mamba_states_after_verify.__annotations__
    _HOOK = fn
    HookRegistry.register(target, fn, HookType.REPLACE)
