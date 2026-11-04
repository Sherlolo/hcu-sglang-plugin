from __future__ import annotations

import functools
from sglang.srt.layers import dp_attention as N
from hcu_sglang.utils.detection import is_hcu

globals().update({name: value for name, value in vars(N).items() if not name.startswith("__")})
_is_hcu = is_hcu()
DpPaddingMode = N.DpPaddingMode


def attn_replica_size():
    parallel = get_parallel()
    return parallel.attn_tp_size * parallel.attn_cp_size


def physical_ep_rank_to_dp_rank(ep_rank, replica_size):
    if ep_rank < 0 or replica_size <= 0:
        raise ValueError(f"Invalid EP/attention topology: ep_rank={ep_rank}, attn_replica_size={replica_size}.")
    return ep_rank // replica_size


def get_attn_dp_tp_group():
    from sglang.srt.distributed import get_attn_dp_tp_group as provider
    return provider()

def dp_gather_slot() -> int:
    """Return this process's index in the DP gather.

    After elastic scale-up, convert the physical EP rank (the TP rank plus the
    join offset) to its logical attention-DP replica; otherwise use the local
    attention-DP rank. Inside a draft scope that owns its attention, it is still
    the slot in the target's gather (see patch_tensor_parallel_group).
    """
    scoped = get_flags().dp.scoped_gather_slot
    if scoped is not None:
        return scoped
    parallel = get_parallel()
    if world_dp_gather_enabled():
        return physical_ep_rank_to_dp_rank(
            parallel.tp_rank + parallel.ep_join_rank_offset,
            attn_replica_size(),
        )
    return parallel.attn_dp_rank

def set_dp_buffer_len_from_batch(forward_batch: ForwardBatch) -> None:
    """Publish the DP gather sizes ``forward_batch`` carries: the buffer
    length, the per-rank token counts as padded for the gather, this rank's
    entry, and the padding mode. Capture batches carry no separately padded
    list, so the raw counts stand in for it."""
    global_num_tokens = forward_batch.global_num_tokens_padded_cpu
    if global_num_tokens is None:
        global_num_tokens = forward_batch.global_num_tokens_cpu
    dp_rank = dp_slot_in(global_num_tokens)
    set_dp_buffer_len(
        forward_batch.global_dp_buffer_len,
        global_num_tokens[dp_rank],
        forward_batch.dp_padding_mode.is_max_len(),
        global_num_tokens,
        forward_batch.global_num_tokens_gpu,
    )

def init_dp_gathered_buffer(model_config: ModelConfig):
    """Size the gathered buffer from the model this worker is about to run."""
    # Kimi linear and other hybrid models may not expose the legacy
    # hybrid_override_pattern field. Reuse the central hybrid detector so
    # eager DP idle ranks take the existing MAX_LEN fabricated-row path.
    from sglang.srt.configs.hybrid_arch import mambaish_config

    get_flags().dp.max_len_with_idle = (
        getattr(model_config.hf_config, "hybrid_override_pattern", None) is not None
        or mambaish_config(model_config) is not None
    )
    _DpGatheredBufferWrapper.set_metadata(
        hidden_size=model_config.hidden_size,
        dtype=model_config.dtype,
        device=torch.device(get_device().device),
    )

def get_dp_local_info(forward_batch: ForwardBatch) -> Tuple[torch.Tensor, torch.Tensor]:
    # `get_dp_local_info` is only called in global DP gather and scatter. We use global DP rank here.

    if forward_batch.dp_local_start_pos is None:
        dp_rank = getattr(forward_batch, "dp_local_token_index", None)
        if dp_rank is None:
            dp_rank = dp_slot_in(forward_batch.global_num_tokens_gpu)
        cumtokens = torch.cumsum(forward_batch.global_num_tokens_gpu, dim=0)
        if dp_rank == 0:
            local_start_pos = torch.zeros_like(cumtokens[0])
        else:
            local_start_pos = cumtokens[dp_rank - 1]
        local_num_tokens = forward_batch.global_num_tokens_gpu[dp_rank]

        forward_batch.dp_local_start_pos = local_start_pos
        forward_batch.dp_local_num_tokens = local_num_tokens

    return forward_batch.dp_local_start_pos, forward_batch.dp_local_num_tokens

def get_dp_local_slice_cpu(
    forward_batch: ForwardBatch,
    can_run_graph: bool,
    cuda_graph_batch: Optional[int],
) -> Tuple[int, int]:
    # CPU (start, length) slice for DP-local data in a rank-padded buffer.
    # Returns Python ints (no D2H sync) and handles the cuda-graph-padded layout.
    global_num_tokens = forward_batch.global_num_tokens_cpu
    dp_rank = getattr(forward_batch, "dp_local_token_index", None)
    if dp_rank is None:
        dp_rank = dp_slot_in(global_num_tokens)
    local_num_tokens = global_num_tokens[dp_rank]
    if can_run_graph:
        local_start_pos = dp_rank * cuda_graph_batch
    else:
        local_start_pos = sum(global_num_tokens[:dp_rank])
    return local_start_pos, local_num_tokens

def _dp_gather_via_all_reduce(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
    cp_shard_counts: Optional[Sequence[int]] = None,
):
    local_start_pos, local_num_tokens = get_dp_local_info(forward_batch)

    global_tokens.fill_(0)
    assert local_tokens.is_contiguous()
    assert global_tokens.is_contiguous()

    # CP ranks hold the same rows of their DP group, and CP rank 0 writes them,
    # unless they pass the counts of their shards; then each writes its own.
    writes = (is_partial or get_parallel().attn_tp_rank == 0) and (
        cp_shard_counts is not None
        or getattr(forward_batch, "cp_local_dp_layout", False)
        or get_parallel().attn_cp_rank == 0
    )

    if local_tokens.shape[0] > 0 and writes:
        assert local_tokens.untyped_storage() is not global_tokens.untyped_storage(), (
            "aliasing between global_tokens and local_tokens not allowed"
        )

        if cp_shard_counts is None:
            memcpy(
                global_tokens, local_tokens, 0, local_start_pos, local_num_tokens, False
            )
        else:
            start, length = _cp_shard_rows(forward_batch, cp_shard_counts)
            global_tokens[start : start + length].copy_(local_tokens[:length])

    # Input IDs are in int 32. We should use inplace_all_reduce for local case because of custom all reduce.
    if world_dp_gather_enabled():
        torch.distributed.all_reduce(
            global_tokens,
            op=torch.distributed.ReduceOp.SUM,
            group=torch.distributed.group.WORLD,
        )
    elif getattr(forward_batch, "cp_local_dp_layout", False):
        global_tokens[:] = get_attn_dp_tp_group().all_reduce(global_tokens)
    else:
        NUM_GPUS_PER_NODE = 8
        if (
            not local_tokens.dtype.is_floating_point
            and get_parallel().tp_size <= NUM_GPUS_PER_NODE
        ):
            from sglang.srt.distributed.parallel_state import inplace_all_reduce

            inplace_all_reduce(
                global_tokens, group_name=get_parallel().tp_group.unique_name
            )

        else:
            global_tokens[:] = tensor_model_parallel_all_reduce(global_tokens)

def _dp_gather(
    global_tokens: torch.Tensor,
    local_tokens: torch.Tensor,
    forward_batch: ForwardBatch,
    is_partial: bool,
    cp_shard_counts: Optional[Sequence[int]] = None,
):
    """Gather each DP group's rows into its slot of the global buffer.

    Under attention CP, without ``cp_shard_counts`` the CP ranks of a DP group
    must hold the same rows, and only CP rank 0's copy is gathered. With it,
    each CP rank holds a different shard of the group's tokens and places its
    own (see dp_gather_partial). A caller whose CP ranks hold different rows
    passes the counts, or restores the full rows on every CP rank first.
    """
    _note_dp_gather_in_prefill_graph()
    if get_parallel().attn_cp_size > 1:
        # Under CP the rows are placed before a sum: an all-gather takes a block
        # from every rank of the TP group, CP ranks included.
        _dp_gather_via_all_reduce(
            global_tokens, local_tokens, forward_batch, is_partial, cp_shard_counts
        )
        return
    if (
        is_dp_gatherv_active()
        and forward_batch.dp_padding_mode is not None
        and not forward_batch.dp_padding_mode.is_max_len()
    ):
        # The gatherv per-rank sizes MUST sum to the pre-allocated global buffer
        # (the MoE runs on the whole buffer, so any unfilled tail = garbage).
        # The buffer was sized from the ceil_align'd global_num_tokens stored via
        # set_dp_buffer_len (forward_batch_info), so the authoritative sizes are
        # get_dp_global_num_tokens() — the SAME source the reduce_scatterv combine
        # uses (symmetric). _dp_gatherv_sizes() reads the raw (un-aligned, and for
        # the MoE-gather context the logprob-token) counts, which do NOT match the
        # buffer for prefill steps -> would force an all_reduce fallback.
        # Prefer the buffer-aligned sizes; fall back to the per-batch sizes only
        # if they happen to match (e.g. the logits gather path).
        _gatherv_sizes = get_dp_global_num_tokens()
        if _gatherv_sizes is None or sum(_gatherv_sizes) != global_tokens.shape[0]:
            _gatherv_sizes = _dp_gatherv_sizes(forward_batch)
        if _gatherv_sizes is not None and sum(_gatherv_sizes) == global_tokens.shape[0]:
            _dp_gather_via_all_gatherv(
                global_tokens, local_tokens, forward_batch, is_partial, _gatherv_sizes
            )
            return
    if (
        forward_batch.dp_padding_mode is not None
        and forward_batch.dp_padding_mode.is_max_len()
    ):
        if get_parallel().attn_cp_size > 1:
            _dp_gather_via_all_reduce(
                global_tokens, local_tokens, forward_batch, is_partial
            )
        else:
            _dp_gather_via_all_gather(
                global_tokens, local_tokens, forward_batch, is_partial
            )
    else:
        _dp_gather_via_all_reduce(
            global_tokens, local_tokens, forward_batch, is_partial
        )

def dp_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor):
    _note_dp_gather_in_prefill_graph()
    if get_parallel().attn_cp_size > 1:
        # MAX_LEN has one equal-sized slot per attention DP.  Combine the TP
        # partials and return the current DP slot to every non-sharded CP copy.
        input = tensor_model_parallel_all_reduce(input)
        local_rows = output.shape[0]
        local_start = get_attention_dp_rank() * local_rows
        output.copy_(input.narrow(0, local_start, local_rows))
        return
    if is_dp_gatherv_active():
        # Variable-length combine matching all_gatherv dispatch: scatter the
        # global (sum_len) tensor back to per-rank token counts. Fall through to
        # the default reduce-scatter path if per-rank sizes are unavailable.
        sizes = get_dp_global_num_tokens()
        if sizes is not None:
            get_parallel().tp_group.reduce_scatterv(input, output=output, sizes=sizes)
            return
    if get_parallel().tp_size == get_parallel().attn_dp_size:
        if _aiter_reduce_scatter_tensor(output, input):
            return
        get_parallel().tp_group.reduce_scatter_tensor(output, input)
    else:
        scattered_local_tokens = input.tensor_split(get_parallel().tp_size)[
            get_parallel().tp_rank
        ]
        get_parallel().tp_group.reduce_scatter_tensor(scattered_local_tokens, input)
        get_parallel().attn_tp_group.all_gather_into_tensor(
            output, scattered_local_tokens
        )

def get_attention_dp_size() -> int:
    return get_parallel().attn_dp_size

def get_attention_dp_rank() -> int:
    return get_parallel().attn_dp_rank

@functools.lru_cache(maxsize=1)
def _dp_use_max_len() -> bool:
    from sglang.srt.environ import envs

    return envs.SGLANG_DP_USE_MAX_LEN.get()

def _aiter_reduce_scatter_tensor(output: torch.Tensor, input: torch.Tensor) -> bool:
    """HCU-only: run the DP-attention MAX_LEN combine over aiter's custom (IPC)
    reduce_scatter. Returns True if it handled the op, False to let the caller
    fall back to `get_parallel().tp_group.reduce_scatter_tensor` (RCCL on HCU).

    `GroupCoordinator._maybe_aiter_reduce_scatter` deliberately excludes HCU
    (`and not _is_hcu`) as a historical precision guard, so the generic path sends
    HCU to RCCL and misses the aiter IPC kernel. This narrow wrapper re-enables the
    aiter kernel ONLY for the HCU DP-attention combine under MAX_LEN, leaving every
    other reduce_scatter_tensor caller on HCU untouched. Mirrors the upstream 0518
    `_aiter_reduce_scatter_tensor`, plus our should_custom_ar / equal-chunk guards.

    Opt-in: needs SGLANG_DP_USE_MAX_LEN=1 (this wrapper) AND SGLANG_USE_AITER_AR=1
    (so ca_comm is the aiter CustomAllreduce that actually has reduce_scatter).
    registered=False is forced: HCU's registered graph-replay path corrupts output
    (same root cause as all_gather_reg; see merge notes §7).
    """
    if not (_is_hcu and _dp_use_max_len()):
        return False
    # aiter custom comm only supports fp16/bf16/fp32.
    if input.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        return False
    if output.dtype != input.dtype:
        return False
    if not (input.is_contiguous() and output.is_contiguous()):
        return False
    ca = getattr(get_parallel().tp_group, "ca_comm", None)
    if ca is None or getattr(ca, "disabled", True):
        return False
    if not (hasattr(ca, "reduce_scatter") and hasattr(ca, "should_custom_ar")):
        return False
    # should_custom_ar bounds the pre-reduce buffer size and validates topology.
    if not ca.should_custom_ar(input):
        return False
    # Equal-chunk only: the global buffer must split evenly into per-rank output.
    if input.shape[0] != output.shape[0] * get_parallel().tp_size:
        return False
    ca.reduce_scatter(input, output, registered=False)
    return True

def dp_reduce_scatter_tensor_async(
    output_local: torch.Tensor,
    global_tokens: torch.Tensor,
    event_key=("combine_tensor", 0),
) -> torch.cuda.Event:
    """Launch the regular DP reduce-scatter combine on the TBO comm stream.

    Unlike ``dp_reduce_scatterv_async``, this supports the MAX_LEN CUDA-graph
    layout and hybrid attention-TP layouts.  In the latter case
    ``dp_reduce_scatter_tensor`` performs the TP reduce-scatter followed by the
    attention-TP all-gather needed to reconstruct TP_ATTN_FULL.
    """
    comm = get_dp_tbo_comm_stream()
    compute = torch.cuda.current_stream()
    ev = _tbo_event(event_key)
    with torch.cuda.stream(comm):
        comm.wait_stream(compute)
        dp_reduce_scatter_tensor(output_local, global_tokens)
        ev.record(comm)
    return ev

def attn_cp_all_to_all_single(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes: Optional[List[int]] = None,
    input_split_sizes: Optional[List[int]] = None,
):
    """All-to-all over dim 0 within the attention CP group (variable splits ok).

    Used by the DSV4 compressor RLC path to repartition round-robin-scattered
    tokens into per-rank contiguous blocks.
    """
    return get_parallel().attn_cp_group.all_to_all_single(
        output,
        input,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
    )

def attn_cp_overlap_all_gather_into_tensor(output: torch.Tensor, input: torch.Tensor):
    return get_parallel().attn_cp_group.all_gather_into_tensor(output, input)

def get_dp_padding_mode(
    cls, is_extend_in_batch, global_num_tokens: List[int]
) -> DpPaddingMode:
    gather_width = dp_gather_width()

    # (trangdough) pplx-kernels a2a is a symmetric collective: every EP rank
    # must dispatch the same number of tokens or the device-side handshake
    # deadlocks (idle DP ranks with 0 tokens never signal their peers).
    # Force MAX_LEN so all ranks are padded to equal token counts.
    from sglang.srt.layers.moe.utils import get_moe_a2a_backend

    moe_a2a_backend = get_moe_a2a_backend()
    if moe_a2a_backend.is_pplx():
        return DpPaddingMode.MAX_LEN

    if moe_a2a_backend.is_deepep_v2() and envs.SGLANG_DEEPEP_V2_FORCE_MAX_LEN.get():
        return DpPaddingMode.MAX_LEN

    # Hybrid-SSM/linear models materialize idle ranks through the existing
    # MAX_LEN fabricated-row path. This must also cover decode idle
    # participation, not only extend batches: DSpark target verification
    # invokes the eager path for idle DP ranks when CUDA graph is disabled.
    if (
        gather_width > 1
        and get_flags().dp.max_len_with_idle
        and min(global_num_tokens) == 0
    ):
        return DpPaddingMode.MAX_LEN

    # When is_extend_in_batch and the gather spans several DP ranks, use
    # SUM_LEN to avoid padding overhead from uneven token distribution.
    # Over one rank, max_len equals sum_len, so prefer MAX_LEN mode
    # to enable symmetric memory optimization (needed for DSA CP, etc.).
    if is_extend_in_batch and gather_width > 1:
        return DpPaddingMode.SUM_LEN

    # we choose the mode that minimizes the communication cost
    # prefer MAX_LEN when communication cost is equal to enable symmetric memory
    max_len = max(global_num_tokens)
    sum_len = sum(global_num_tokens)
    if sum_len * 2 >= max_len * gather_width:
        return cls.MAX_LEN
    else:
        return cls.SUM_LEN

def get_default_mode_in_cuda_graph(cls) -> DpPaddingMode:
    # TODO(kkhuang-amd): noqa, temporary work-around for rocm 7.0.0 alpha
    # it can be safely removed later, once RCCL fixed
    if _USE_ROCM700A_WA:
        return cls.SUM_LEN
    # HCU keeps SUM_LEN until the aiter graph all-gather is validated; opt
    # into MAX_LEN for all-gather plus fused reduce-scatter.
    if _is_hcu and not _dp_use_max_len():
        return cls.SUM_LEN
    return cls.MAX_LEN

def _around(implementation):
    def hook(original, *args, **kwargs):
        if not is_hcu():
            return original(*args, **kwargs)
        return implementation(*args, **kwargs)
    return hook


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    for name in ['dp_gather_slot', 'set_dp_buffer_len_from_batch', 'init_dp_gathered_buffer', 'get_dp_local_info', 'get_dp_local_slice_cpu', '_dp_gather_via_all_reduce', '_dp_gather', 'dp_reduce_scatter_tensor']:
        HookRegistry.register(f"sglang.srt.layers.dp_attention.{name}", _around(globals()[name]), HookType.AROUND)
    for name in ("get_dp_padding_mode", "get_default_mode_in_cuda_graph"):
        HookRegistry.register(f"sglang.srt.layers.dp_attention.DpPaddingMode.{name}", _around(globals()[name]), HookType.AROUND)
    if is_hcu():
        for name in ['get_attention_dp_size', 'get_attention_dp_rank', '_dp_use_max_len', '_aiter_reduce_scatter_tensor', 'dp_reduce_scatter_tensor_async', 'attn_cp_all_to_all_single', 'attn_cp_overlap_all_gather_into_tensor']:
            setattr(N, name, globals()[name])
