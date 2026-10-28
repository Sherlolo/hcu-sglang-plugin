"""RLC (Repartition-Local Compression) metadata for the DeepSeek-V4 c4 compressor.

Pure-CPU metadata only: block partition / all-to-all routing / halo / local compress-event
structure / expansion rows. No communication and no kernel calls -- those live in the consumer
`compressor.py::_forward_rlc`.

Dataflow this enables (per prefill chunk, over the chunk's `extend` tokens): round-robin-scattered
tokens are all-to-all'd into "one contiguous block-run per rank + one leading halo block", each rank
compresses its blocks locally, then an all-gather recovers the full compact output in global block
order.

Supported (unlike the previous version, which asserted `L % ratio == 0` and ignored the state pool):
  * NON-ALIGNED sequence lengths -- blocks use CEIL division for partitioning; compress events use
    FLOOR (a partial tail < ratio carries no event). The partial-tail tokens are still routed but
    produce no compressed output; downstream the c4 `out_loc` sends them to the reserved dummy slot 0
    (they are never read), so no partial-tail block value is needed.
  * PREFIX > 0 (chunked/continuation prefill) -- this module only needs the chunk's `extend_lens`
    for the block geometry; `local_events` carries an `at_boundary` flag so the consumer can patch
    each sequence's extend-FIRST block's overlap from the (replicated) state pool. `prefix_lens`
    itself is consumed in `compressor.py`, not here.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

def get_rlc_index(extend_lens: List[int], cp_size: int, ratio: int) -> List[List[int]]:
    """Partition the global compression blocks across CP ranks; return, per rank, the list of GLOBAL
    flat token indices it owns.

    Each rank owns a CONTIGUOUS run of the global block sequence (front-loaded: the first
    `block_sum % cp_size` ranks get one extra block). When a rank's first owned block is mid-sequence
    (i.e. not a sequence's first block) and `ratio == 4`, one leading "halo" block is prepended so the
    rank can compute that block's c4 overlap locally; the consumer drops the halo's output.

    Blocks per sequence use CEIL division, so a non-aligned sequence's partial tail counts as one
    block for partitioning (it carries no compress event later).
    """
    block_size = [(L + ratio - 1) // ratio for L in extend_lens]
    block_size_sum = int(sum(block_size))
    block_num_t = block_size_sum // cp_size
    block_num_f = block_num_t + 1
    device_f = block_size_sum % cp_size
    seq_index_all_new: List[List[int]] = []
    seq_index_start = 0
    block_num_start = 0
    batch_index_now = 0
    for i in range(cp_size):
        seq_index: List[int] = []
        block_num_now = block_num_f if i < device_f else block_num_t
        if block_num_now == 0:
            seq_index_all_new.append(seq_index)
            continue
        if i != 0 and block_num_start != 0 and (ratio == 4):
            seq_index.extend(range(seq_index_start - ratio, seq_index_start))
        block_num_batch = block_size[batch_index_now] - block_num_start
        while block_num_batch <= block_num_now:
            seq_index.extend(range(seq_index_start, seq_index_start + (extend_lens[batch_index_now] - block_num_start * ratio)))
            block_num_now -= block_num_batch
            seq_index_start += extend_lens[batch_index_now] - block_num_start * ratio
            block_num_start = 0
            batch_index_now += 1
            if batch_index_now >= len(extend_lens):
                break
            block_num_batch = block_size[batch_index_now] - block_num_start
        if block_num_now > 0:
            seq_index.extend(range(seq_index_start, seq_index_start + block_num_now * ratio))
            block_num_start += block_num_now
            seq_index_start += block_num_now * ratio
        seq_index_all_new.append(seq_index)
    return seq_index_all_new

@dataclass
class RLCMetadata:
    """Data-INDEPENDENT RLC metadata (depends only on `extend_lens`, `ratio`, `cp_size`, `cp_rank`).
    Compute once per forward and reuse across every c4 layer. Nothing here depends on tensor values.
    The state-pool write indices, the local compress plan, and the prefix-overlap patch descriptors
    are built by the consumer (they need the backend's paged metadata / Triton), not here."""
    cp_size: int
    ratio: int
    n_halo: int
    counts: List[int]
    max_c: int
    send_idx: List[int]
    in_splits: List[int]
    out_splits: List[int]
    recv_perm: List[int]
    ev: List[int]
    local_events: List[Tuple[int, bool, int, bool]]
    local_extend_lens: List[int]
    local_seg_seqs: List[int]
    local_seg_boundary: List[bool]
    num_blocks: int

def compute_rlc_metadata(extend_lens: List[int], ratio: int, cp_size: int, cp_rank: int) -> RLCMetadata:
    """Compute the RLC routing / halo / local-event / expansion metadata for one prefill chunk.

    `extend_lens` are the GLOBAL per-sequence extend (this-chunk) lengths -- every sequence, not just
    this rank's shard. Round-robin rule: global flat token `g` belongs to `rank = g % cp_size`; the
    sender maps `g` -> its local shard row `g // cp_size`.
    """
    R = ratio
    lens = [int(x) for x in extend_lens]
    seq_index_all_new = get_rlc_index(lens, cp_size, R)
    event_pos: List[int] = []
    off = 0
    for L in lens:
        for k in range(L // R):
            event_pos.append(off + (k + 1) * R - 1)
        off += L

    def n_halo_of(rr: int) -> int:
        idx = seq_index_all_new[rr]
        if rr == 0 or not idx:
            return 0
        prev = seq_index_all_new[rr - 1]
        return 1 if prev and idx[0] <= prev[-1] else 0

    def owned_count(rr: int) -> int:
        idx = seq_index_all_new[rr]
        if not idx:
            return 0
        olo = idx[0] + n_halo_of(rr) * R
        ohi = idx[-1] + 1
        return sum((1 for e in event_pos if olo <= e < ohi))
    send_idx: List[int] = []
    in_splits: List[int] = []
    for d in range(cp_size):
        ids = [g for g in seq_index_all_new[d] if g % cp_size == cp_rank]
        in_splits.append(len(ids))
        send_idx.extend((g // cp_size for g in ids))
    my_recv = seq_index_all_new[cp_rank]
    out_splits = [sum((1 for g in my_recv if g % cp_size == s)) for s in range(cp_size)]
    recv_order_ids = [g for s in range(cp_size) for g in my_recv if g % cp_size == s]
    recv_perm = sorted(range(len(recv_order_ids)), key=lambda i: recv_order_ids[i])
    n_halo = n_halo_of(cp_rank)
    local_events: List[Tuple[int, bool, int, bool]] = []
    local_extend_lens: List[int] = []
    local_seg_seqs: List[int] = []
    local_seg_boundary: List[bool] = []
    if my_recv:
        (lo, hi) = (my_recv[0], my_recv[-1] + 1)
        off_g = 0
        loc = 0
        for (b, L) in enumerate(lens):
            seg_lo = max(lo, off_g)
            seg = min(hi, off_g + L) - seg_lo
            if seg > 0:
                at_boundary = seg_lo == off_g
                local_extend_lens.append(seg)
                local_seg_seqs.append(b)
                local_seg_boundary.append(at_boundary)
                for j in range(seg):
                    if (j + 1) % R == 0:
                        local_events.append((loc + j, j < 2 * R - 1, b, at_boundary))
                loc += seg
            off_g += L
    counts = [owned_count(rr) for rr in range(cp_size)]
    max_c = max(counts) if counts else 0
    ev = sorted(event_pos)
    return RLCMetadata(cp_size=cp_size, ratio=R, n_halo=n_halo, counts=counts, max_c=max_c, send_idx=send_idx, in_splits=in_splits, out_splits=out_splits, recv_perm=recv_perm, ev=ev, local_events=local_events, local_extend_lens=local_extend_lens, local_seg_seqs=local_seg_seqs, local_seg_boundary=local_seg_boundary, num_blocks=len(event_pos))
