from __future__ import annotations
import logging
import warnings
from dataclasses import dataclass, field, fields
from typing import Any, Iterator, List, Optional
import torch
from sglang.srt.environ import envs
from hcu_sglang.models.deepseek.current_v4.hcu_backend.mqa_logits_utils import mqa_logits_budget_bytes, mqa_logits_needs_budget_check, mqa_logits_row_bytes, mqa_logits_rows_per_chunk, mqa_logits_should_chunk
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import is_in_breakable_cuda_graph
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import is_in_tc_piecewise_cuda_graph
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from hcu_sglang.utils.detection import is_dcu as is_hcu
from sglang.srt.utils import is_hip, is_sm120_supported, is_xpu
logger = logging.getLogger(__name__)
_is_hcu = is_hcu()
_IS_SM120 = is_sm120_supported()
'\nSome comments on the common terms used in DeepSeekV4Backend:\n\ntopk_lengths:\n    NOTE: TL;DR: topk_lengths == seq_lens\n    The FlashMLA sparse decode kernel will attend to `k` tokens for each query.\n    `topk_lengths` indicates how many tokens each query will attend to.\n    This should be named as `seq_lens`, but we simply follow the naming convention.\n\npage_table:\n    The page table indicates which pages each request is assigned to.\n    Each value in the page table is the page index in the TokenToKVPool.\n    This page index is irrelevant to the actual `page_size`.\n\npage_indices:\n    The real indices used to index into the KV cache.\n    This can be computed from the `page_table` and `page_size`.\n    e.g. page_indices[i, j] = page_table[i, j // page_size] * page_size + (j % page_size)\n    For sparse C4 top-512 attention, the indices will be selected from the C4 page indices.\n    In implementation, we don\'t materialize the full C4 `page_indices`,\n    but calculate them from `page_table` on-the-fly in the attention kernel.\n\npositions:\n    The position of the last token for each request.\n    For compress token, the positions must be times of compress ratio.\n    For example, for C4, raw_position=11 will trigger a compression,\n    But the RoPE\'s position, during compression, must be 8 instead of 11.\n\nSome other notes:\n    c4_ / c128_: means "compressed by 4" / "compressed by 128".\n    compressed_page_size: physical indexer pool page size\n    compressed_seq_lens: seq_lens // 4, but bounded by at least 1, due to flash_mla requirement.\n    c4_sparse: means "compressed by 4" but only attend to top-512 tokens.\n               all related length will be clipped to 512.\n'
_LARGE_INDEXER_QUERY_THRESHOLD = 11673
_SM120_INDEXER_M_CHUNK = 4096

def plan_indexer_row_chunks(*, num_rows: int, num_cols: int, budget_bytes: Optional[int], sm120_row_cap: Optional[int]) -> Optional[int]:
    """Query rows per paged-indexer chunk; None runs the whole batch in one call.

    The fp32 logits are [num_rows, num_cols] per layer, so the chunk is the
    smaller of the SM120 kernel cap and what the memory budget allows.
    """
    rows_per_chunk = None
    if sm120_row_cap is not None and num_rows > sm120_row_cap:
        rows_per_chunk = sm120_row_cap
    if budget_bytes is not None:
        (need_chunk, budget_bytes) = mqa_logits_should_chunk(num_rows=num_rows, num_cols=num_cols, get_budget_bytes=lambda : budget_bytes, rocm=is_hip())
        by_budget = mqa_logits_rows_per_chunk(num_rows=num_rows, row_bytes=mqa_logits_row_bytes(num_cols), budget_bytes=budget_bytes) if need_chunk else None
        if by_budget is not None:
            rows_per_chunk = by_budget if rows_per_chunk is None else min(rows_per_chunk, by_budget)
    return rows_per_chunk

def iter_row_chunks(*, num_rows: int, rows_per_chunk: Optional[int]) -> Iterator[slice]:
    if rows_per_chunk is None or rows_per_chunk >= num_rows:
        yield slice(0, num_rows)
        return
    for start in range(0, num_rows, rows_per_chunk):
        yield slice(start, min(start + rows_per_chunk, num_rows))

def copy_metadata(*, src, dst, check_eq_fields: List[str], copy_fields: List[str], assign_fields: Optional[List[str]]=None):
    assign_fields = assign_fields or []
    for field_name in check_eq_fields:
        src_val = getattr(src, field_name)
        dst_val = getattr(dst, field_name)
        assert src_val == dst_val, f'field_name={field_name!r} src_val={src_val!r} dst_val={dst_val!r}'
    for field_name in copy_fields:
        src_val = getattr(src, field_name)
        dst_val = getattr(dst, field_name)
        if src_val is None and dst_val is None:
            continue
        assert dst_val is not None, f'field_name={field_name!r} src_val={src_val!r} dst_val={dst_val!r}'
        if hasattr(dst_val, 'copy_'):
            dst_val.copy_(src_val)
        else:
            warnings.warn(f'field_name={field_name!r} type(dst_val)={type(dst_val)!r} does not have copy_, use setattr')
            setattr(dst, field_name, src_val)
    for field_name in assign_fields:
        setattr(dst, field_name, getattr(src, field_name))
    provided_fields = check_eq_fields + copy_fields + assign_fields
    provided_fields_unique = set(provided_fields)
    assert len(provided_fields) == len(provided_fields_unique), f'provided_fields={provided_fields!r} has dup'
    all_fields = {f.name for f in fields(src)}
    provided_fields = set(provided_fields)
    assert provided_fields == all_fields, f'provided_fields - all_fields={provided_fields - all_fields!r}, all_fields - provided_fields={all_fields - provided_fields!r}'

@dataclass
class NonPagedIndexerPlan:
    page_table: torch.Tensor
    gather_seq_lens: torch.Tensor
    ks: torch.Tensor
    ke: torch.Tensor
    seq_len_sum: int
    max_seq_len: int
    max_seqlen_k: int
    query_rows: int
    rows_per_chunk: Optional[int] = None

@dataclass
class PagedIndexerMetadata:
    page_size: int
    compressed_page_size: int
    page_table: torch.Tensor
    compressed_seq_lens: torch.Tensor
    use_topk_v2: bool
    force_deep_gemm_metadata: bool = False
    use_prefill_cuda_graph: bool = False
    compress_ratio: int = 4
    row_chunk: int = 0
    rows_per_request: int = 1
    deep_gemm_metadata: Any = field(init=False, repr=False)
    topk_metadata: torch.Tensor = field(init=False, repr=False)
    nonpaged_plan: Optional[NonPagedIndexerPlan] = field(init=False, repr=False, default=None)
    rows_per_chunk: Optional[int] = field(init=False, repr=False, default=None)
    mqa_logits_budget_bytes: Optional[int] = field(init=False, repr=False, default=None)
    topk_metadata_chunks: Optional[List[torch.Tensor]] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if _is_hcu:
            self.deep_gemm_metadata = self._lightop_schedule()
        elif (is_hip() or is_xpu() or envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get() or envs.SGLANG_OPT_USE_AITER_INDEXER.get()) and (not self.force_deep_gemm_metadata):
            self.deep_gemm_metadata = None
        else:
            import deep_gemm
            use_jit_indexer = not self.force_deep_gemm_metadata and (envs.SGLANG_OPT_USE_JIT_INDEXER_METADATA.get() or self.compressed_seq_lens.numel() > _LARGE_INDEXER_QUERY_THRESHOLD)
            if use_jit_indexer:
                from hcu_sglang.models.deepseek.current_v4.hcu_backend.attn import get_paged_mqa_logits_metadata
            else:
                from deep_gemm import get_paged_mqa_logits_metadata
            compressed_seq_lens = self.compressed_seq_lens.to(torch.int32)
            if compressed_seq_lens.dim() == 1:
                compressed_seq_lens = compressed_seq_lens.unsqueeze(-1)
            num_rows = compressed_seq_lens.shape[0]
            self.mqa_logits_budget_bytes = self._mqa_logits_budget(num_rows=num_rows)
            self.rows_per_chunk = plan_indexer_row_chunks(num_rows=num_rows, num_cols=self.max_compressed_seq_len, budget_bytes=self.mqa_logits_budget_bytes, sm120_row_cap=_SM120_INDEXER_M_CHUNK if _IS_SM120 else None)
            if self.rows_per_chunk is not None:
                logger.debug('DSV4 indexer chunks %d query rows x %d compressed cols into %d-row chunks (logits budget %s bytes)', num_rows, self.max_compressed_seq_len, self.rows_per_chunk, self.mqa_logits_budget_bytes)
            if self.row_chunk > 0:
                self.deep_gemm_metadata = torch.stack([get_paged_mqa_logits_metadata(compressed_seq_lens[_s:_s + self.row_chunk], self.compressed_page_size, deep_gemm.get_num_sms()) for _s in range(0, compressed_seq_lens.shape[0], self.row_chunk)])
            elif self.rows_per_chunk is not None:
                self.deep_gemm_metadata = [get_paged_mqa_logits_metadata(compressed_seq_lens[rows], self.compressed_page_size, deep_gemm.get_num_sms()) for rows in iter_row_chunks(num_rows=num_rows, rows_per_chunk=self.rows_per_chunk)]
            else:
                self.deep_gemm_metadata = get_paged_mqa_logits_metadata(compressed_seq_lens, self.compressed_page_size, deep_gemm.get_num_sms())
            assert isinstance(self.deep_gemm_metadata, (torch.Tensor, list))
        if self.use_topk_v2:
            from hcu_sglang.models.deepseek.current_v4.hcu_backend.topk import plan_topk_v2
            self.topk_metadata = plan_topk_v2(self.compressed_seq_lens)
            if self.rows_per_chunk is not None:
                self.topk_metadata_chunks = [plan_topk_v2(self.compressed_seq_lens[rows]) for rows in iter_row_chunks(num_rows=self.compressed_seq_lens.shape[0], rows_per_chunk=self.rows_per_chunk)]
        else:
            self.topk_metadata = torch.empty((0,))
        assert self.page_size == 256, 'the system hardcodes page_size=256'
        assert self.compressed_seq_lens.shape[0] % self.rows_per_request == 0, f'self.rows_per_request = {self.rows_per_request!r} must divide self.compressed_seq_lens.shape = {self.compressed_seq_lens.shape!r}'
        assert self.page_size % self.compress_ratio == 0, f'compress_ratio {self.compress_ratio} must divide page_size {self.page_size}'

    def _lightop_schedule(self) -> Optional[torch.Tensor]:
        """LightOp's split-KV schedule for the low-ratio (c1/c2) decode indexer,
        in DeepGEMM's slot: a single tensor, so graph replay copies it in place.
        None elsewhere (c4, row-chunked prefill), where nothing reads it."""
        from hcu_sglang.models.deepseek.current_v4.hcu_backend import lightop_indexer
        if self.compress_ratio not in (1, 2) or self.row_chunk > 0 or self.use_prefill_cuda_graph or (not lightop_indexer.lightop_paged_indexer_available()) or (not lightop_indexer.schedule_rows_supported(self.compressed_seq_lens.numel(), self.rows_per_request)):
            return None
        return lightop_indexer.get_paged_mqa_logits_schedule(context_lens=self.compressed_seq_lens, rows_per_request=self.rows_per_request)

    def _mqa_logits_budget(self, *, num_rows: int) -> Optional[int]:
        """Free-memory budget for this forward's logits; None disables chunking.

        Graph-backed forwards keep a single call: their shapes are fixed at
        capture and the free-memory read would sync the host mid-capture.
        """
        if self.use_prefill_cuda_graph or not self.compressed_seq_lens.is_cuda:
            return None
        if not mqa_logits_needs_budget_check(num_rows=num_rows, num_cols=self.max_compressed_seq_len):
            return None
        if get_is_capture_mode() or torch.cuda.is_current_stream_capturing() or is_in_breakable_cuda_graph() or is_in_tc_piecewise_cuda_graph():
            return None
        return mqa_logits_budget_bytes(device_index=self.compressed_seq_lens.device.index, allow_sync=True)

    @property
    def max_seq_len(self) -> int:
        return self.page_table.shape[1] * self.page_size

    @property
    def max_compressed_seq_len(self) -> int:
        return self.page_table.shape[1] * self.compressed_page_size

    def row_chunks(self):
        num_rows = self.compressed_seq_lens.shape[0]
        if self.row_chunk > 0:
            rows_per_chunk = self.row_chunk
        elif isinstance(self.deep_gemm_metadata, list):
            assert self.rows_per_chunk is not None, 'chunked DeepGEMM metadata requires rows_per_chunk'
            rows_per_chunk = self.rows_per_chunk
        else:
            return [(slice(0, num_rows), self.deep_gemm_metadata)]
        chunks = [(slice(start, min(start + rows_per_chunk, num_rows)), plan) for (start, plan) in zip(range(0, num_rows, rows_per_chunk), self.deep_gemm_metadata)]
        assert chunks and chunks[-1][0].stop == num_rows, f'chunk schedules do not cover all rows: num_rows={num_rows!r} rows_per_chunk={rows_per_chunk!r} len(chunks)={len(chunks)!r}'
        return chunks

    def copy_(self, other: PagedIndexerMetadata):
        chunked = isinstance(self.deep_gemm_metadata, list) or isinstance(other.deep_gemm_metadata, list)
        if is_hip() and (not _is_hcu) or chunked:
            copy_fields = ['page_table', 'compressed_seq_lens']
            assign_fields = ['deep_gemm_metadata', 'nonpaged_plan']
        else:
            copy_fields = ['page_table', 'compressed_seq_lens', 'deep_gemm_metadata']
            assign_fields = ['nonpaged_plan']
        copy_fields += ['topk_metadata']
        assign_fields += ['rows_per_chunk', 'mqa_logits_budget_bytes', 'topk_metadata_chunks']
        copy_metadata(src=other, dst=self, check_eq_fields=['page_size', 'compressed_page_size', 'compress_ratio', 'row_chunk', 'rows_per_request', 'force_deep_gemm_metadata', 'use_prefill_cuda_graph', 'use_topk_v2'], copy_fields=copy_fields, assign_fields=assign_fields)
        self.nonpaged_plan = None

def maybe_copy_inplace(dst, *, src) -> None:
    assert type(src) == type(dst)
    if dst is not None:
        dst.copy_(src)

def expand_index_page_table(page_table: torch.Tensor, *, full_page_size: int, compress_ratio: int, index_page_size: int) -> torch.Tensor:
    """Block table of a low-ratio indexer-K pool, which pages at `index_page_size`
    slots: [bs, n] -> [bs, n * blocks_per_page] int32. The kernel reads compressed
    slot j at page_table[b, j // index_page_size] * index_page_size + j %
    index_page_size, which after this expansion is the c1/c2 pool slot of the same
    position."""
    slots_per_page = full_page_size // compress_ratio
    assert slots_per_page % index_page_size == 0, f'full_page_size = {full_page_size!r} / compress_ratio = {compress_ratio!r} must be a multiple of index_page_size = {index_page_size!r}'
    blocks_per_page = slots_per_page // index_page_size
    if blocks_per_page == 1:
        return page_table
    (bs, n) = page_table.shape
    base = page_table.to(torch.int64) * blocks_per_page
    offsets = torch.arange(blocks_per_page, device=page_table.device, dtype=torch.int64)
    expanded = base.unsqueeze(-1) + offsets
    return expanded.reshape(bs, n * blocks_per_page).to(torch.int32)
