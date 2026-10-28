"""Current HCU gather/upconvert extension over the official packed-cache ABI."""
from __future__ import annotations
import sglang.kernels.ops.attention.dsv4.dequant_k_cache as upstream
globals().update({name:value for name,value in vars(upstream).items() if not name.startswith('__')})

def gather_upconvert_k_cache_paged(
    quant_k_cache: torch.Tensor,
    token_indices: torch.Tensor,
    topk_lengths: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
    compact_indices: Optional[torch.Tensor] = None,
    output_offsets: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather packed DSV4 FP8 KV rows and up-convert them to BF16.

    This is the Triton equivalent of the LightOp decode gather used by SparkV1.
    Each query gets a private, contiguous page in ``out``.  The returned indices
    address that compact cache and can therefore be passed directly to the
    existing BF16 ``flash_mla_with_kvcache`` sparse-decode path.

    Args:
        quant_k_cache: Packed DSV4 cache. Accepted layouts are the raw
            ``(num_pages, bytes_per_page_padded)`` allocation and its FlashMLA
            view ``(num_pages, page_size, 1, 584)``. Page padding is preserved
            through ``stride(0)``.
        token_indices: ``(num_queries, topk)`` or ``(num_queries, 1, topk)``
            physical token locations in ``quant_k_cache``.
        topk_lengths: ``(num_queries,)`` valid prefix lengths.
        page_size: source cache page size.
        out: optional BF16 ``(num_queries, output_topk, 1, 512)`` workspace.
            ``output_topk`` may be larger than the source ``topk`` when two
            caches are packed into one FlashMLA input.
        compact_indices: optional int32 ``(num_queries, 1, output_topk)``
            workspace.
        output_offsets: optional int32 ``(num_queries,)`` destination column
            for each query. This lets a second cache start immediately after
            the first cache's valid prefix instead of after its padded width.
    """
    assert quant_k_cache.dim() in (2, 4), f"unexpected {quant_k_cache.shape=}"
    assert quant_k_cache.is_cuda
    assert token_indices.dtype in (torch.int32, torch.int64)
    if token_indices.dim() == 3:
        assert token_indices.shape[1] == 1
        token_indices_2d = token_indices[:, 0, :]
    else:
        assert token_indices.dim() == 2
        token_indices_2d = token_indices
    token_indices_2d = token_indices_2d.contiguous()

    topk_lengths = topk_lengths.reshape(-1).contiguous()
    assert topk_lengths.dtype == torch.int32
    num_queries, topk = token_indices_2d.shape
    assert topk_lengths.numel() == num_queries
    assert page_size > 0

    quant_k_cache_u8 = quant_k_cache.view(torch.uint8)
    num_pages = quant_k_cache_u8.shape[0]
    page_stride_bytes = quant_k_cache_u8.stride(0)
    assert page_stride_bytes >= page_size * (NOPE_ROPE_BYTES + PADDED_SCALE_PER_TOKEN)

    if out is None:
        output_topk = topk
        out = torch.empty(
            (num_queries, output_topk, 1, DIM_NOPE + DIM_ROPE),
            dtype=torch.bfloat16,
            device=quant_k_cache.device,
        )
    else:
        assert out.shape[0] == num_queries
        assert out.shape[2:] == (1, DIM_NOPE + DIM_ROPE)
        assert out.dtype == torch.bfloat16
        output_topk = out.shape[1]
        assert output_topk >= topk

    if compact_indices is None:
        compact_indices = torch.empty(
            (num_queries, 1, output_topk),
            dtype=torch.int32,
            device=token_indices.device,
        )
    else:
        assert compact_indices.shape == (num_queries, 1, output_topk)
        assert compact_indices.dtype == torch.int32

    use_output_offsets = output_offsets is not None
    if output_offsets is None:
        # The kernel ignores this pointer when USE_OUTPUT_OFFSETS=False.
        output_offsets = topk_lengths
    else:
        output_offsets = output_offsets.reshape(-1).contiguous()
        assert output_offsets.dtype == torch.int32
        assert output_offsets.numel() == num_queries

    if num_queries == 0 or topk == 0:
        return out, compact_indices

    # Keep the original page stride.  The 4-D FlashMLA view excludes the
    # per-page padding from its logical shape and is therefore non-contiguous;
    # reshape() would silently copy it and invalidate PAGE_STRIDE_BYTES.
    buf_fp8 = quant_k_cache_u8.view(fp8_dtype)
    buf_bf16 = quant_k_cache_u8.view(torch.bfloat16)
    buf_uint8 = quant_k_cache_u8
    _gather_upconvert_k_cache_paged_kernel[(num_queries, topk)](
        out,
        compact_indices,
        buf_fp8,
        buf_bf16,
        buf_uint8,
        token_indices_2d,
        topk_lengths,
        output_offsets,
        out.stride(0),
        out.stride(1),
        compact_indices.stride(0),
        compact_indices.stride(2),
        token_indices_2d.stride(0),
        token_indices_2d.stride(1),
        PAGE_STRIDE_BYTES=page_stride_bytes,
        PAGE_SIZE=page_size,
        NUM_PAGES=num_pages,
        OUTPUT_TOPK=output_topk,
        USE_OUTPUT_OFFSETS=use_output_offsets,
        DIM_NOPE=DIM_NOPE,
        DIM_ROPE=DIM_ROPE,
        TILE_SIZE=TILE_SIZE,
        NUM_SCALE_TILES=NUM_SCALE_TILES,
        NOPE_ROPE_BYTES=NOPE_ROPE_BYTES,
        PADDED_SCALE_PER_TOKEN=PADDED_SCALE_PER_TOKEN,
        S_OFFSET_BYTES=page_size * NOPE_ROPE_BYTES,
    )
    return out, compact_indices


@triton.jit
def _gather_upconvert_k_cache_paged_kernel(
    output_ptr,
    compact_indices_ptr,
    buf_fp8_ptr,
    buf_bf16_ptr,
    buf_uint8_ptr,
    token_indices_ptr,
    topk_lengths_ptr,
    output_offsets_ptr,
    output_stride_0,
    output_stride_1,
    compact_indices_stride_0,
    compact_indices_stride_2,
    indices_stride_0,
    indices_stride_1,
    PAGE_STRIDE_BYTES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_PAGES: tl.constexpr,
    OUTPUT_TOPK: tl.constexpr,
    USE_OUTPUT_OFFSETS: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_SCALE_TILES: tl.constexpr,
    NOPE_ROPE_BYTES: tl.constexpr,
    PADDED_SCALE_PER_TOKEN: tl.constexpr,
    S_OFFSET_BYTES: tl.constexpr,
):
    # HCU Triton otherwise keeps program IDs and runtime strides in i32.  For
    # the full CP-local prefill batch, query_id * output_stride_0 can exceed
    # 2**31 elements (notably the combined SWA+C128 BF16 workspace), wrapping
    # the destination address and causing an HSA VMFault.  Promote every index
    # participating in pointer arithmetic before the multiplication.
    query_id = tl.program_id(0).to(tl.int64)
    topk_id = tl.program_id(1).to(tl.int64)
    valid_length = tl.load(topk_lengths_ptr + query_id).to(tl.int64)
    output_offset = query_id * 0
    if USE_OUTPUT_OFFSETS:
        output_offset = tl.load(output_offsets_ptr + query_id).to(tl.int64)
    output_col = output_offset + topk_id
    in_valid_prefix = topk_id < valid_length
    loc = tl.load(
        token_indices_ptr + query_id * indices_stride_0 + topk_id * indices_stride_1,
        mask=in_valid_prefix,
        other=0,
    ).to(tl.int64)
    valid = in_valid_prefix & (loc >= 0) & (loc < NUM_PAGES * PAGE_SIZE)
    safe_loc = tl.where(valid, loc, 0)
    page_idx = safe_loc // PAGE_SIZE
    in_page = safe_loc % PAGE_SIZE
    page_byte_base = page_idx * PAGE_STRIDE_BYTES
    token_data_base = page_byte_base + in_page * NOPE_ROPE_BYTES
    token_scale_base = (
        page_byte_base + S_OFFSET_BYTES + in_page * PADDED_SCALE_PER_TOKEN
    )
    out_row_base = query_id * output_stride_0 + output_col * output_stride_1

    nope_offs = tl.arange(0, TILE_SIZE)
    for tile_id in tl.static_range(NUM_SCALE_TILES):
        fp8_off = token_data_base + tile_id * TILE_SIZE + nope_offs
        fp8_vals = tl.load(buf_fp8_ptr + fp8_off, mask=valid, other=0.0).to(tl.float32)
        scale_u8 = tl.load(
            buf_uint8_ptr + token_scale_base + tile_id,
            mask=valid,
            other=127,
        ).to(tl.int32)
        scale_pow2 = tl.exp2((scale_u8 - 127).to(tl.float32))
        out_off = out_row_base + tile_id * TILE_SIZE + nope_offs
        tl.store(
            output_ptr + out_off,
            (fp8_vals * scale_pow2).to(output_ptr.dtype.element_ty),
        )

    rope_offs = tl.arange(0, DIM_ROPE)
    bf16_off = (token_data_base + DIM_NOPE) // 2 + rope_offs
    rope_data = tl.load(buf_bf16_ptr + bf16_off, mask=valid, other=0.0)
    tl.store(output_ptr + out_row_base + DIM_NOPE + rope_offs, rope_data)

    compact_loc = (query_id * OUTPUT_TOPK + output_col).to(tl.int32)
    tl.store(
        compact_indices_ptr
        + query_id * compact_indices_stride_0
        + output_col * compact_indices_stride_2,
        tl.where(in_valid_prefix, compact_loc, 0),
    )


