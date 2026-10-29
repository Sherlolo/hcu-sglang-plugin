"""Current fused compressor using owner-aware main-KV write views."""
from __future__ import annotations
import sglang.srt.layers.attention.deepseek_v4_backend as upstream
globals().update({name:value for name,value in vars(upstream).items() if not name.startswith('__')})

def _low_ratio_compress_fused(
    self, layer, x, req, pos, *, draft_len=1, extend_offsets=None
) -> None:
    """Decode or target-verify; an extend batch passes ``extend_offsets``, the int32
    (first row, row count) of each request."""
    from sglang.kernels.ops.attention.dsv4.low_ratio_compress import (
        c1_decode_norm_rope_store,
        c2_decode_norm_rope_store,
        c2_prefill_norm_rope_store,
    )
    # The JIT writer has a portable HIP FP4 encoder as well as the CUDA
    # intrinsic path. Keep norm, RoPE, both quantization stages and the
    # paged cache store in one launch on both backends.
    from sglang.kernels.ops.attention.dsv4.fp4_indexer_rope import (
        index_k_norm_rope_pack_store,
    )
    pool = self.token_to_kv_pool
    core = self.forward_metadata.core_metadata
    compressor = layer.compressor
    layer_id = layer.layer_id
    # Contiguous complex64 freqs_cis gives a real/imag-interleaved view without copying.
    freqs_cis = torch.view_as_real(layer.freqs_cis).flatten(-2)
    kv_cache = pool.get_extra_key_buffer_for_write(layer_id)
    page_size = pool.get_extra_key_page_size(layer_id)
    # The pool's page format: V4, or the V4.1 fp8 / fp4 layouts.
    kv_layout = (
        KVLayout.UNIFORM_FP8
        if pool.uniform_fp8
        else pool.get_extra_key_layout(layer_id)
    )
    if layer.compress_ratio == 1:
        # At ratio 1, c1_out_loc equals the int64 raw_out_loc supplied by the scheduler.
        latent = c1_decode_norm_rope_store(
            compressor.wkv(x),
            compressor.norm.weight.data,
            pos,
            core.raw_out_loc,
            compressor.norm.eps,
            freqs_cis,
            kv_cache,
            page_size=page_size,
            layout=kv_layout,
        )
        out_loc = core.c1_out_loc
    else:
        # CompressStatePool stores each request's pending pairs in a position ring.
        # KVAndScore rows use | kv | score |, addressed as req * ring_size + pos % ring_size.
        state = pool.get_attention_compress_states(layer_id)
        args = (
            compressor.project_fused(x),
            state.kv_score_buffer.kv_score,
            compressor.norm.weight.data,
            pos,
            req,
            core.raw_out_loc,
            compressor.norm.eps,
            freqs_cis,
            kv_cache,
        )
        if extend_offsets is None:
            latent = c2_decode_norm_rope_store(
                *args,
                page_size=page_size,
                ring_size=state.ring_size,
                draft_len=draft_len,
                layout=kv_layout,
            )
        else:
            latent = c2_prefill_norm_rope_store(
                *args,
                *extend_offsets,
                page_size=page_size,
                ring_size=state.ring_size,
                layout=kv_layout,
            )
        out_loc = core.c2_out_loc

    indexer = layer.indexer
    if indexer is not None and indexer.owns_k:
        # out_loc is -1 for an incomplete group and 0 for padding;
        # the kernel suppresses both stores.
        assert out_loc is not None
        if pool.low_ratio_index_k_is_split(layer_id):
            # ROCm keeps the index-K payload and scales in two buffers
            from sglang.srt.layers.attention.dsv4.low_ratio_backend_hip import (
                store_index_k_split,
            )

            return store_index_k_split(
                pool, layer, indexer, latent, freqs_cis, pos, out_loc, layer_id
            )
        index_k_norm_rope_pack_store(
            indexer.forward_wk(latent),
            indexer.k_norm.weight.data,
            indexer.k_norm.eps,
            freqs_cis,
            pos,
            out_loc,
            pool.get_index_k_with_scale_buffer(layer_id),
            ratio=layer.compress_ratio,
        )


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from . import low_ratio
    base='sglang.kernels.ops.attention.dsv4.low_ratio_compress.'
    for name in ('c1_decode_norm_rope_store','c2_decode_norm_rope_store','c2_prefill_norm_rope_store'):
        HookRegistry.register(base+name,getattr(low_ratio,name),HookType.REPLACE)
    HookRegistry.register('sglang.srt.layers.attention.deepseek_v4_backend.DeepseekV4AttnBackend._low_ratio_compress_fused',
        _low_ratio_compress_fused,HookType.REPLACE)
