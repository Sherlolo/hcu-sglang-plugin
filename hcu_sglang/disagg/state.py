"""V4 descriptor production at the public SGLang state-assembly boundary."""
from __future__ import annotations
from hcu_sglang.mem_cache.hcu_v4_pool import DeepSeekV4TokenToKVPool
from hcu_sglang.mem_cache.cp_cache_layer_split.transfer import configure_v4_transfer
from .protocol import StateType


def around_setup_state_kv_args(original, kv_args, token_to_kv_pool,
        draft_token_to_kv_pool=None, total_kv_layers=None, req_to_token_pool=None):
    result = original(kv_args, token_to_kv_pool, draft_token_to_kv_pool,
        total_kv_layers, req_to_token_pool)
    state_types = kv_args.state_types
    # The official producer has no format list; empty format means the existing ABI.
    kv_args.state_data_formats = ["" for _ in state_types]
    kv_args.kv_layout = None
    kv_args.cp_cache_layer_split = False
    kv_args.v4_transfer_metadata = {}
    if not hasattr(kv_args, "hcu_kda_state_sharded"):
        kv_args.hcu_kda_state_sharded = False
    if isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool):
        kv_args.kv_layout = token_to_kv_pool.pd_kv_layout()
        if isinstance(draft_token_to_kv_pool, DeepSeekV4TokenToKVPool):
            ptrs, spans, items = draft_token_to_kv_pool.get_swa_valid_from_buf_infos()
            if ptrs:
                kv_args.state_types.append(StateType.REQUEST_ROW)
                kv_args.state_data_ptrs.append(ptrs)
                kv_args.state_data_lens.append(spans)
                kv_args.state_item_lens.append(items)
                kv_args.state_data_formats.append("")
                kv_args.state_dim_per_tensor.append([])
                kv_args.state_conv_shard_groups.append([])
                kv_args.state_slice_outer_counts.append([])
                kv_args.state_layer_ids.append([])
        configure_v4_transfer(kv_args, token_to_kv_pool, draft_token_to_kv_pool)
    return result
