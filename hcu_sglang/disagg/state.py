"""V4 descriptor production at the public SGLang state-assembly boundary."""
from __future__ import annotations
from hcu_sglang.mem_cache.hcu_v4_pool import DeepSeekV4TokenToKVPool
from hcu_sglang.mem_cache.cp_cache_layer_split.transfer import configure_v4_transfer
from .protocol import StateType


def around_setup_state_kv_args(original, kv_args, token_to_kv_pool,
        draft_token_to_kv_pool=None, total_kv_layers=None, req_to_token_pool=None,
        pd_hidden_pool=None):
    result = original(kv_args, token_to_kv_pool, draft_token_to_kv_pool,
        total_kv_layers, req_to_token_pool, pd_hidden_pool)
    kv_args.kv_layout = None
    kv_args.cp_cache_layer_split = False
    kv_args.v4_transfer_metadata = {}
    if isinstance(token_to_kv_pool, DeepSeekV4TokenToKVPool):
        kv_args.kv_layout = token_to_kv_pool.pd_kv_layout()
        configure_v4_transfer(kv_args, token_to_kv_pool, draft_token_to_kv_pool)
    return result
