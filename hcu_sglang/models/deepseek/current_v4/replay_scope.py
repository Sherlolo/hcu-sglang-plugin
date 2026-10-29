"""HCU decoder-tail scope and eager cache-store boundary."""
from __future__ import annotations
from contextlib import contextmanager
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import eager_on_graph
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph.context_manager import get_tc_piecewise_forward_context
_in_replay_tail = False

def in_replay_tail_graph() -> bool:
    """True while the tail layers capture or replay a decoder tail graph."""
    return _in_replay_tail


@contextmanager
def _replay_tail_scope():
    global _in_replay_tail
    _in_replay_tail = True
    try:
        yield
    finally:
        _in_replay_tail = False


def _tail_kv_store(attention, x, positions, qkv_a) -> None:
    # The SWA store writes the step's live tail slots, so a tail graph breaks here.
    forward_batch = get_tc_piecewise_forward_context().forward_batch
    num_rows = forward_batch.global_num_token_non_padded_cpu
    if num_rows == 0:
        return
    attention._compute_kv_to_cache(
        x[:num_rows],
        positions[:num_rows],
        forward_batch,
        get_attn_backend(),
        qkv_a=None if qkv_a is None else qkv_a[:num_rows],
    )


bcg_tail_kv_store = eager_on_graph(True)(_tail_kv_store)
