from hcu_sglang.layers.attention.current_dsa.forward_batch_utils import effective_forward_mode
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from hcu_sglang.models.deepseek.current_v4.hcu_backend.compat import is_dsa_prefill_cp_round_robin_split

def can_dsa_cp_split(seq_len: int, cp_size: int, use_dsa: bool, forward_batch):
    if (
        cp_size <= 1
        or not use_dsa
        or not effective_forward_mode(forward_batch).is_context_parallel_extend()
        or not is_dsa_enable_prefill_cp()
        or sum(forward_batch.extend_seq_lens_cpu) < cp_size
    ):
        return False
    if is_dsa_prefill_cp_round_robin_split():
        # Physical input tensors are padded by ForwardBatch to the CP alignment.
        # The original per-request token count may be uneven across ranks.
        if seq_len % cp_size != 0:
            return False
        cur_cp_seq_len = seq_len // cp_size
    else:
        cur_cp_seq_len = seq_len // (cp_size * 2)
    return cur_cp_seq_len != 0

from sglang.srt.layers.cp.base import get_cp_strategy
from hcu_sglang.models.deepseek.deepseek_nextn.methods.model import enable_cp_v2

def is_cp_active(forward_batch) -> bool:
    """Return whether the current forward batch is running through CP."""
    # HCU defaults to platform CP unless explicitly opted in; NPU/MUSA retain
    # their platform CP implementations. Treating those
    # batches as strategy-CP here shards the model inputs a second time while
    # their attention backends still use the legacy layout.
    if not enable_cp_v2():
        return False

    forward_mode = getattr(forward_batch, "forward_mode", None)
    if forward_mode is None or not forward_mode.is_context_parallel_extend():
        return False

    strategy = get_cp_strategy()
    if strategy is None:
        return False

    input_ids = getattr(forward_batch, "input_ids", None)
    if input_ids is None:
        return False

    # DP attention synchronizes the CP layout before ForwardBatch reaches the
    # model.  An all-zero list means that another non-idle DP replica cannot
    # enter CP for this global forward (for example, a 1-token batch paired
    # with a 16K Prefill).  Every replica must then use the ordinary full-TP
    # layout; allowing only the locally eligible replica to shard would make
    # the model-body collectives disagree.
    global_cp_tokens = getattr(forward_batch, "global_cp_num_tokens_cpu", None)
    if global_cp_tokens is not None and not any(global_cp_tokens):
        return False

    return strategy.can_apply(len(input_ids), forward_batch)

is_cp_v2_active = is_cp_active
