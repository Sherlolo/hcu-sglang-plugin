"""Read parallel state through the pinned SGLang RuntimeContext API."""
from sglang.srt.runtime_context import get_parallel


def get_attn_tensor_model_parallel_rank():
    return get_parallel().attn_tp_rank


def get_attn_tensor_model_parallel_world_size():
    return get_parallel().attn_tp_size


def get_moe_expert_parallel_world_size():
    return get_parallel().moe_ep_size


def get_moe_tensor_parallel_world_size():
    return get_parallel().moe_tp_size


def get_tensor_model_parallel_world_size():
    return get_parallel().tp_size


def get_pp_group():
    return get_parallel().pp_group
