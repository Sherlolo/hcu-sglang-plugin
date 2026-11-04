from __future__ import annotations

from sglang.srt.arg_groups.overrides import resolved_view
from sglang.srt.runtime_context import get_server_args
from sglang.srt.configs.model_config import get_dsa_index_kpool
from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda
from hcu_sglang.utils.detection import is_hcu


def is_glm5_next_hcu(config) -> bool:
    from hcu_sglang.utils.detection import is_hcu

    config = getattr(config, "text_config", config)
    return is_hcu() and getattr(config, "model_type", "") in (
        "glm5_next",
        "glm5v_next",
        "glm5next_text",
        "glm5_next_text",
    )


def get_glm5_next_runtime_args():
    return resolved_view(get_server_args())


def should_remap_pd_dsa_seed_to_local_slots(
    server_args: "ServerArgs", model_config=None
) -> bool:
    """Whether a PD IndexShare seed can be localized for fused TopK.

    Prefill transfers request-relative positions.  A decode worker can safely
    turn those positions into its own allocator slots only when it owns the
    complete, non-CP page table and does not need a HiSparse page swap.
    """

    hf_config = getattr(model_config, "hf_config", None)
    hcu_glm5_no_kpool = (
        is_glm5_next_hcu(hf_config) and get_dsa_index_kpool(hf_config) <= 1
    )
    return (
        (is_cuda() or is_hcu())
        and envs.SGLANG_DSA_FUSE_TOPK.get()
        and server_args.disaggregation_mode == "decode"
        and not server_args.enable_hisparse
        and server_args.attn_cp_size == 1
        and not hcu_glm5_no_kpool
    )
