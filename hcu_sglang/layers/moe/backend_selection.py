from sglang.srt.layers.moe.utils import get_moe_a2a_backend, get_moe_runner_backend
from sglang.srt.environ import envs


def will_use_aiter_moe() -> bool:
    """Return whether the effective HIP MoE runner is AITER.

    An explicit ``--moe-runner-backend aiter`` selection takes precedence
    over the broad ``SGLANG_USE_AITER`` default. DAS also honors
    ``SGLANG_ROCM_USE_AITER_MOE`` for the channelwise W8A8 path.

    The 128-aligned AITER weight-padding rewrite from upstream #36601 is
    intentionally not applied here: HCU Flash-Next channelwise experts use
    N1=160 no-shuffle ASM configs.
    """
    from hcu_sglang.utils.detection import is_hcu

    if not is_hcu():
        return False
    backend = get_moe_runner_backend()
    if backend.is_aiter():
        a2a_backend = get_moe_a2a_backend()
        if not a2a_backend.supports_aiter():
            raise ValueError(
                "moe_runner_backend=aiter is incompatible with "
                f"moe_a2a_backend={a2a_backend.value}."
            )
        if not a2a_backend.is_none() and not envs.SGLANG_USE_AITER.get():
            raise ValueError(
                "Explicit moe_runner_backend=aiter without SGLANG_USE_AITER=1 "
                "is currently supported only with moe_a2a_backend=none; "
                f"got moe_a2a_backend={a2a_backend.value}."
            )
        return True
    if not backend.is_auto():
        return False
    from sglang.srt.utils import get_bool_env_var

    return (
        envs.SGLANG_USE_AITER.get()
        or envs.SGLANG_INT4_WEIGHT.get()
        or get_bool_env_var("SGLANG_ROCM_USE_AITER_MOE")
    ) and get_moe_a2a_backend().supports_aiter()
