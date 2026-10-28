"""HCU extensions for SGLang server argument choices."""

from __future__ import annotations

from sglang.srt.arg_groups.arg_utils import is_record
from hcu_sglang.config import minimax_opt_enabled

HCU_ATTENTION_BACKEND = "hcu_mla"
HCU_QUANTIZATION_METHODS = ("w4a8_int8",)
HCU_SERVER_ARG_DEFAULTS = {
    "minimax_opt": False,
}


def _extend_unique(values: list[str], additions: tuple[str, ...]) -> None:
    values.extend(value for value in additions if value not in values)


def ensure_choices(server_args) -> bool:
    required = (
        "ATTENTION_BACKEND_CHOICES",
        "DRAFT_ATTENTION_BACKEND_CHOICES",
        "QUANTIZATION_CHOICES",
        "CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS",
    )
    if not all(hasattr(server_args, name) for name in required):
        return False
    for name, default in HCU_SERVER_ARG_DEFAULTS.items():
        if name == "minimax_opt" and is_record(server_args.ServerArgs):
            # A compatibility read view, not an undeclared write to frozen inputs.
            setattr(server_args.ServerArgs, name, property(lambda _record: minimax_opt_enabled()))
        elif not hasattr(server_args.ServerArgs, name):
            setattr(server_args.ServerArgs, name, default)

    _extend_unique(
        server_args.ATTENTION_BACKEND_CHOICES,
        (HCU_ATTENTION_BACKEND,),
    )
    _extend_unique(
        server_args.DRAFT_ATTENTION_BACKEND_CHOICES,
        (HCU_ATTENTION_BACKEND,),
    )
    _extend_unique(
        server_args.QUANTIZATION_CHOICES,
        HCU_QUANTIZATION_METHODS,
    )

    if (
        HCU_ATTENTION_BACKEND
        not in server_args.CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
    ):
        server_args.add_chunked_prefix_cache_attention_backend(HCU_ATTENTION_BACKEND)
    return True


def register() -> None:
    from sglang.srt import server_args

    ensure_choices(server_args)
