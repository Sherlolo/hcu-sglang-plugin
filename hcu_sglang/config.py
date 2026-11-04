"""Plugin-owned process options, inherited by spawned SGLang workers."""
import os

MINIMAX_OPT_ENV = "HCU_PLUGIN_MINIMAX_OPT"


def minimax_opt_enabled():
    raw = os.environ.get(MINIMAX_OPT_ENV, "0").lower()
    if raw not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError(f"{MINIMAX_OPT_ENV}: invalid boolean {raw!r}")
    return raw in {"1", "true", "yes", "on"}


def configure_minimax_opt(value):
    if value is not None:
        os.environ[MINIMAX_OPT_ENV] = "1" if value else "0"


CP_CACHE_LAYER_SPLIT_ENV = "HCU_PLUGIN_CP_CACHE_LAYER_SPLIT"


def cp_cache_layer_split_enabled():
    """Candidate CP selection, inherited by workers; full service remains WIP."""
    raw = os.environ.get(CP_CACHE_LAYER_SPLIT_ENV, "0").lower()
    if raw not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError(f"{CP_CACHE_LAYER_SPLIT_ENV}: invalid boolean {raw!r}")
    return raw in {"1", "true", "yes", "on"}


def configure_cp_cache_layer_split(value):
    if value is not None:
        os.environ[CP_CACHE_LAYER_SPLIT_ENV] = "1" if value else "0"


CUSTOM_ALL_REDUCE_BACKEND_ENV = "HCU_PLUGIN_CUSTOM_ALL_REDUCE_BACKEND"
CUSTOM_ALL_REDUCE_BACKENDS = ("auto", "native", "aiter", "off")


def custom_all_reduce_backend():
    value = os.environ.get(CUSTOM_ALL_REDUCE_BACKEND_ENV, "auto").lower()
    if value not in CUSTOM_ALL_REDUCE_BACKENDS:
        raise ValueError(f"{CUSTOM_ALL_REDUCE_BACKEND_ENV}: invalid backend {value!r}")
    return value


def configure_custom_all_reduce_backend(value):
    if value is not None:
        if value not in CUSTOM_ALL_REDUCE_BACKENDS:
            raise ValueError(f"Invalid custom all-reduce backend: {value!r}")
        os.environ[CUSTOM_ALL_REDUCE_BACKEND_ENV] = value


def resolved_custom_all_reduce_backend(disabled=False):
    from hcu_sglang.utils.detection import is_hcu
    from sglang.srt.utils import get_bool_env_var
    backend = custom_all_reduce_backend()
    if disabled:
        return "off"
    if backend == "auto" and is_hcu() and get_bool_env_var("SGLANG_USE_AITER_AR", default="false"):
        return "aiter"
    return backend


EP_STATIC_DISPATCH_POLICY_ENV = "HCU_PLUGIN_EP_STATIC_DISPATCH_POLICY"


def ep_static_dispatch_policy():
    value = os.environ.get(EP_STATIC_DISPATCH_POLICY_ENV, "nearest")
    if value not in ("nearest", "locality_fair"):
        raise ValueError(f"Invalid static expert dispatch policy: {value!r}")
    return value


def configure_ep_static_dispatch_policy(value):
    if value is not None:
        if value not in ("nearest", "locality_fair"):
            raise ValueError(f"Invalid static expert dispatch policy: {value!r}")
        os.environ[EP_STATIC_DISPATCH_POLICY_ENV] = value


HY3_SP_ENV = "HCU_PLUGIN_HY3_SP"


def hy3_sp_enabled():
    raw = os.environ.get(HY3_SP_ENV, "0").lower()
    if raw not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError(f"{HY3_SP_ENV}: invalid boolean {raw!r}")
    return raw in {"1", "true", "yes", "on"}


def configure_hy3_sp(value):
    if value is not None:
        os.environ[HY3_SP_ENV] = "1" if value else "0"


def minimax_opt_from_config(config):
    return minimax_opt_enabled()


def hy3_sp_from_config(config):
    return hy3_sp_enabled()


def register_parallel_options():
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu():
        return
    from sglang.srt.arg_groups.arg_utils import Derived
    from sglang.srt.arg_groups.fields.parallel import Parallel
    from sglang.srt import runtime_context
    for name, provider in (
        ("minimax_opt", "hcu_sglang.config.minimax_opt_from_config"),
        ("hy3_sp", "hcu_sglang.config.hy3_sp_from_config"),
        ("enable_dsa_prefill_context_parallel", "hcu_sglang.config.dsa_legacy_cp_enabled"),
        ("enable_prefill_context_parallel", "hcu_sglang.config.generic_legacy_cp_enabled"),
        ("dsa_prefill_cp_mode", "hcu_sglang.config.dsa_legacy_cp_mode"),
        ("prefill_cp_mode", "hcu_sglang.config.generic_legacy_cp_mode"),
    ):
        if not hasattr(Parallel, name):
            setattr(Parallel, name, Derived(fn=provider))
    runtime_context._parallel_fields.cache_clear()
    runtime_context._install_parallel_properties()


def _legacy_cp_configured(config):
    return bool(config.enable_prefill_cp and config.cp_strategy is not None)


def dsa_legacy_cp_enabled(config):
    return _legacy_cp_configured(config) and getattr(config, "attention_backend", None) in ("dsa", "dsv4")


def generic_legacy_cp_enabled(config):
    return _legacy_cp_configured(config) and not dsa_legacy_cp_enabled(config)


def dsa_legacy_cp_mode(config):
    if not _legacy_cp_configured(config):
        return "round-robin-split"
    return {"zigzag": "in-seq-split", "interleave": "round-robin-split"}[config.cp_strategy]


def generic_legacy_cp_mode(config):
    if not _legacy_cp_configured(config):
        return "in-seq-split"
    return {"zigzag": "in-seq-split", "interleave": "round-robin-split"}[config.cp_strategy]
