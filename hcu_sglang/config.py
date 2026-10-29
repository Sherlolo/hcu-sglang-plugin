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
