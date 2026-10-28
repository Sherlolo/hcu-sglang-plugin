"""Diagnostics hooks: plugin activation banner and minimal trace hooks.

All hooks are switchable and fail-closed towards upstream behaviour:
no hook ever alters return values, and diagnostics failures are logged,
never raised into engine startup.
"""

import logging
import os
import sys

from sglang_hcu_plugin._version import __version__

logger = logging.getLogger("sglang_hcu_plugin")

BANNER_ENV = "HCU_PLUGIN_BANNER"  # default on; set to 0 to silence
TRACE_ENV = "HCU_PLUGIN_TRACE_ADD_PREFIX"  # default off


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.__post_init__",
        _banner_after_post_init,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.utils.common.add_prefix",
        _trace_add_prefix,
        HookType.AROUND,
    )


def load_banner() -> None:
    """Printed at plugin load time (before ServerArgs exists).

    ``print(flush=True)`` because logging is not configured yet at that
    stage; one line per process keeps activation auditable in server logs.
    """
    if os.environ.get(BANNER_ENV, "1") != "1":
        return
    print(
        f"[sglang-hcu-plugin {__version__}] loaded pid={os.getpid()} argv0={sys.argv[0]}",
        flush=True,
    )


def _banner_after_post_init(result=None, *args, **kwargs):
    """AFTER hook on ServerArgs.__post_init__: log the resolved HCU runtime."""
    if os.environ.get(BANNER_ENV, "1") != "1":
        return None
    try:
        server_args = args[0] if args else None
        model_path = getattr(server_args, "model_path", "?")
        tp_size = getattr(server_args, "tp_size", "?")
        attention_backend = getattr(server_args, "attention_backend", None)
        lines = [
            f"[sglang-hcu-plugin {__version__}] ServerArgs ready: "
            f"model={model_path} tp={tp_size} attention_backend={attention_backend}",
            f"[sglang-hcu-plugin {__version__}] adapter shadow sgl_kernel -> {_sgl_kernel_origin()}",
        ]
        for line in lines:
            logger.info(line)
            print(line, flush=True)
    except Exception:  # diagnostics must never break startup
        logger.exception("hcu-plugin banner failed")
    return None


def _trace_add_prefix(original_fn, *args, **kwargs):
    """AROUND hook proving the hook chain wraps real SGLang calls."""
    result = original_fn(*args, **kwargs)
    if os.environ.get(TRACE_ENV, "0") == "1":
        logger.info("[sglang-hcu-plugin] add_prefix(%r, %r) -> %r", args, kwargs, result)
    return result


def _sgl_kernel_origin() -> str:
    try:
        import sgl_kernel

        return getattr(sgl_kernel, "__file__", None) or "(namespace package)"
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"import failed: {exc}"
