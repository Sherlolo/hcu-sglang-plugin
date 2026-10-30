"""sglang_hcu_plugin: HCU capability plugin for SGLang (v0.5.22+ plugin framework).

Registered in the ``sglang.srt.plugins`` entry-point group; SGLang loads it
before ``ServerArgs`` construction (officially guaranteed ordering in
``entrypoints/engine.py``), so hooks registered here apply to the whole
server lifecycle, including argument post-processing in every process.

First-version scope (integration bring-up):

1. Load banner: print one line per process when the plugin loads, so
   plugin activation is visible in server logs.
2. Environment banner: after ``ServerArgs.__post_init__``, log the HCU
   runtime picture (which ``sgl_kernel`` the adapter shadow library
   resolved to, chosen attention backends).
3. ``add_prefix`` trace hook (off by default, enable with
   ``HCU_PLUGIN_TRACE_ADD_PREFIX=1``): minimal behavioural proof that the
   hook chain really wraps SGLang functions.

Differentiated capabilities (operator fusion, specialized scheduling,
performance kernels calling lightop/aiter) are added incrementally on top
of this skeleton; each must keep upstream behaviour when its switch is off.
"""

from sglang_hcu_plugin._version import __version__

__all__ = ["__version__", "register"]


def register() -> None:
    from sglang_hcu_plugin import diagnostics

    diagnostics.register_hooks()
    diagnostics.load_banner()
