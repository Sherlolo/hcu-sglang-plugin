#!/usr/bin/env python3
"""Verify real SGLang loading, source identities and complete Hook application."""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
from pathlib import Path
from unittest.mock import patch


class PluginErrors(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR)
        self.messages = []

    def emit(self, record):
        if record.name.startswith("sglang.srt.plugins"):
            self.messages.append(record.getMessage())


def validate(official_python, plugin_root, *, disable_entry_points=False,
             expected_targets=None):
    import sglang
    import sglang_dcu_plugin
    import sglang.srt.plugins as plugins
    from sglang.srt.plugins.hook_registry import HookRegistry
    from sglang.srt.arg_groups import resolution_hooks

    errors = []
    for label, module, root in (
        ("official", sglang, official_python),
        ("plugin", sglang_dcu_plugin, plugin_root),
    ):
        source = Path(module.__file__).resolve()
        if not source.is_relative_to(Path(root).resolve()):
            errors.append(f"{label} imported from unexpected path: {source}")

    captured = PluginErrors()
    logger = logging.getLogger("sglang.srt.plugins")
    logger.addHandler(captured)
    try:
        HookRegistry.reset()
        plugins._plugins_loaded = False
        discovery = (patch.object(plugins, "entry_points", return_value=[])
                     if disable_entry_points else contextlib.nullcontext())
        with discovery:
            plugins.load_plugins()
    finally:
        logger.removeHandler(captured)

    targets = sorted(HookRegistry._hooks)
    unapplied = sorted(set(targets) - HookRegistry._patched)
    errors.extend(captured.messages)
    if not targets:
        errors.append("No hooks registered: an empty loader run is not success")
    if unapplied:
        errors.append(f"{len(unapplied)} registered targets were not applied")
    if expected_targets is not None:
        expected = set(expected_targets)
        if set(targets) != expected:
            errors.append("Target contract differs: missing=" + str(sorted(expected-set(targets)))
                          + " unexpected=" + str(sorted(set(targets)-expected)))

    named = sorted(name for name, wrappers in resolution_hooks._HOOKS.items()
                   if any(w.__module__.startswith("sglang_dcu_plugin") for w in wrappers))
    required = {"handle_model_specific_adjustments", "handle_a2a_moe",
                "handle_attention_backend_compatibility"}
    if not required <= set(named):
        errors.append("Missing named resolution seams: " + str(sorted(required-set(named))))
    return {
        "status": "failed" if errors else "passed",
        "official": sglang.__file__, "plugin": sglang_dcu_plugin.__file__,
        "external_entry_points_disabled": disable_entry_points,
        "registered": len(targets), "applied": len(set(targets)&HookRegistry._patched),
        "unapplied": unapplied, "named_resolution_hooks": named,
        "targets": targets, "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-python", type=Path, required=True)
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable-entry-points", action="store_true")
    parser.add_argument("--expected-targets", type=Path)
    args = parser.parse_args()
    expected = (json.loads(args.expected_targets.read_text())
                if args.expected_targets else None)
    report = validate(args.official_python, args.plugin_root,
                      disable_entry_points=args.disable_entry_points,
                      expected_targets=expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "targets"}, indent=2))
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
