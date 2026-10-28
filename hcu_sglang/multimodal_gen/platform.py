"""HCU platform discovery for multimodal generation."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from typing import Callable, Optional


_DEVICE_COUNT_PREFIX = "HCUSMI_DEVICE_COUNT="


def probe_hcusmi_device_count() -> int:
    probe = f'''\
import hcusmi
hcusmi.hcusmi_init()
try:
    print("{_DEVICE_COUNT_PREFIX}" + str(len(hcusmi.hcusmi_get_processor_handles())))
finally:
    hcusmi.hcusmi_shut_down()
'''
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        check=True,
        text=True,
        timeout=10,
    )
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_DEVICE_COUNT_PREFIX):
            return int(line.removeprefix(_DEVICE_COUNT_PREFIX))
    raise RuntimeError("hcusmi probe did not report a device count")


def around_rocm_platform_plugin(
    original: Callable[[], Optional[str]],
) -> Optional[str]:
    try:
        if (
            importlib.util.find_spec("hcusmi") is not None
            and probe_hcusmi_device_count() > 0
        ):
            return "sglang.multimodal_gen.runtime.platforms.rocm.RocmPlatform"
    except Exception:
        pass
    return original()


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.multimodal_gen.runtime.platforms.rocm_platform_plugin",
        around_rocm_platform_plugin,
        HookType.AROUND,
    )
