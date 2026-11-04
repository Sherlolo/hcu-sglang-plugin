# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path


PACKAGES = ('hcu_sglang', 'sglang_hcu_plugin')


def resolve_plugin_package(root: Path, package: str | None = None) -> str:
    if package is not None and package not in PACKAGES:
        raise ValueError(f'unsupported plugin package: {package}')
    candidates = [name for name in PACKAGES if (root / name / '__init__.py').is_file()]
    if package is not None:
        if package not in candidates:
            raise ValueError(f'missing plugin package: {root / package}')
        return package
    if len(candidates) != 1:
        raise ValueError(f'expected one plugin package under {root}, found {candidates}; select --plugin-package')
    return candidates[0]
