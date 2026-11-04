# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(import_module("hcu_sglang.models.deepseek.current_v4.execution.boundary"), name)
