# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations
from typing import Optional, Iterable, Mapping, List
from types import MappingProxyType
from torch.nn import Module
from sglang.srt.layers.quantization.compressed_tensors import utils as N
from hcu_sglang.utils.detection import is_hcu

def find_matched_target(
    layer_name: Optional[str],
    module: Module,
    targets: Iterable[str],
    fused_mapping: Mapping[str, List[str]] = MappingProxyType({}),
) -> str:
    """
    Helper function to look up which "target" in the compressed-tensors
    config that a layer corresponds to.

    Recall that a compressed-tensors configs has a concept of
    config_groups, where each layer can be quantized with with a different
    scheme.

    targets in each config_group will be a list of either layer names
    (or regexes corresponding to layer names) or names of torch Modules.

    First, we try to match the layer_name with a target
    Second, we try to match the module's name with a target
    Third, we try to map the layer_name to a list of fused module names.
        *All* component module names must match in order for a match to be
        successful. A successful match returns the first component target

    :param layer_name: layer name
    :param module: torch.nn.Module
    :param targets: list of targets to match the layer against
    :param fused_mapping: map from fused layer names to its components
    :param fused_strategy: either "all" or "any". If using "all", fused
        layers match if "all" of its components match
    """

    if layer_name is None:
        layer_name = ""

    matched_target = N._find_first_match(layer_name, targets)

    if matched_target is None:
        for cls in type(module).mro():
            if cls is object:
                continue
            matched_target = N._find_first_match(cls.__name__, targets, True)
            if matched_target is not None:
                break

    if matched_target is None:
        matched_target = N._match_fused_layer(layer_name, targets, fused_mapping)

    if matched_target is None:
        raise ValueError(
            f"Unable to find matching target for {layer_name} in the "
            "compressed-tensors config."
        )

    return matched_target


def around_find_target(original, layer_name, module, targets, fused_mapping=MappingProxyType({})):
    if not is_hcu():
        return original(layer_name, module, targets, fused_mapping)
    return find_matched_target(layer_name, module, targets, fused_mapping)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register(
        "sglang.srt.layers.quantization.compressed_tensors.utils.find_matched_target",
        around_find_target, HookType.AROUND,
    )
