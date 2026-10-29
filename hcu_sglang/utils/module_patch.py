"""Small helpers for calling original SGLang functions from replacement hooks."""

import importlib


def resolve_attr(module_name, attr_path):
    module = importlib.import_module(module_name)
    obj = module
    for name in attr_path.split("."):
        obj = getattr(obj, name)
    return obj


def get_original_attr(module_name, attr_path):
    obj = resolve_attr(module_name, attr_path)
    return getattr(obj, "__wrapped__", obj)
