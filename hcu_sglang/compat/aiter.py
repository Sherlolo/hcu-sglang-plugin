"""Compatibility shims for optional APIs absent from older HCU AITER wheels."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import types


_FUSED_MODULE = (
    "aiter.ops.triton.gemm.fused.fused_gemm_afp4wfp4_split_cat"
)


def _unsupported_fused_gemm(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError(
        "The installed AITER build does not provide "
        "fused_gemm_afp4wfp4_split_cat. This kernel is required only by the "
        "Quark W4A4 MXFP4 path; install a newer compatible AITER build to use it."
    )


def _package(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    return module


def install_aiter_compat() -> bool:
    try:
        importlib.import_module(_FUSED_MODULE)
        return False
    except ModuleNotFoundError as exc:
        if exc.name not in {
            "aiter.ops.triton.gemm",
            "aiter.ops.triton.gemm.fused",
            _FUSED_MODULE,
        }:
            raise

    triton = importlib.import_module("aiter.ops.triton")
    gemm = _package("aiter.ops.triton.gemm")
    fused = _package("aiter.ops.triton.gemm.fused")
    leaf = types.ModuleType(_FUSED_MODULE)
    leaf.fused_gemm_afp4wfp4_split_cat = _unsupported_fused_gemm
    sys.modules[_FUSED_MODULE] = leaf

    triton.gemm = gemm
    gemm.fused = fused
    fused.fused_gemm_afp4wfp4_split_cat = leaf
    return True


class _AiterCompatibilityLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        if module.__name__ == _FUSED_MODULE:
            module.fused_gemm_afp4wfp4_split_cat = _unsupported_fused_gemm


class _AiterCompatibilityFinder(importlib.abc.MetaPathFinder):
    """Supply only absent optional modules, after ordinary import finders."""

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if fullname not in {
            "aiter.ops.triton.gemm",
            "aiter.ops.triton.gemm.fused",
            _FUSED_MODULE,
        }:
            return None
        return importlib.machinery.ModuleSpec(
            fullname,
            _AiterCompatibilityLoader(),
            is_package=fullname != _FUSED_MODULE,
        )


def bootstrap_aiter_compat() -> bool:
    """Install a lazy shim without importing vendor code during Python startup.

    AITER invokes Python-based build tools such as ``ninja --version``. Eagerly
    importing AITER from usercustomize in that child waits on the parent's JIT
    lock and deadlocks the build. The final finder runs only if real finders
    failed, preserving an installed vendor implementation.
    """
    if any(isinstance(finder, _AiterCompatibilityFinder) for finder in sys.meta_path):
        return False
    sys.meta_path.append(_AiterCompatibilityFinder())
    return True
