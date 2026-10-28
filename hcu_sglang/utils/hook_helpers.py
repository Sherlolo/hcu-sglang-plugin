"""Shared helpers for REPLACE / AROUND hooks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from hcu_sglang.utils.detection import is_dcu

F = TypeVar("F", bound=Callable[..., Any])


def dcu_only(original: F, dcu_impl: F) -> F:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if is_dcu():
            return dcu_impl(*args, **kwargs)
        return original(*args, **kwargs)

    wrapper.__name__ = getattr(dcu_impl, "__name__", "dcu_only_wrapper")
    return wrapper  # type: ignore[return-value]
