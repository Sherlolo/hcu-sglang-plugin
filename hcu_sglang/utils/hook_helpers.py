"""Shared helpers for REPLACE / AROUND hooks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from hcu_sglang.utils.detection import is_hcu

F = TypeVar("F", bound=Callable[..., Any])


def hcu_only(original: F, hcu_impl: F) -> F:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if is_hcu():
            return hcu_impl(*args, **kwargs)
        return original(*args, **kwargs)

    wrapper.__name__ = getattr(hcu_impl, "__name__", "hcu_only_wrapper")
    return wrapper  # type: ignore[return-value]
