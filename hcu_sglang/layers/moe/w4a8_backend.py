from __future__ import annotations
from contextvars import ContextVar
from contextlib import contextmanager
from sglang.srt.environ import envs

W4A8_TPMOE_BACKEND_AUTO = "auto"

W4A8_TPMOE_BACKEND_LIGHTOP = "lightop"

W4A8_TPMOE_BACKEND_AITER = "aiter"

W4A8_TPMOE_BACKEND_TRITON = "triton"

W4A8_TPMOE_BACKENDS = frozenset(
    {
        W4A8_TPMOE_BACKEND_AUTO,
        W4A8_TPMOE_BACKEND_LIGHTOP,
        W4A8_TPMOE_BACKEND_AITER,
        W4A8_TPMOE_BACKEND_TRITON,
    }
)

_dspark_w4a8_tpmoe_backend_override = ContextVar(
    "dspark_w4a8_tpmoe_backend_override", default=None
)

def normalize_w4a8_tpmoe_backend(requested_backend: str, *, env_name: str) -> str:
    backend = requested_backend.strip().lower()
    if backend not in W4A8_TPMOE_BACKENDS:
        supported = ", ".join(repr(value) for value in sorted(W4A8_TPMOE_BACKENDS))
        raise ValueError(
            f"Unsupported {env_name}={requested_backend!r}. "
            f"Supported values: {supported}."
        )
    return backend

def _resolve_dspark_w4a8_tpmoe_backend() -> str | None:
    requested_backend = envs.SGLANG_DSPARK_FORCE_W4A8_TPMOE_BACKEND.get()
    if requested_backend is None:
        return None
    return normalize_w4a8_tpmoe_backend(
        requested_backend,
        env_name="SGLANG_DSPARK_FORCE_W4A8_TPMOE_BACKEND",
    )

def get_dspark_w4a8_tpmoe_backend_override() -> str | None:
    return _dspark_w4a8_tpmoe_backend_override.get()

@contextmanager
def dspark_w4a8_tpmoe_backend_context():
    """Apply the DSpark W4A8 TP-MoE backend to one draft build."""
    token = _dspark_w4a8_tpmoe_backend_override.set(
        _resolve_dspark_w4a8_tpmoe_backend()
    )
    try:
        yield
    finally:
        _dspark_w4a8_tpmoe_backend_override.reset(token)
