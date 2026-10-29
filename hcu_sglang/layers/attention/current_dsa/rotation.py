from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from scipy.linalg import hadamard
from sglang.srt.layers.attention.dsa.dsa_indexer import _is_hip, _is_xpu
from sglang.srt.utils import get_bool_env_var
from hcu_sglang.utils.detection import is_dcu
_is_hcu = is_dcu()
_use_fast_hadamard_transform = get_bool_env_var("SGLANG_USE_FAST_HADAMARD_TRANSFORM")
_HADAMARD_CACHE = {}
if _is_hcu and _use_fast_hadamard_transform:
    from fast_hadamard_transform import hadamard_transform as hcu_fast_hadamard_transform


def _get_hadamard_matrix(
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    key = (dim, str(device), str(dtype))
    if key not in _HADAMARD_CACHE:
        dim_padded = 2 ** math.ceil(math.log2(dim))
        matrix = hadamard(dim_padded, dtype=float)
        _HADAMARD_CACHE[key] = torch.tensor(
            matrix,
            dtype=dtype,
            device=device,
        )
    return _HADAMARD_CACHE[key]

def hadamard_transform_optimized(
    x: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """HCU fallback using a cached dense Hadamard transform."""
    x_shape = x.shape
    dim = x.shape[-1]
    dim_padded = 2 ** math.ceil(math.log2(dim))
    x_2d = x.reshape(-1, dim)
    if dim != dim_padded:
        x_2d = F.pad(x_2d, (0, dim_padded - dim))
    out = F.linear(x_2d, _get_hadamard_matrix(dim, x.device, x.dtype))
    if scale != 1.0:
        out = out * scale
    return out[..., :dim].reshape(*x_shape)

def rotate_activation(x: torch.Tensor, apply_scale: bool = True) -> torch.Tensor:
    if _is_hip and not _is_hcu:
        from fast_hadamard_transform import hadamard_transform
    elif _is_xpu:
        from sgl_kernel import hadamard_transform
    elif not _is_hcu:
        from sglang.kernels.ops.quantization.hadamard import hadamard_transform

    hidden_size = x.size(-1)
    assert (hidden_size & (hidden_size - 1)) == 0, (
        "Hidden size must be a power of 2 for Hadamard transform."
    )
    scale = hidden_size**-0.5 if apply_scale else 1.0
    if _is_hcu and _use_fast_hadamard_transform:
        return hcu_fast_hadamard_transform(x, scale=scale)
    if _is_hcu:
        return hadamard_transform_optimized(x, scale=scale)
    return hadamard_transform(x, scale=scale)
