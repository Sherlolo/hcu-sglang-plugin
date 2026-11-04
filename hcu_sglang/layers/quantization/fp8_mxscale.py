# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
from functools import lru_cache
from typing import List, Optional, Tuple
import torch
from torch.nn import Module
from sglang.srt.layers.quantization import fp8_utils as N, fp8 as P
from hcu_sglang.utils.detection import is_hcu

if N._use_aiter:
    from aiter.ops.quant import per_group_quant_hip

_DENSE_MXSCALE = False
_DENSE_MXSCALE_LOGGED = set()

DSV4_DENSE_MXSCALE_DEFAULT_SHAPES = frozenset(
    {
        (2048, 7168),  # wqkv_a
        (1536, 7168),  # wq_a (unfused: DSpark draft)
        (8192, 1536),  # wq_b TP8, indexer wq_b
        (16384, 1536),  # wq_b TP4
        (65536, 1536),  # wq_b attn-TP1 (DP attention)
        (7168, 2048),  # wo_b TP8
        (7168, 4096),  # wo_b TP4
        (7168, 16384),  # wo_b attn-TP1 (DP attention)
        (6144, 7168),  # shared expert w13, TP1
        (7168, 3072),  # shared expert w2, TP1
        (768, 7168),  # shared expert w13, TP8
        (7168, 384),  # shared expert w2, TP8
        (7168, 21504),  # DSpark main_proj
    }
)

@lru_cache(maxsize=1)
def aiter_has_dense_mxscale() -> bool:
    """Whether the installed aiter dispatches e8m0 block scales in
    gemm_a8w8_blockscale_bpreshuffle. Older aiter takes the e8m0 bytes as fp32
    scales and silently returns wrong results."""
    try:
        from aiter.ops.gemm_op_a8w8 import get_mxscale_bpreshuffle_config  # noqa: F401
    except Exception:
        return False
    return True

def set_dense_mxscale(enabled: bool = True) -> None:
    global _DENSE_MXSCALE
    if enabled and not aiter_has_dense_mxscale():
        N.logger.warning(
            "SGLANG_DSV4_DENSE_MXSCALE is set but the installed aiter has no e8m0 "
            "mxscale bpreshuffle GEMM; keeping the fp32-scale GEMM."
        )
        enabled = False
    _DENSE_MXSCALE = enabled

@lru_cache(maxsize=1)
def _dense_mxscale_shapes() -> Optional[frozenset]:
    """The (N, K) whitelist, or None for every shape."""
    spec = N.envs.SGLANG_DSV4_DENSE_MXSCALE_SHAPES.get().strip()
    if not spec:
        return DSV4_DENSE_MXSCALE_DEFAULT_SHAPES
    if spec.lower() == "all":
        return None
    shapes = set()
    for item in spec.split(","):
        n, k = item.strip().split(":")
        shapes.add((int(n), int(k)))
    return frozenset(shapes)

def dense_mxscale_wanted(n: int, k: int) -> bool:
    if not (_DENSE_MXSCALE and N._use_aiter_bpreshuffle_gfx95):
        return False
    shapes = _dense_mxscale_shapes()
    return shapes is None or (n, k) in shapes

def block_fp8_scale_to_e8m0_128(scale: torch.Tensor) -> Optional[torch.Tensor]:
    """fp32 128x128 block scales [N/128, K/128] -> e8m0 bytes of the same shape,
    or None if any scale is not a positive normal power of two (not lossless)."""
    s = scale.detach().float().contiguous()
    bits = s.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    ok = (s > 0) & ((bits & 0x7FFFFF) == 0) & (exp > 0) & (exp < 0xFF)
    if not bool(ok.all()):
        return None
    return exp.to(torch.uint8).view(torch.float8_e8m0fnu).contiguous()

def is_e8m0_scale(scale: Optional[torch.Tensor]) -> bool:
    return scale is not None and scale.dtype == torch.float8_e8m0fnu

def dense_mxscale_ready(*linears) -> bool:
    """Whether every given linear consumes e8m0 (mxscale) activations, i.e. a
    shared producer of their fp8 operand should emit ue8m0 scales."""
    if not _DENSE_MXSCALE:
        return False
    for linear in linears:
        if linear is None:
            continue
        weight = getattr(linear, "weight", None)
        if getattr(weight, "_mxscale_w_scale", None) is None:
            return False
    return True

def quant_fp8_e8m0_128(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-token 1x128 fp8 quant with ue8m0 scales, the scale bytes column-major
    ([K/128, M] storage viewed as [M, K/128]) as the mxscale GEMM reads them."""
    return per_group_quant_hip(
        x,
        quant_dtype=N.aiter.dtypes.fp8,
        group_size=128,
        transpose_scale=True,
        scale_type=torch.float8_e8m0fnu,
    )

def aiter_w8a8_block_fp8_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    block_size: List[int],
    weight_scale: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # assert input_scale is None
    input_2d = input.view(-1, input.shape[-1])
    output_shape = [*input.shape[:-1], weight.shape[0]]
    output_dtype = torch.bfloat16 if input_scale is not None else input.dtype
    # DP-attention idle ranks run zero-token forwards (eager has no padded graph
    # bucket); CK blockscale GEMM fails M == 0 with invalid configuration argument.
    if input_2d.shape[0] == 0:
        return input.new_empty(output_shape, dtype=output_dtype)

    # dense linears converted from MXFP8 carry a rowwise-fp8 copy; its ptpc GEMM
    # beats the block-fp8 GEMMs at decode-sized M
    if input_scale is None:
        ptpc_weight = getattr(weight, "_ptpc_weight", None)
        if (
            ptpc_weight is not None
            and input_2d.shape[0] <= N.MXFP8_DENSE_PTPC_DECODE_MAX_M
        ):
            out = N.apply_fp8_ptpc_linear(
                input=input_2d,
                weight=ptpc_weight,
                weight_scale=weight._ptpc_scale,
                bias=bias,
            )
            return out.to(input.dtype).view(*output_shape)

    # DeepSeek-V4 e8m0 mxscale route (SGLANG_DSV4_DENSE_MXSCALE): same
    # (16, 16)-preshuffled weight, e8m0 copy of its scale. A producer that
    # already emitted an fp32-scale operand keeps the fp32 GEMM below.
    mx_w_scale = getattr(weight, "_mxscale_w_scale", None)
    if is_e8m0_scale(input_scale) or (mx_w_scale is not None and input_scale is None):
        if mx_w_scale is None:
            raise RuntimeError(
                "e8m0-scaled fp8 activation reached a linear without an e8m0 "
                f"weight scale (weight {tuple(weight.shape)})"
            )
        out_dtype = torch.bfloat16 if input_scale is not None else input_2d.dtype
        if input_2d.shape[0] == 0:
            return input_2d.new_empty(output_shape, dtype=out_dtype)
        if input_scale is None:
            q_input, x_scale = quant_fp8_e8m0_128(input_2d)
        else:
            q_input, x_scale = input_2d, input_scale
        output = N.gemm_a8w8_blockscale_bpreshuffle(
            q_input, weight, x_scale, mx_w_scale, dtype=torch.bfloat16
        )
        if bias is not None:
            output += bias
        return output.to(dtype=out_dtype).view(*output_shape)

    n, k = weight.shape

    if N._use_aiter_bpreshuffle_gfx95:
        use_triton = N.use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k)
    elif N._use_aiter_gfx95:
        # gfx95 on ROCm < 7.2: keep the (faster) CK path at/below the per-shape
        # CK-safe M bound; above it, ck_gemm_a8w8_blockscale returns NaN, so use
        # Triton. Unlisted shapes keep their original decision. Fixed in ROCm 7.2.
        _ck_safe_m = N._AITER_GFX95_CK_W8A8_MAX_SAFE_M.get((n, k))
        use_triton = N.use_aiter_triton_gemm_w8a8_tuned_gfx950(n, k) or (
            _ck_safe_m is not None and input_2d.shape[0] > _ck_safe_m
        )
    elif N._use_aiter_gfx942:
        use_triton = not N.envs.SGLANG_AITER_GFX942_BLOCKSCALE_USE_CK.get()
    else:
        use_triton = True

    # if input_scale not None, input is quanted
    if input_scale is not None:
        q_input = input_2d
        x_scale = input_scale
        if N._use_aiter_bpreshuffle_gfx95 and not use_triton:
            x_scale = N.materialize_bpreshuffle_fp8_scale(x_scale)
        # On ROCm >= 7.2, scale is in bpreshuffle's transposed layout.
        # Triton needs a row-major view, so adjust strides only. No copy.
        elif use_triton and N._use_aiter_bpreshuffle_gfx95:
            x_scale = N.view_aiter_fused_rms_transposed_fp8_scale(x_scale)
    else:
        materialize_bpreshuffle_scale = N._use_aiter_bpreshuffle_gfx95 and not use_triton
        # No-copy bpreshuffle scale: emit it already transposed and stride-reinterpret
        # to the column-major bpreshuffle layout, instead of a .t().contiguous().t()
        # copy. Bit-identical for M>=2; M==1 keeps materialize (there the [1,G] and
        # [G,1] byte orders coincide, so materialize is a no-op view anyway).
        emit_bpreshuffle_scale = (
            materialize_bpreshuffle_scale and input_2d.shape[0] >= 2
        )
        q_input, x_scale = N.aiter_per1x128_quant(
            input_2d,
            quant_dtype=N.aiter.dtypes.fp8,
            transpose_scale=emit_bpreshuffle_scale,
        )
        if emit_bpreshuffle_scale:
            x_scale = N.view_aiter_fused_rms_transposed_fp8_scale(x_scale)
        elif materialize_bpreshuffle_scale:
            x_scale = N.materialize_bpreshuffle_fp8_scale(x_scale)

    if use_triton:
        gemm_a8w8_blockscale_op = N.triton_gemm_a8w8_blockscale
    elif N._use_aiter_bpreshuffle_gfx95:
        gemm_a8w8_blockscale_op = N.gemm_a8w8_blockscale_bpreshuffle
    else:
        gemm_a8w8_blockscale_op = N.ck_gemm_a8w8_blockscale

    output = gemm_a8w8_blockscale_op(
        q_input,
        weight,
        x_scale,
        weight_scale,
        dtype=output_dtype,
    )

    if bias is not None:
        output += bias

    return output.to(dtype=output_dtype).view(*output_shape)

def _maybe_prepare_dense_mxscale(self, layer: Module) -> None:
    """SGLANG_DSV4_DENSE_MXSCALE: give a (16, 16)-preshuffled 128x128 block-fp8
        weight an e8m0 copy of its scale, which routes aiter_w8a8_block_fp8_linear
        onto aiter's mxscale bpreshuffle GEMM. Lossless only for power-of-two
        scales (ue8m0 checkpoints); any other layer stays on the fp32 GEMM. The
        weight bytes are shared, the copy costs N*K/16384 bytes."""
    if self.weight_block_size != [128, 128] or self.convert_mxfp8_to_block:
        return
    if layer.weight.dtype != torch.float8_e4m3fn:
        return
    n, k = layer.weight.shape
    if not dense_mxscale_wanted(n, k):
        return
    scale = layer.weight_scale_inv.data
    if tuple(scale.shape) != (n // 128, k // 128) or n % 128 or k % 128:
        return
    e8m0 = block_fp8_scale_to_e8m0_128(scale)
    if e8m0 is None:
        P.logger.warning(
            "Dense mxscale: %s keeps the fp32 GEMM (scales are not powers of two)",
            getattr(layer, "prefix", tuple(layer.weight.shape)),
        )
        return
    layer.weight._mxscale_w_scale = e8m0
    if (n, k) not in _DENSE_MXSCALE_LOGGED:
        _DENSE_MXSCALE_LOGGED.add((n, k))
        P.logger.info(
            "Dense mxscale (e8m0 bpreshuffle GEMM) enabled for N=%d K=%d", n, k
        )

def around_aiter_block_linear(original, input, weight, block_size, weight_scale, input_scale=None, bias=None):
    if not is_hcu():
        return original(input, weight, block_size, weight_scale, input_scale, bias)
    return aiter_w8a8_block_fp8_linear(input, weight, block_size, weight_scale, input_scale, bias)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    if not is_hcu():
        return
    for name in (
        "aiter_has_dense_mxscale", "set_dense_mxscale", "_dense_mxscale_shapes",
        "dense_mxscale_wanted", "block_fp8_scale_to_e8m0_128", "is_e8m0_scale",
        "dense_mxscale_ready", "quant_fp8_e8m0_128",
    ):
        value = globals()[name]
        previous = getattr(N, name, value)
        if previous is not value:
            raise RuntimeError(f"Native FP8 symbol {name} already exists; re-review the source contract")
        setattr(N, name, value)
    previous = getattr(P.Fp8LinearMethod, "_maybe_prepare_dense_mxscale", _maybe_prepare_dense_mxscale)
    if previous is not _maybe_prepare_dense_mxscale:
        raise RuntimeError("Native FP8 dense mxscale preparation already exists; re-review the source contract")
    P.Fp8LinearMethod._maybe_prepare_dense_mxscale = _maybe_prepare_dense_mxscale
    HookRegistry.register(
        "sglang.srt.layers.quantization.fp8_utils.aiter_w8a8_block_fp8_linear",
        around_aiter_block_linear, HookType.AROUND,
    )
