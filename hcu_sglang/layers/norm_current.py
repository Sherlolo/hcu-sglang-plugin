# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Optional, Tuple, Union

import torch
from torch import nn
from lightop import op
from sglang.srt.layers import layernorm as N
from hcu_sglang.utils.detection import is_hcu

_is_hcu = is_hcu()
_use_hcu_lightop_gemma_rmsnorm = _is_hcu and N.get_bool_env_var(
    "SGLANG_USE_LIGHTOP_GEMMA_RMSNORM"
)


def gemma_fused_add_rmsnorm_hcu(*args, **kwargs):
    from lightop import gemma_fused_add_rmsnorm
    return gemma_fused_add_rmsnorm(*args, **kwargs)


def gemma_rms_norm_fp8_quant_hcu(*args, **kwargs):
    from lightop import gemma_rms_norm_per_token_fp8_quant
    return gemma_rms_norm_per_token_fp8_quant(*args, **kwargs)


def _call_vllm_fused_add_rms_norm(
    op,
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Call either the current in-place or legacy out-parameter vLLM ABI."""
    arity = _fused_add_rms_norm_arity(op)
    if arity == 4:
        op(x, residual, weight, eps)
        return x, residual

    if arity == 6:
        out = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        op(out, x, residual_out, residual, weight, eps)
        return out, residual_out

    # Some extension wrappers do not expose an inspectable signature. A wrong
    # argument count fails before kernel launch, so it is safe to probe the
    # current ABI and fall back only on TypeError.
    try:
        op(x, residual, weight, eps)
    except TypeError as current_abi_error:
        out = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        try:
            op(out, x, residual_out, residual, weight, eps)
        except TypeError:
            raise current_abi_error
        return out, residual_out
    return x, residual


@lru_cache(maxsize=None)
def _fused_add_rms_norm_arity(op) -> Optional[int]:
    """Return the Python wrapper arity for known vLLM RMSNorm ABIs."""
    try:
        parameters = inspect.signature(op).parameters.values()
    except (TypeError, ValueError):
        return None
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    if any(parameter.kind == parameter.VAR_POSITIONAL for parameter in parameters):
        return None
    return len(positional) if len(positional) in (4, 6) else None


def rms_forward_hip(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
    quant_linear: Optional[nn.Module] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # HCU uses LightOp directly; only generic ROCm depends on vLLM ops.
    if not _is_hcu and not N._has_vllm_rms_norm:
        return self.forward_native(x, residual, post_residual_addition)

    if N.is_batch_invariant_mode_enabled():
        if (
            residual is not None
            or self.cast_x_before_out_mul
            or N.get_exec().deterministic.rl_on_policy_target == "fsdp"
        ):
            return self.forward_native(x, residual, post_residual_addition)
        return N.rms_norm_batch_invariant(
            x,
            self.weight.data,
            self.variance_epsilon,
        )

    if not x.is_contiguous():
        x = x.contiguous()

    if residual is not None:
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        if _is_hcu:
            op.fused_add_rms_norm_opt(
                x,
                residual,
                self.weight.data,
                self.variance_epsilon,
            )
            return x, residual
        return _call_vllm_fused_add_rms_norm(
            N.fused_add_rms_norm,
            x,
            residual,
            self.weight.data,
            self.variance_epsilon,
        )
    if _is_hcu:
        out = torch.empty_like(x)
        op.rms_norm_opt(out, x, self.weight.data, self.variance_epsilon)
        return out
    out = torch.empty_like(x)
    N.rms_norm(out, x, self.weight.data, self.variance_epsilon)
    return out


def gemma_forward_hip(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    if N._use_aiter and N._has_rocm_triton_gemma_rms_norm:
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            return N.rocm_triton_gemma_fused_add_rmsnorm(
                x, residual, self.weight.data, self.variance_epsilon
            )
        return N.rocm_triton_gemma_rmsnorm(x, self.weight.data, self.variance_epsilon)

    if _is_hcu:
        if not x.is_contiguous():
            x = x.contiguous()
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            return gemma_fused_add_rmsnorm_hcu(
                x, residual, self.weight.data, self.variance_epsilon
            )
        out = torch.empty_like(x)
        op.rms_norm_opt(out, x, self.gemma_weight, self.variance_epsilon)
        return out

    if not N._has_vllm_rms_norm:
        return self.forward_native(x, residual, post_residual_addition)

    if N._use_aiter:
        # AITER's ROCm rmsnorm2d_fwd has the same dtype requirement here;
        # keep Gemma RMSNorm on native torch math for correctness.
        return self.forward_native(x, residual, post_residual_addition)
    else:
        w = self.gemma_weight
        # vllm API: rms_norm(out, input, weight, eps) -> None (in-place).
        # fused_add_rms_norm has both current four-argument and legacy
        # six-argument wheel ABIs; the helper handles either contract.
        if not x.is_contiguous():
            x = x.contiguous()
        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            return _call_vllm_fused_add_rms_norm(
                N.fused_add_rms_norm,
                x,
                residual,
                w,
                self.variance_epsilon,
            )
        out = torch.empty_like(x)
        N.rms_norm(out, x, w, self.variance_epsilon)
        return out


def gemma_forward_with_lightop_fp8_quant(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
):
    """Gemma RMSNorm + dynamic per-token OCP-FP8 quant for LightOp GEMM.

    LightOp updates ``residual`` in place with ``x + residual`` when a
    residual is supplied.  ``update_input=False`` avoids writing a BF16
    normalized copy back to ``x`` because the following channel-FP8 GEMM
    consumes only the returned ``(fp8, scale)`` tuple.
    """
    if not _use_hcu_lightop_gemma_rmsnorm:
        raise RuntimeError(
            "LightOp Gemma RMSNorm+FP8 quant requires "
            "SGLANG_USE_LIGHTOP_GEMMA_RMSNORM=1 on HCU gfx93x"
        )
    if post_residual_addition is not None:
        if residual is None:
            residual = post_residual_addition
        else:
            residual = residual + post_residual_addition

    quantized = gemma_rms_norm_fp8_quant_hcu(
        x,
        self.weight.data,
        self.variance_epsilon,
        fp8type=0,
        residual=residual,
        update_input=False,
    )
    return quantized if residual is None else (quantized, residual)


def around_gemma_forward_hip(original, self, x, residual=None, post_residual_addition=None):
    if not is_hcu():
        return original(self, x, residual, post_residual_addition)
    return gemma_forward_hip(self, x, residual, post_residual_addition)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    N.GemmaRMSNorm.forward_with_lightop_fp8_quant = gemma_forward_with_lightop_fp8_quant
    HookRegistry.register(
        "sglang.srt.layers.layernorm.GemmaRMSNorm.forward_hip",
        around_gemma_forward_hip,
        HookType.AROUND,
    )
