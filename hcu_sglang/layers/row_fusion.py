from __future__ import annotations
import sglang.srt.layers.linear as N
from hcu_sglang.utils.detection import is_dcu

_use_fused_silu_mul_quant = N.get_bool_env_var("SGLANG_USE_FUSED_SILU_MUL_QUANT")
_use_fused_silu_mul_fp8_quant = N.envs.SGLANG_USE_FUSED_SILU_MUL_FP8_QUANT.get() or _use_fused_silu_mul_quant
_use_fused_bailing_silu_mul_fp8_quant = N.get_bool_env_var("SGLANG_USE_FUSED_BAILING_SILU_MUL_FP8_QUANT")
_use_fused_dpskv4_silu_mul_fp8_quant = N.get_bool_env_var("SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT")
if _use_fused_silu_mul_quant:
    try:
        from lightop.activation import lm_fuse_silu_mul_quant
    except Exception as e:
        print(f"Error: Import fused silu_mul_quant error: {e}")

_lightop_fuse_silu_mul_clamp_quant = None
if _use_fused_silu_mul_quant:
    try:
        from lightop import fuse_silu_mul_clamp_quant

        _lightop_fuse_silu_mul_clamp_quant = fuse_silu_mul_clamp_quant
    except ImportError:
        pass

_lightop_fuse_silu_mul_fp8_quant = None
if (
    _use_fused_silu_mul_fp8_quant
    or _use_fused_bailing_silu_mul_fp8_quant
    or _use_fused_dpskv4_silu_mul_fp8_quant
):
    try:
        from lightop.activation import (
            fuse_silu_mul_fp8_quant as _lightop_fuse_silu_mul_fp8_quant,
        )
    except ImportError:
        try:
            # Current HCU wheels export this operator from the package root.
            from lightop import (
                fuse_silu_mul_fp8_quant as _lightop_fuse_silu_mul_fp8_quant,
            )
        except ImportError:
            pass


def supports_fused_silu_mul_quant_input(self) -> bool:
    return bool(
        _use_fused_silu_mul_quant
        and _lightop_fuse_silu_mul_clamp_quant is not None
        and getattr(
            self.quant_method,
            "supports_prequantized_input",
            False,
        )
    )

def supports_fused_silu_mul_fp8_quant_input(self) -> bool:
    return bool(
        _use_fused_silu_mul_fp8_quant
        and _lightop_fuse_silu_mul_fp8_quant is not None
        and getattr(
            getattr(self, "scheme", None),
            "supports_fp8_prequantized_input",
            False,
        )
    )

def forward(
    self,
    input_,
    skip_all_reduce=False,
    forward_batch=None,
    use_fused_silu_mul_quant: N.Optional[bool] = False,
    use_fused_silu_mul_fp8_quant: N.Optional[bool] = False,
    swiglu_limit: N.Optional[float] = None,
    output_tensor=None,
    input_quant_args=None,
):
    tp_rank, tp_size = N.get_group_rank_size(self.tp_group)
    if self.input_is_parallel:
        input_parallel = input_
    else:
        splitted_input = N.split_tensor_along_last_dim(input_, num_partitions=tp_size)
        input_parallel = splitted_input[tp_rank].contiguous()

    # Matrix multiply.
    assert self.quant_method is not None
    # Only fuse bias add into GEMM for rank 0 (this ensures that
    # bias will not get added more than once in TP>1 case)
    bias_ = None if (tp_rank > 0 or self.skip_bias_add) else self.bias

    if input_quant_args is not None and N.get_forward().sp_active and tp_size > 1:
        raise ValueError(
            "Prequantized row input is unsupported by sequence-parallel reduction"
        )

    # Megatron SP "g-bar": reduce-scatter along the token dim instead of
    # all-reduce, leaving the output sharded for the next SP LayerNorm region.
    # Fires regardless of reduce_results: o_proj / down are built
    # reduce_results=False, so under SP the linear owns the reduction.
    if (
        N.get_forward().sp_active
        and tp_size > 1
        and not skip_all_reduce
        and output_tensor is None
    ):
        output = N.layernorm_sp.row_parallel_gbar_matmul(self, input_parallel, bias_)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias

    if self.use_dp_attention_reduce:
        symm_ctx = N.use_symmetric_memory(N.get_parallel().attn_tp_group)
    else:
        symm_ctx = N.use_symmetric_memory(
            N.get_parallel().tp_group, disabled=not N.is_allocation_symmetric()
        )

    if input_quant_args is not None:
        if not self.input_is_parallel:
            raise ValueError(
                "Prequantized row-parallel input must already be local"
            )
        if (
            use_fused_silu_mul_quant
            or use_fused_silu_mul_fp8_quant
            or output_tensor is not None
        ):
            raise ValueError(
                "Prequantized input conflicts with other fused/output modes"
            )
        if not getattr(self.quant_method, "supports_prequantized_input", False):
            raise TypeError(
                "Quantization method does not support prequantized input"
            )
        with symm_ctx as sm:
            output_parallel = self.quant_method.apply(
                self, input_parallel, bias=bias_, input_quant_args=input_quant_args
            )
            if sm is not None:
                sm.tag(output_parallel)
    elif use_fused_silu_mul_quant:
        if swiglu_limit is None:
            xq, xs = lm_fuse_silu_mul_quant(input_parallel)
            quant_kwargs = {"silu_quant_args": [xq, xs]}
            gemm_input = input_parallel
        else:
            if not self.supports_fused_silu_mul_quant_input():
                raise RuntimeError("LightOp clamp quantization is unavailable")
            xq, xs = _lightop_fuse_silu_mul_clamp_quant(
                input_parallel, swiglu_limit
            )
            gemm_input = input_parallel[..., : input_parallel.shape[-1] // 2]
            quant_kwargs = {"input_quant_args": (xq, xs)}
        with symm_ctx as sm:
            output_parallel = self.quant_method.apply(
                self,
                gemm_input,
                bias=bias_,
                **quant_kwargs,
            )
            if sm is not None:
                sm.tag(output_parallel)
    elif use_fused_silu_mul_fp8_quant:
        if not self.supports_fused_silu_mul_fp8_quant_input():
            raise RuntimeError("LightOp FP8 SwiGLU quantization is unavailable")
        if input_parallel.dim() != 2 or not input_parallel.is_contiguous():
            raise ValueError(
                "Fused FP8 SwiGLU quantization requires a contiguous 2D input"
            )

        num_tokens, gate_up_width = input_parallel.shape
        if gate_up_width % 2 != 0:
            raise ValueError("SwiGLU gate/up width must be even")
        input_x = N.torch.empty(
            (num_tokens, gate_up_width // 2),
            device=input_parallel.device,
            dtype=N.torch.float8_e4m3fn,
        )
        x_scale = N.torch.empty(
            (num_tokens, 1), device=input_parallel.device, dtype=N.torch.float32
        )
        _lightop_fuse_silu_mul_fp8_quant(
            input_parallel,
            fp8type=0,
            output=input_x,
            scales=x_scale,
            limit=swiglu_limit,
        )

        with symm_ctx as sm:
            output_parallel = self.quant_method.apply(
                self,
                (input_x, x_scale, input_parallel.dtype),
                bias=bias_,
            )
            if sm is not None:
                sm.tag(output_parallel)
    else:
        with symm_ctx:
            if output_tensor is None:
                output_parallel = self.quant_method.apply(
                    self, input_parallel, bias=bias_
                )
            else:
                apply_into = getattr(self.quant_method, "apply_into", None)
                if apply_into is None:
                    raise RuntimeError(
                        f"{type(self.quant_method).__name__} cannot write into "
                        "caller-owned linear output"
                    )
                output_parallel = apply_into(
                    self, input_parallel, output_tensor, bias=bias_
                )

    # skip_all_reduce: explicit call-site override. Also honor the
    # mlp_reduce_scatter ForwardFlag published by the decoder — callers
    # should not thread it into modules.
    if (
        ((self.reduce_results and tp_size > 1) or self.use_decode_attn_tp)
        and not skip_all_reduce
        and not N.should_skip_mlp_all_reduce()
    ):
        if self.use_dp_attention_reduce:
            output = N.get_parallel().attn_tp_group.all_reduce(output_parallel)
        else:
            quantize_communications = (
                (
                    not forward_batch.forward_mode.is_decode_or_idle()
                    and N.get_exec().comm.enable_quant_communications
                )
                if forward_batch is not None
                else False
            )
            if quantize_communications:
                output = N.tensor_model_parallel_quant_all_reduce(output_parallel)
            elif self.parallel_group == "tp":
                output = N.get_parallel().tp_group.all_reduce(output_parallel)
            else:
                output = N.tensor_model_parallel_all_reduce(output_parallel)
    else:
        output = output_parallel

    output_bias = self.bias if self.skip_bias_add else None

    return output, output_bias
def around_row_forward(original, layer, *args, **kwargs):
    if is_dcu() and any(name in kwargs for name in ("use_fused_silu_mul_quant", "use_fused_silu_mul_fp8_quant", "input_quant_args")):
        return forward(layer, *args, **kwargs)
    return original(layer, *args, **kwargs)


def register_hooks():
    if not is_dcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    N.RowParallelLinear.supports_fused_silu_mul_quant_input = supports_fused_silu_mul_quant_input
    N.RowParallelLinear.supports_fused_silu_mul_fp8_quant_input = supports_fused_silu_mul_fp8_quant_input
    HookRegistry.register("sglang.srt.layers.linear.RowParallelLinear.forward", around_row_forward, HookType.AROUND)
