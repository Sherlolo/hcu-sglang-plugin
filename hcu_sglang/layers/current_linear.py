# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/layers/linear.py"""

from __future__ import annotations

from typing import Literal
from sglang.srt.layers import linear as N
from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.layers.fused_rms_quant import (
    fused_rms_norm_per_token_quant as lm_faster_rmsquant,
)

_is_hcu = is_hcu()
_use_fused_rms_quant = N.get_bool_env_var("SGLANG_USE_LEGACY_FUSED_RMS_QUANT")


def _resolve_linear_group(
    parallel_group: N.Optional[N.LinearParallelGroup],
) -> N.GroupCoordinator | N._LogicalGroup | N._ReplicatedGroup | None:
    if isinstance(parallel_group, N.ReplicatedParallelGroup):
        group = _resolve_linear_group(parallel_group.group)
        if group is None:
            if parallel_group.replica_size != 1:
                raise ValueError("Cannot replicate a size-one group")
            return None
        return N._ReplicatedGroup(group, parallel_group.replica_size)
    if parallel_group == "replicated":
        return None
    name = "tp" if parallel_group is None else parallel_group
    if name not in ("tp", "attn_tp", "shared_experts_tp") and not (
        _is_hcu and name == "attn_cp"
    ):
        raise ValueError(f"Unknown linear parallel_group: {parallel_group!r}")
    parallel = N.get_parallel()
    try:
        group = getattr(parallel, f"{name}_group")
    except RuntimeError:
        group = None
    if group is None:
        if name == "shared_experts_tp":
            raise ValueError(f"The {name} group must exist before construction")
        size = getattr(parallel, f"{name}_size")
        if size > 1:
            return N._LogicalGroup(getattr(parallel, f"{name}_rank"), size)
    return group


def replicated_forward(
    self,
    x: N.torch.Tensor,
    rms_weight: N.Optional[N.torch.Tensor] = None,
    residual: N.Optional[N.torch.Tensor] = None,
    quant_args: N.Optional[list] = None,
    update_hd: N.Optional[bool] = True,
    input_quant_args: N.Optional[N.Tuple[N.torch.Tensor, N.torch.Tensor]] = None,
    rms_norm_eps: N.Optional[float] = None,
) -> N.Tuple[N.torch.Tensor, N.Optional[N.torch.Tensor]]:
    if quant_args is not None:
        if input_quant_args is not None:
            raise ValueError("Specify only one prequantized input argument")
        input_quant_args = quant_args
    bias = self.bias if not self.skip_bias_add else None
    assert self.quant_method is not None
    if input_quant_args is not None:
        if not getattr(self.quant_method, "supports_prequantized_input", False):
            raise TypeError(
                f"{type(self.quant_method).__name__} does not support prequantized input"
            )
        output = self.quant_method.apply(
            self, x, bias, input_quant_args=input_quant_args
        )
    elif _use_fused_rms_quant and rms_weight is not None:
        if rms_norm_eps is None:
            raise ValueError("rms_norm_eps must be provided for fused RMS+quant")
        i_q, _scales = lm_faster_rmsquant(
            input=x,
            rms_weight=rms_weight,
            epsilon=rms_norm_eps,
            quant_dtype=N.torch.int8,
            residual=residual,
            update_input=update_hd,
        )

        input_quant_args = (i_q, _scales)
        output = self.quant_method.apply(
            self, x, bias, input_quant_args=input_quant_args
        )
    else:
        output = self.quant_method.apply(self, x, bias)
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


def column_init(
    self,
    input_size: int,
    output_size: int,
    bias: bool = True,
    gather_output: bool = False,
    skip_bias_add: bool = False,
    params_dtype: N.Optional[torch.dtype] = None,
    quant_config: N.Optional[N.QuantizationConfig] = None,
    eps: N.Optional[float] = 1e-6,
    output_sizes: N.Optional[N.List[int]] = None,
    prefix: str = "",
    *,
    use_presharded_weights: bool = False,
    skip_block_quant_check: bool = False,
    parallel_group: N.Optional[N.LinearParallelGroup] = None,
):
    self._initialize_partition(
        input_size=input_size,
        output_size=output_size,
        bias=bias,
        gather_output=gather_output,
        skip_bias_add=skip_bias_add,
        params_dtype=params_dtype,
        quant_config=quant_config,
        output_sizes=output_sizes,
        prefix=prefix,
        use_presharded_weights=use_presharded_weights,
        skip_block_quant_check=skip_block_quant_check,
        parallel_group=parallel_group,
        tp_group=_resolve_linear_group(parallel_group),
        eps=eps,
    )


def column_initialize_partition(
    self,
    input_size: int,
    output_size: int,
    bias: bool = True,
    gather_output: bool = False,
    skip_bias_add: bool = False,
    params_dtype: N.Optional[N.torch.dtype] = None,
    quant_config: N.Optional[N.QuantizationConfig] = None,
    output_sizes: N.Optional[N.List[int]] = None,
    prefix: str = "",
    *,
    tp_group: N.GroupCoordinator | N._LogicalGroup | N._ReplicatedGroup | None,
    eps: float = 1e-6,
    use_presharded_weights: bool = False,
    skip_block_quant_check: bool = False,
    parallel_group: N.Optional[N.LinearParallelGroup] = None,
):
    # Bind the selected group before quantization creates weight loaders.
    self.tp_group = tp_group
    self.parallel_group = parallel_group
    _, tp_size = N.get_group_rank_size(tp_group)
    super(N.ColumnParallelLinear, self).__init__(
        input_size, output_size, skip_bias_add, params_dtype, quant_config, prefix
    )

    self.with_bias = bias
    self.gather_output = gather_output
    self.use_presharded_weights = use_presharded_weights
    self.eps = eps
    # Divide the weight matrix along the last dimension.
    assert self.quant_method is not None
    self.output_size_per_partition = N.divide(self.output_size, tp_size)
    self.output_partition_sizes = [self.output_size_per_partition]
    # If QKV or MergedColumn, use output size of each partition.
    if hasattr(self, "output_sizes"):
        self.output_partition_sizes = [
            N.divide(output_size, tp_size) for output_size in self.output_sizes
        ]

    if output_sizes is None:
        output_sizes = [output_size]

    self.quant_method.create_weights(
        layer=self,
        input_size_per_partition=self.input_size,
        output_partition_sizes=self.output_partition_sizes,
        input_size=self.input_size,
        output_size=self.output_size,
        params_dtype=self.params_dtype,
        skip_block_quant_check=skip_block_quant_check,
        weight_loader=(
            self.weight_loader_v2
            if (
                self.quant_method.__class__.__name__ in N.WEIGHT_LOADER_V2_SUPPORTED
                or (
                    self.scheme is not None
                    and self.scheme.requires_weight_loader_v2
                )
            )
            else self.weight_loader
        ),
    )
    if bias:
        self.bias = N.Parameter(
            N.torch.zeros(self.output_size_per_partition, dtype=params_dtype)
        )
        N.set_weight_attrs(
            self.bias,
            {
                "output_dim": 0,
                "weight_loader": self.weight_loader,
            },
        )
    else:
        self.register_parameter("bias", None)


def column_forward(
    self,
    input_,
    rms_weight=None,
    residual=None,
    update_hd=True,
    input_quant_args=None,
):
    bias = self.bias if not self.skip_bias_add else None

    if input_quant_args is not None or (
        _is_hcu and _use_fused_rms_quant and rms_weight is not None
    ):
        if input_quant_args is None:
            i_q, scales = lm_faster_rmsquant(
                input=input_,
                rms_weight=rms_weight,
                epsilon=self.eps,
                quant_dtype=N.torch.int8,
                residual=residual,
                update_input=update_hd,
            )
            input_quant_args = [i_q, scales]
        assert self.quant_method is not None
        if not getattr(self.quant_method, "supports_prequantized_input", False):
            raise TypeError(
                f"{type(self.quant_method).__name__} does not support prequantized input"
            )
        output_parallel = self.quant_method.apply(
            self, input_, bias, input_quant_args=input_quant_args
        )
    else:
        # Megatron SP "g": the input is this rank's [M_pad/tp, K] sequence shard;
        # all-gather to the full sequence and matmul. Participants (qkv/gate_up)
        # have gather_output=False, so there is no output all-gather to reconcile.
        _, tp_size = N.get_group_rank_size(self.tp_group)
        if N.get_forward().sp_active and tp_size > 1:
            output = N.layernorm_sp.column_parallel_g_matmul(self, input_, bias)
            output_bias = self.bias if self.skip_bias_add else None
            return output, output_bias

        # Matrix multiply.
        assert self.quant_method is not None
        output_parallel = self.quant_method.apply(self, input_, bias)
    if self.gather_output:
        # All-gather across the partitions.
        if self.parallel_group == "replicated":
            output = output_parallel
        elif self.parallel_group == "attn_tp":
            output = N.get_parallel().attn_tp_group.all_gather(output_parallel, -1)
        elif self.parallel_group == "tp":
            output = N.get_parallel().tp_group.all_gather(output_parallel, -1)
        else:
            output = N.tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


def merged_init(
    self,
    input_size: int,
    output_sizes: N.List[int],
    bias: bool = True,
    gather_output: bool = False,
    skip_bias_add: bool = False,
    params_dtype: N.Optional[torch.dtype] = None,
    quant_config: N.Optional[N.QuantizationConfig] = None,
    prefix: str = "",
    *,
    use_presharded_weights: bool = False,
    parallel_group: N.Optional[N.LinearParallelGroup] = None,
):
    self.eps = 1e-6  # TODO: use config.rms_norm_eps
    self.with_bias = bias
    self.output_sizes = output_sizes
    tp_group = _resolve_linear_group(parallel_group)
    _, tp_size = N.get_group_rank_size(tp_group)
    assert all(output_size % tp_size == 0 for output_size in output_sizes)
    self.use_presharded_weights = use_presharded_weights
    super(N.MergedColumnParallelLinear, self)._initialize_partition(
        input_size=input_size,
        output_size=sum(output_sizes),
        bias=bias,
        gather_output=gather_output,
        skip_bias_add=skip_bias_add,
        params_dtype=params_dtype,
        quant_config=quant_config,
        prefix=prefix,
        tp_group=tp_group,
        parallel_group=parallel_group,
        use_presharded_weights=use_presharded_weights,
    )
    self.prefix = prefix


def merged_forward(
    self,
    input_,
    rms_weight: N.Optional[N.torch.Tensor] = None,
    residual: N.Optional[N.torch.Tensor] = None,
    update_hd: N.Optional[bool] = True,
    input_quant_args: N.Optional[N.Tuple[N.torch.Tensor, N.torch.Tensor]] = None,
):
    if input_quant_args is not None:
        return super(N.MergedColumnParallelLinear, self).forward(input_, input_quant_args=input_quant_args)
    if _use_fused_rms_quant and rms_weight is not None:
        input_quant_args = None
        assert residual is not None and rms_weight is not None
        i_q, _scales = lm_faster_rmsquant(
            input=input_,
            rms_weight=rms_weight,
            epsilon=self.eps,
            quant_dtype=N.torch.int8,
            residual=residual,
            update_input=update_hd,
        )

        input_quant_args = [i_q, _scales]

        bias = self.bias if not self.skip_bias_add else None
        assert self.quant_method is not None
        if not getattr(self.quant_method, "supports_prequantized_input", False):
            raise TypeError(
                f"{type(self.quant_method).__name__} does not support prequantized input"
            )
        output_parallel = self.quant_method.apply(
            self, input_, bias, input_quant_args=input_quant_args
        )

        if self.gather_output:
            # All-gather across the partitions.
            output = N.tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        # if not self.return_bias:
        #     return output
        return output, residual, i_q, _scales, output_bias

    return super(N.MergedColumnParallelLinear, self).forward(input_)


def _around(method):
    def call(original, *args, **kwargs):
        if not is_hcu():
            return original(*args, **kwargs)
        return method(*args, **kwargs)
    return call


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    if is_hcu():
        N.LinearParallelGroup = Literal["tp", "attn_tp", "attn_cp", "replicated", "shared_experts_tp"] | N.ReplicatedParallelGroup
    for target, method in (
        ("_resolve_linear_group", _resolve_linear_group),
        ("ReplicatedLinear.forward", replicated_forward),
        ("ColumnParallelLinear.__init__", column_init),
        ("ColumnParallelLinear._initialize_partition", column_initialize_partition),
        ("ColumnParallelLinear.forward", column_forward),
        ("MergedColumnParallelLinear.__init__", merged_init),
        ("MergedColumnParallelLinear.forward", merged_forward),
    ):
        HookRegistry.register("sglang.srt.layers.linear." + target, _around(method), HookType.AROUND)
