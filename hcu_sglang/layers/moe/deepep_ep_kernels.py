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

import logging
from functools import lru_cache
from typing import Optional, Tuple

import torch

from hcu_sglang.utils.detection import is_dcu as is_hcu
import triton

from sglang.srt.environ import envs
from sglang.srt.utils import ceil_div, is_cuda, is_hip, is_musa

logger = logging.getLogger(__name__)


_is_cuda = is_cuda()
_is_hip = is_hip()
_is_hcu = is_hcu()
_is_musa = is_musa()

if _is_hip:
    # HIP does not provide the CUDA JIT ``per_token_group_quant`` symbol used
    # by the generic SGLang wrapper.  The platform-selected alias falls back
    # to the native Triton raw kernel instead.
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_max as _fp8_max
    from sglang.kernels.ops.quantization.fp8_kernel import (
        per_token_group_quant_fp8,
    )
elif _is_cuda or _is_musa:
    from sglang.kernels.ops.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8 as per_token_group_quant_fp8,
    )

import triton.language as tl

from sglang.srt.utils import get_bool_env_var

use_groupgemm = get_bool_env_var("SGLANG_GROUPGEMM", default="true")


@triton.jit
def _fwd_kernel_ep_scatter_1_use_groupgemm(
    num_recv_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)


@triton.jit
def _fwd_kernel_ep_scatter_1(
    num_recv_tokens_per_expert,
    num_valid_tokens_per_expert,
    expert_start_loc,
    m_indices,
    num_experts: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_EXPERT_NUM: tl.constexpr,
):
    cur_expert = tl.program_id(0)

    offset_cumsum = tl.arange(0, BLOCK_EXPERT_NUM)
    tokens_per_expert = tl.load(
        num_recv_tokens_per_expert + offset_cumsum,
        mask=offset_cumsum < num_experts,
        other=0,
    )
    cumsum = tl.cumsum(tokens_per_expert) - tokens_per_expert
    tl.store(expert_start_loc + offset_cumsum, cumsum, mask=offset_cumsum < num_experts)

    cur_expert_start = tl.load(expert_start_loc + cur_expert)
    cur_expert_padded_token_num = tl.load(num_recv_tokens_per_expert + cur_expert)
    cur_expert_valid_token_num = tl.load(num_valid_tokens_per_expert + cur_expert)

    m_indices_start_ptr = m_indices + cur_expert_start
    off_expert = tl.arange(0, BLOCK_E)

    for start_m in tl.range(0, cur_expert_padded_token_num, BLOCK_E, num_stages=4):
        offsets = start_m + off_expert
        tl.store(
            m_indices_start_ptr + offsets,
            tl.where(offsets < cur_expert_valid_token_num, cur_expert, -1),
        )


@triton.jit
def _fwd_kernel_ep_scatter_2(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_x_stride1,
    recv_x_scale,
    recv_x_scale_stride0,
    recv_x_scale_stride1,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_tensor_scale_stride1,
    output_index,
    output_index_stride0,
    output_index_stride1,
    expert_start,
    num_experts,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    SCALE_HIDDEN_SIZE: tl.constexpr,
    SCALE_HIDDEN_SIZE_PAD: tl.constexpr,
    # Platform-specific semaphore for atomic_add performance tuning
    ATOMIC_ADD_SEM: tl.constexpr,
    IS_FP8: tl.constexpr,
):
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)

    offset_in = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offset_in < HIDDEN_SIZE

    index_in_s = tl.arange(0, SCALE_HIDDEN_SIZE_PAD)
    mask_s = index_in_s < SCALE_HIDDEN_SIZE

    for token_id_int32 in range(start_token_id, total_token_num, grid_num):
        token_id = token_id_int32.to(tl.int64)
        to_copy = tl.load(recv_x + token_id * recv_x_stride0 + offset_in, mask=mask)
        if IS_FP8:
            to_copy_s = tl.load(
                recv_x_scale
                + token_id * recv_x_scale_stride0
                + index_in_s * recv_x_scale_stride1,
                mask=mask_s,
            )

        for topk_idx_int32 in tl.range(0, topk_num, 1, num_stages=4):
            topk_index = topk_idx_int32.to(tl.int64)
            global_expert_id = tl.load(
                recv_topk + token_id * recv_topk_stride0 + topk_index
            )
            expert_id = global_expert_id - expert_start
            output_index_ptr = (
                output_index + token_id * output_index_stride0 + topk_index
            )
            valid = (expert_id >= 0) & (expert_id < num_experts)
            # The post-permute path uses this index as the validity sentinel.
            # Initialize non-local/padding lanes in this same kernel.
            tl.store(output_index_ptr, -1)
            if valid:
                dest_token_index_int32 = tl.atomic_add(
                    expert_start_loc + expert_id, 1, sem=ATOMIC_ADD_SEM
                )
                dest_token_index = dest_token_index_int32.to(tl.int64)

                tl.store(output_index_ptr, dest_token_index_int32)
                output_tensor_ptr = (
                    output_tensor + dest_token_index * output_tensor_stride0
                )
                tl.store(output_tensor_ptr + offset_in, to_copy, mask=mask)
                if IS_FP8:
                    output_tensor_scale_ptr = (
                        output_tensor_scale
                        + dest_token_index * output_tensor_scale_stride0
                    )
                    tl.store(
                        output_tensor_scale_ptr
                        + index_in_s * output_tensor_scale_stride1,
                        to_copy_s,
                        mask=mask_s,
                    )


@triton.jit
def _fwd_kernel_ep_scatter_quant_fp8(
    total_token_num,
    expert_start_loc,
    recv_x,
    recv_x_stride0,
    recv_topk,
    recv_topk_stride0,
    recv_topk_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_scale,
    output_tensor_scale_stride0,
    output_index,
    output_index_stride0,
    expert_start,
    num_experts: tl.constexpr,
    topk_num: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    HIDDEN_SIZE_PAD: tl.constexpr,
    BIT8_MAX: tl.constexpr,
    ATOMIC_ADD_SEM: tl.constexpr,
):
    """Fuse HCU whole-row FP8 quantization with compact expert scatter.

    MiniMax channel-FP8 uses one FP32 activation scale per token.  Quantizing
    into a temporary [M, K] tensor and then copying every routed row performs
    an avoidable full read/write pass.  This kernel computes the scale once per
    source token and writes the FP8 row directly to all valid expert slots.
    """
    start_token_id = tl.program_id(0)
    grid_num = tl.num_programs(0)
    offsets = tl.arange(0, HIDDEN_SIZE_PAD)
    mask = offsets < HIDDEN_SIZE

    for token_id_int32 in range(start_token_id, total_token_num, grid_num):
        token_id = token_id_int32.to(tl.int64)
        values = tl.load(
            recv_x + token_id * recv_x_stride0 + offsets,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        absmax = tl.maximum(tl.max(tl.abs(values)), 1.0e-10)
        scale = absmax / BIT8_MAX
        scale_inv = 1.0 / scale
        quantized = tl.clamp(values * scale_inv, -BIT8_MAX, BIT8_MAX).to(
            output_tensor.dtype.element_ty
        )

        for topk_idx_int32 in tl.range(0, topk_num, 1, num_stages=4):
            topk_idx = topk_idx_int32.to(tl.int64)
            expert_id = tl.load(
                recv_topk + token_id * recv_topk_stride0 + topk_idx * recv_topk_stride1
            )
            expert_id = expert_id - expert_start
            if (expert_id >= 0) & (expert_id < num_experts):
                dest_token_index_int32 = tl.atomic_add(
                    expert_start_loc + expert_id, 1, sem=ATOMIC_ADD_SEM
                )
                dest_token_index = dest_token_index_int32.to(tl.int64)
                tl.store(
                    output_index + token_id * output_index_stride0 + topk_idx,
                    dest_token_index_int32,
                )
                tl.store(
                    output_tensor + dest_token_index * output_tensor_stride0 + offsets,
                    quantized,
                    mask=mask,
                )
                tl.store(
                    output_tensor_scale
                    + dest_token_index * output_tensor_scale_stride0,
                    scale,
                )


@torch.no_grad()
def ep_scatter(
    recv_x: torch.Tensor,
    recv_x_scale: torch.Tensor,
    recv_topk: torch.Tensor,
    num_recv_tokens_per_expert: torch.Tensor,
    num_valid_tokens_per_expert: torch.Tensor,
    expert_start_loc: torch.Tensor,
    output_tensor: torch.Tensor,
    output_tensor_scale: torch.Tensor,
    m_indices: torch.Tensor,
    output_index: torch.Tensor,
    scale_ue8m0: bool = False,
    quant_block_size: int = 128,
    expert_alignment: int = 128,
    expert_start: int = 0,
    quantize_fp8: bool = False,
):
    # tl.arange needs pow2, and the kernel's unmasked stores need BLOCK_E to
    # divide the expert_alignment-padded segments; lowbit satisfies both.
    BLOCK_E = expert_alignment & -expert_alignment
    BLOCK_D = quant_block_size  # block size of quantization
    num_warps = 8
    num_experts = num_recv_tokens_per_expert.shape[0]
    hidden_size = recv_x.shape[1]
    # grid = (triton.cdiv(hidden_size, BLOCK_D), num_experts)
    grid = num_experts
    if use_groupgemm:
        BLOCK_E = 256
        scale_hidden_size = recv_x_scale.shape[-1]
    else:
        BLOCK_E = 128  # token num of per expert is aligned to 128
        BLOCK_D = quant_block_size  # block size of quantization
        scale_hidden_size = hidden_size // BLOCK_D
    if scale_ue8m0:
        # ue8m0 scales are packed here (4 scales per int32),
        # hence the effective size of this dimension is divided by 4.
        scale_hidden_size = ceil_div(scale_hidden_size, 4)

    assert m_indices.shape[0] % BLOCK_E == 0
    is_fp8 = recv_x_scale is not None and recv_x.dtype != torch.bfloat16
    if quantize_fp8:
        assert _is_hcu, "fused compact quant-scatter is currently HCU-only"
        assert not scale_ue8m0
        assert recv_x.dtype == torch.bfloat16
        assert output_tensor.dtype == torch.float8_e4m3fn
        assert quant_block_size == hidden_size
        assert output_tensor_scale.dtype == torch.float32
        assert output_tensor_scale.shape[1] == 1
        assert recv_x_scale is None
    if is_fp8:
        assert recv_x_scale.dtype == output_tensor_scale.dtype, (
            f"recv_x_scale.dtype: {recv_x_scale.dtype}, output_tensor_scale.dtype: {output_tensor_scale.dtype}"
        )
        assert (
            recv_x_scale.shape[1] == output_tensor_scale.shape[1] == scale_hidden_size
        )

    if use_groupgemm:
        _fwd_kernel_ep_scatter_1_use_groupgemm[(grid,)](
            num_recv_tokens_per_expert,
            expert_start_loc,
            m_indices,
            num_experts=num_experts,
            num_warps=num_warps,
            BLOCK_E=BLOCK_E,
            BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
        )
    else:
        _fwd_kernel_ep_scatter_1[(grid,)](
            num_recv_tokens_per_expert,
            num_valid_tokens_per_expert,
            expert_start_loc,
            m_indices,
            num_experts=num_experts,
            num_warps=num_warps,
            BLOCK_E=BLOCK_E,
            BLOCK_EXPERT_NUM=triton.next_power_of_2(num_experts),
        )

    grid = min(recv_topk.shape[0], 1024 * 8)

    if quantize_fp8:
        _fwd_kernel_ep_scatter_quant_fp8[(grid,)](
            recv_topk.shape[0],
            expert_start_loc,
            recv_x,
            recv_x.stride(0),
            recv_topk,
            recv_topk.stride(0),
            recv_topk.stride(1),
            output_tensor,
            output_tensor.stride(0),
            output_tensor_scale,
            output_tensor_scale.stride(0),
            output_index,
            output_index.stride(0),
            expert_start,
            num_experts,
            topk_num=recv_topk.shape[1],
            num_warps=num_warps,
            HIDDEN_SIZE=hidden_size,
            HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
            BIT8_MAX=_fp8_max,
            ATOMIC_ADD_SEM=None if not _is_musa else "relaxed",
        )
        return

    _fwd_kernel_ep_scatter_2[(grid,)](
        recv_topk.shape[0],
        expert_start_loc,
        recv_x,
        recv_x.stride(0),
        recv_x.stride(1),
        recv_x_scale,
        recv_x_scale.stride(0) if is_fp8 else 0,
        recv_x_scale.stride(1) if is_fp8 else 0,
        recv_topk,
        recv_topk.stride(0),
        recv_topk.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        output_tensor_scale,
        output_tensor_scale.stride(0) if is_fp8 else 0,
        output_tensor_scale.stride(1) if is_fp8 else 0,
        output_index,
        output_index.stride(0),
        output_index.stride(1),
        expert_start,
        num_experts,
        topk_num=recv_topk.shape[1],
        num_warps=num_warps,
        HIDDEN_SIZE=hidden_size,
        HIDDEN_SIZE_PAD=triton.next_power_of_2(hidden_size),
        SCALE_HIDDEN_SIZE=scale_hidden_size,
        SCALE_HIDDEN_SIZE_PAD=triton.next_power_of_2(scale_hidden_size),
        # XXX (MUSA): Atomic add with "relaxed" semaphore on musa backend for better performance
        ATOMIC_ADD_SEM=None if not _is_musa else "relaxed",
        IS_FP8=is_fp8,
    )
    return


@triton.jit
def _fwd_kernel_ep_gather(
    total_token_num,
    input_token_num,
    input_tensor,
    input_tensor_stride0,
    input_tensor_stride1,
    recv_topk_ids,
    recv_topk_ids_stride0,
    recv_topk_ids_stride1,
    recv_topk_weight,
    recv_topk_weight_stride0,
    recv_topk_weight_stride1,
    input_index,
    input_index_stride0,
    input_index_stride1,
    output_tensor,
    output_tensor_stride0,
    output_tensor_stride1,
    topk_num: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    cur_block_int32 = tl.program_id(0)
    cur_block = cur_block_int32.to(tl.int64)

    start_cur_token_int32 = tl.program_id(1)

    grid_num = tl.num_programs(1)

    for cur_token_int32 in range(start_cur_token_int32, total_token_num, grid_num):
        cur_token = cur_token_int32.to(tl.int64)

        off_d = tl.arange(0, BLOCK_D)
        accumulator = tl.zeros([BLOCK_D], dtype=tl.float32)

        for topk_index_int32 in range(0, topk_num):
            topk_index = topk_index_int32.to(tl.int64)

            expert_id = tl.load(
                recv_topk_ids + cur_token * recv_topk_ids_stride0 + topk_index
            )
            if expert_id >= 0:
                source_token_index_int32 = tl.load(
                    input_index + cur_token * input_index_stride0 + topk_index
                )
                if (source_token_index_int32 >= 0) & (
                    source_token_index_int32 < input_token_num
                ):
                    source_token_index = source_token_index_int32.to(tl.int64)

                    acc_weight = tl.load(
                        recv_topk_weight
                        + cur_token * recv_topk_weight_stride0
                        + topk_index
                    )
                    tmp = tl.load(
                        input_tensor
                        + source_token_index * input_tensor_stride0
                        + cur_block * BLOCK_D
                        + off_d
                    )
                    accumulator += tmp.to(tl.float32) * acc_weight

        tl.store(
            output_tensor
            + cur_token * output_tensor_stride0
            + cur_block * BLOCK_D
            + off_d,
            accumulator.to(output_tensor.dtype.element_ty),
        )


@torch.no_grad()
def ep_gather(
    input_tensor: torch.Tensor,
    recv_topk_ids: torch.Tensor,
    recv_topk_weight: torch.Tensor,
    input_index: torch.Tensor,
    output_tensor: torch.Tensor,
):
    num_warps = 2
    num_tokens = output_tensor.shape[0]
    hidden_size = input_tensor.shape[1]
    if use_groupgemm:
        # The kernel has no tail mask, so BLOCK_D must divide hidden_size.
        # Use the largest power-of-two divisor up to 1024. For example, Kimi
        # K3 routed hidden size 3584 selects 512 instead of the invalid 1024.
        largest_pow2_divisor = hidden_size & -hidden_size
        BLOCK_D = min(largest_pow2_divisor, 1024)
    else:
        BLOCK_D = 128 if hidden_size % 1024 != 0 else 1024  # block size of quantization
    assert hidden_size % BLOCK_D == 0
    grid = (triton.cdiv(hidden_size, BLOCK_D), min(num_tokens, 1024))
    _fwd_kernel_ep_gather[grid](
        num_tokens,
        input_tensor.shape[0],
        input_tensor,
        input_tensor.stride(0),
        input_tensor.stride(1),
        recv_topk_ids,
        recv_topk_ids.stride(0),
        recv_topk_ids.stride(1),
        recv_topk_weight,
        recv_topk_weight.stride(0),
        recv_topk_weight.stride(1),
        input_index,
        input_index.stride(0),
        input_index.stride(1),
        output_tensor,
        output_tensor.stride(0),
        output_tensor.stride(1),
        topk_num=recv_topk_ids.shape[1],
        num_warps=num_warps,
        BLOCK_D=BLOCK_D,
    )
    return


@triton.jit
def count_expert_histogram_kernel(
    topk_ids_ptr,
    counts_ptr,
    numel,
    E: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * 1024 + tl.arange(0, 1024)  # [1024]
    mask = offs < numel
    expert_id = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
    valid_mask = (expert_id >= 0) & (expert_id < E)
    tl.atomic_add(counts_ptr + expert_id, 1, mask=valid_mask)


@triton.jit
def compute_slots_kernel(
    counts_ptr,
    slots_ptr,
    E,
):
    i = tl.program_id(0)
    if i < E:
        cnt = tl.load(counts_ptr + i)
        slot = (cnt + 255) // 256  # equivalent to ceil(cnt/256)
        tl.store(slots_ptr + i, slot)


@triton.jit
def build_m_indices_kernel(
    prefix_slots_ptr,
    actual_counts_ptr,
    m_indices_ptr,
    E,
    MAX_E: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total_slots = tl.load(prefix_slots_ptr + E - 1)
    total_elements = total_slots * 256

    mask = offs < total_elements

    slot_id = offs >> 8  # Equivalent to offs // 256

    expert = tl.zeros([BLOCK], dtype=tl.int32)
    for i in range(0, MAX_E):
        ps = tl.load(prefix_slots_ptr + i, mask=i < E, other=2**30)
        expert += slot_id >= ps

    start_slot = tl.where(
        expert > 0, tl.load(prefix_slots_ptr + expert - 1, mask=expert > 0, other=0), 0
    )

    rel_slot = slot_id - start_slot
    pos_in_slot = offs & 0xFF
    rel_pos = rel_slot * 256 + pos_in_slot

    count_for_expert = tl.load(actual_counts_ptr + expert, mask=mask)

    valid_pos = rel_pos < count_for_expert

    result = tl.where(valid_pos, expert, -1)
    tl.store(m_indices_ptr + offs, result, mask=mask)


def build_m_indices_triton(topk_ids, device, num_experts):

    actual_counts = torch.zeros(num_experts, device=topk_ids.device, dtype=torch.int32)
    grid = (triton.cdiv(topk_ids.numel(), 1024),)
    count_expert_histogram_kernel[grid](
        topk_ids, actual_counts, topk_ids.numel(), E=num_experts
    )

    slots_per_expert = torch.empty_like(actual_counts)
    compute_slots_kernel[(num_experts,)](actual_counts, slots_per_expert, num_experts)

    prefix_slots = torch.cumsum(slots_per_expert, dim=0)

    total_slots = prefix_slots[-1].item()
    total_elements = total_slots * 256
    m_indices = torch.full(
        (total_elements,),
        -1,
        device=device,
        dtype=torch.int32,
    )

    BLOCK = 1024
    grid = (triton.cdiv(total_elements, BLOCK),)
    build_m_indices_kernel[grid](
        prefix_slots,
        actual_counts,
        m_indices,
        num_experts,
        MAX_E=num_experts,
        BLOCK=BLOCK,
    )
    return m_indices

