# Copyright 2023-2026 SGLang Team
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

from typing import List, Optional, Tuple
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.model_executor.runner import base_cuda_graph_runner as N
from hcu_sglang.utils.detection import is_hcu


def get_batch_sizes_to_capture(
    model_runner: ModelRunner,
    captured_req_width: int = 1,
    *,
    gathered_buffer_required: Optional[bool] = None,
) -> Tuple[List[int], List[int]]:
    """Build the (capture_bs, compile_bs) lists for the decode runner.

    Filters cuda_graph_config[decode].bs by attention-tp/cp alignment
    constraints and clamps to req_to_token_pool.size.
    """

    capture_bs = list(N.get_exec().graph.cuda_graph_config.decode.bs)
    num_max_requests = model_runner.req_to_token_pool.size
    dllm_config = DllmConfig.from_server_args(model_runner.server_args)
    if dllm_config is not None and dllm_config.requires_separate_context_encoding:
        num_max_requests = min(num_max_requests, dllm_config.max_running_requests)

    mul_base = N.get_cuda_graph_batch_size_alignment(
        gathered_buffer_required=gathered_buffer_required
    )
    # TBO splits each request's rows across two micro-batches, so the
    # alignment constraint applies per request rather than per token row.
    alignment_width = captured_req_width
    if N.get_exec().overlap.enable_two_batch_overlap:
        alignment_width = 1

    # pad `num_max_requests` to avoid being filtered out
    num_max_requests = N.ceil_align(num_max_requests, mul_base)
    if max(capture_bs) > num_max_requests:
        # In some cases (e.g., with a small GPU or --max-running-requests), the #max-running-requests
        # is very small. We add more values here to make sure we capture the maximum bs.
        capture_bs += [num_max_requests]

    # Model input token count = bs * alignment_width; must be a multiple of attn_tp_size.
    capture_bs = [bs for bs in capture_bs if bs * alignment_width % mul_base == 0]
    capture_bs = [bs for bs in capture_bs if bs <= num_max_requests]
    capture_bs = list(sorted(set(capture_bs)))

    assert len(capture_bs) > 0 and capture_bs[0] > 0, f"{capture_bs=}"
    compile_bs = (
        [bs for bs in capture_bs if bs <= N.get_exec().graph.torch_compile_max_bs]
        if N.get_flags().capture.enable_torch_compile
        else []
    )
    return capture_bs, compile_bs


def around_batch_sizes(original, *args, **kwargs):
    if not is_hcu():
        return original(*args, **kwargs)
    return get_batch_sizes_to_capture(*args, **kwargs)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register("sglang.srt.model_executor.runner.base_cuda_graph_runner.get_batch_sizes_to_capture", around_batch_sizes, HookType.AROUND)
