"""Shared runtime imports for extracted models.utils fork methods."""

from __future__ import annotations

import torch

from sglang.srt.layers.utils.cp_utils import is_prefill_context_parallel_enabled
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_cuda, is_hip

_is_cuda = is_cuda()
_is_hip = is_hip()
