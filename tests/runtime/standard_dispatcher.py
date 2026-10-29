from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat

bootstrap_aiter_compat()

from sglang.srt.layers.moe.token_dispatcher import standard as S
from sglang.srt.layers.moe.topk import StandardTopKOutput
from hcu_sglang.layers.moe import standard_dispatcher as plugin


class StandardDispatcherTests(unittest.TestCase):
    def make_dispatcher(self):
        backend = SimpleNamespace()
        for name in ('aiter', 'auto', 'flashinfer_cutlass', 'flashinfer_cutedsl',
                     'flashinfer_trtllm', 'experimental_sgl_trtllm', 'flashinfer_trtllm_routed',
                     'hpc_ops', 'flashinfer_megamoe', 'flashinfer_mxfp4'):
            setattr(backend, 'is_' + name, lambda: False)
        config = SimpleNamespace(num_experts=5, num_local_experts=3, num_fused_shared_experts=1)
        with patch.object(S, 'get_parallel', return_value=SimpleNamespace(moe_ep_size=2, moe_ep_rank=0)), \
             patch.object(S, 'get_moe_runner_backend', return_value=backend):
            return S.StandardDispatcher(config)

    def dispatch(self, dispatcher, ids, *, hcu=True, pad=False):
        topk = StandardTopKOutput(torch.ones_like(ids, dtype=torch.float32), ids, None)
        hidden = torch.arange(ids.shape[0] * 4).reshape(ids.shape[0], 4).float()
        with patch.object(plugin, 'is_dcu', return_value=hcu), \
             patch.object(S, 'get_device', return_value='cpu'), \
             patch.object(S, 'should_use_flashinfer_cutlass_moe_fp4_allgather', return_value=False), \
             patch.object(S, '_MASK_DP_PAD_MOE', pad), \
             patch.object(S, 'is_dp_max_padding', return_value=pad), \
             patch.object(S, 'mask_dp_pad_moe_topk_ids', side_effect=lambda values: values[-1].fill_(-1)):
            return plugin.around_dispatch(S.StandardDispatcher.dispatch, dispatcher, hidden, topk)

    def test_real_dispatch_preserves_drop_ids_and_bounds_and_shared_experts(self):
        dispatcher = self.make_dispatcher()
        ids = torch.tensor([[-1, 0, 1, 2, 4, 5, -2]], dtype=torch.int64)
        output = self.dispatch(dispatcher, ids)
        self.assertEqual(output.topk_output.topk_ids.tolist(), [[-1, 0, 1, -1, 2, -1, -1]])
        self.assertEqual(ids.tolist(), [[-1, 0, 1, 2, 4, 5, -2]])
        self.assertEqual(dispatcher.local_expert_mapping.tolist(), [0, 1, -1, -1, 2])

    def test_counterexample_original_aliases_drop_to_shared_expert(self):
        output = self.dispatch(self.make_dispatcher(), torch.tensor([[-1, 4]]), hcu=False)
        self.assertEqual(output.topk_output.topk_ids.tolist(), [[2, 2]])

    def test_dp_pad_mask_applies_after_mapping(self):
        output = self.dispatch(self.make_dispatcher(), torch.tensor([[0, 4], [1, 4]]), pad=True)
        self.assertEqual(output.topk_output.topk_ids.tolist(), [[0, 2], [-1, -1]])

    def test_empty_batch_and_skip_mapping(self):
        dispatcher = self.make_dispatcher()
        output = self.dispatch(dispatcher, torch.empty((0, 2), dtype=torch.int64))
        self.assertEqual(tuple(output.topk_output.topk_ids.shape), (0, 2))
        dispatcher.skip_local_expert_mapping = True
        output = self.dispatch(dispatcher, torch.tensor([[-1, 4]]))
        self.assertEqual(output.topk_output.topk_ids.tolist(), [[-1, 4]])


if __name__ == '__main__':
    unittest.main()
