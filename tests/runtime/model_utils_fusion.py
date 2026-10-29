from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat

bootstrap_aiter_compat()

from sglang.srt.layers.cp import utils as cp_utils
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from hcu_sglang.models.model_utils.methods import utils as plugin


class ModelUtilsFusionTests(unittest.TestCase):
    def enabled(self, *, hcu=True, cuda=True, hip=False, dtype=torch.bfloat16, cp=False, mask=None, swa=False):
        pool = object.__new__(plugin.SWAKVPool) if swa else SimpleNamespace()
        pool.dtype = dtype
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND, input_ids=torch.arange(4), dcp_kv_mask=mask)
        strategy = SimpleNamespace(can_apply=lambda count, current: cp and count == 4 and current is batch)
        with patch.object(plugin.C, '_is_dcu', hcu), patch.object(plugin, '_is_cuda', cuda), \
             patch.object(plugin, '_is_hip', hip), patch.object(cp_utils, 'get_cp_strategy', return_value=strategy), \
             forward_context(ForwardContext(attn_backend=SimpleNamespace(token_to_kv_pool=pool))):
            self.assertFalse(hasattr(batch, 'token_to_kv_pool'))
            return plugin.enable_fused_set_kv_buffer(batch)

    def test_hcu_keeps_cuda_bf16_gate_and_rejects_hip_layout(self):
        self.assertTrue(self.enabled())
        self.assertFalse(self.enabled(cuda=False, hip=True))
        self.assertFalse(self.enabled(dtype=torch.float16))
        self.assertFalse(self.enabled(swa=True))

    def test_current_batch_cp_and_dcp_mask_disable_cuda_fusion(self):
        self.assertFalse(self.enabled(cp=True))
        self.assertFalse(self.enabled(mask=torch.tensor([True])))
        self.assertTrue(self.enabled(cp=False))

    def test_non_hcu_preserves_current_hip_gate(self):
        self.assertTrue(self.enabled(hcu=False, cuda=False, hip=True, dtype=torch.float16))
        self.assertFalse(self.enabled(hcu=False, cuda=False, hip=True, mask=torch.tensor([True])))
        self.assertFalse(self.enabled(hcu=False, cuda=False, hip=False))


if __name__ == '__main__':
    unittest.main()
