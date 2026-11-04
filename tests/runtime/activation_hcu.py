from __future__ import annotations

import unittest
from unittest.mock import Mock,patch
import torch
import torch.nn.functional as F
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers import norm_activation as H
from sglang.srt.layers import activation as N
from sglang.srt.environ import envs
from sglang.srt.plugins.hook_registry import HookRegistry
from types import SimpleNamespace


class ActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset();H.register_hooks();HookRegistry.apply_hooks()

    def test_constructor_matches_hcu_aiter_and_deterministic_gates(self):
        with patch.object(N,'_use_aiter',True),patch.object(N,'publish_role',return_value=None),envs.SGLANG_OPT_USE_AITER_SILU_MUL.override(True):
            with patch.object(H,'is_hcu',return_value=True):self.assertEqual(N.SiluAndMul()._forward_method.__name__,'forward_cuda')
            with patch.object(H,'is_hcu',return_value=False):self.assertEqual(N.SiluAndMul()._forward_method.__name__,'forward_aiter')
        with patch.object(N,'_use_aiter',True),patch.object(N,'publish_role',return_value='target'),patch.object(N,'get_exec',return_value=SimpleNamespace(deterministic=SimpleNamespace(rl_on_policy_target='fsdp'))):
            self.assertEqual(N.SiluAndMul()._forward_method.__name__,'forward_native')

    def test_actual_lightop_public_forward_matches_fp32_reference(self):
        self.assertTrue(H.is_hcu());torch.manual_seed(316)
        for dtype in (torch.bfloat16,torch.float16):
            for batch,hidden in [(1,256),(8,2048),(64,512)]:
                with self.subTest(dtype=dtype,batch=batch,hidden=hidden):
                    layer=N.SiluAndMul();x=torch.randn(batch,2*hidden,device='cuda',dtype=dtype)
                    expected=(F.silu(x[:,:hidden].float())*x[:,hidden:].float()).to(dtype)
                    result=layer(x);torch.cuda.synchronize()
                    self.assertEqual(result.shape,(batch,hidden));self.assertEqual(result.dtype,dtype)
                    tolerance=.02 if dtype==torch.bfloat16 else .002
                    torch.testing.assert_close(result,expected,rtol=tolerance,atol=tolerance)

    def test_graph_replay_reads_updated_gate_and_up_input(self):
        layer=N.SiluAndMul();x=torch.randn(8,1024,device='cuda',dtype=torch.bfloat16)
        for _ in range(3):layer(x)
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=layer(x)
        for shift in (0.,.5):
            x.copy_(torch.randn_like(x)+shift);expected=(F.silu(x[:,:512].float())*x[:,512:].float()).to(x.dtype)
            graph.replay();torch.cuda.synchronize();torch.testing.assert_close(out,expected,rtol=.02,atol=.02)

    def test_non_hcu_preserves_effective_original(self):
        marker=object();original=Mock(return_value=marker);layer=object();value=object()
        with patch.object(H,'is_hcu',return_value=False):self.assertIs(H.around_silu_and_mul_forward_cuda(original,layer,value),marker)
        original.assert_called_once_with(layer,value)


if __name__=='__main__':unittest.main()
