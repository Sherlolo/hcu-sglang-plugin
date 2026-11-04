from __future__ import annotations
import math
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class LinearBf16Fp32GpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(1701)
        HookRegistry.reset();register();HookRegistry.apply_hooks()

    def data(self,m,n,k):
        return torch.randn(m,k,device='cuda',dtype=torch.bfloat16)/math.sqrt(k),torch.randn(n,k,device='cuda',dtype=torch.bfloat16)

    def verify(self,x,w,result):
        reference=x.cpu().double()@w.cpu().double().T
        self.assertEqual(result.dtype,torch.float32)
        self.assertEqual(result.device,x.device)
        self.assertTrue(bool(torch.isfinite(result).all()))
        torch.testing.assert_close(result.cpu().double(),reference,rtol=2e-4,atol=2e-5)

    def test_direct_kernel_normal_and_splitk_boundaries(self):
        from hcu_sglang.models.deepseek.current_v4.hcu_backend.hcu_linear_bf16_fp32 import hcu_linear_bf16_fp32
        shapes=[(1,256,128),(15,512,256),(16,256,6144),(17,512,6144),(64,512,128),(1,1024,512),(32,1024,4096),(7,2048,512),(8,2048,512),(64,2048,4096)]
        for shape in shapes:
            with self.subTest(shape=shape):
                x,w=self.data(*shape);result=hcu_linear_bf16_fp32(x,w);torch.cuda.synchronize();self.verify(x,w,result)

    def test_splitk_repeated_calls_are_zero_initialized(self):
        from hcu_sglang.models.deepseek.current_v4.hcu_backend.hcu_linear_bf16_fp32 import hcu_linear_bf16_fp32
        for n in [1024,2048]:
            x=torch.ones(17,512,device='cuda',dtype=torch.bfloat16)
            w=torch.ones(n,512,device='cuda',dtype=torch.bfloat16)
            for sign in [1,-1,0,1]:
                w.fill_(sign);result=hcu_linear_bf16_fp32(x,w);torch.cuda.synchronize()
                torch.testing.assert_close(result,torch.full_like(result,512*sign),rtol=0,atol=0)

    def test_public_sgl_dispatch_executes_owned_kernel(self):
        from hcu_sglang.models.deepseek.current_v4.hcu_backend import bf16_fp32 as O,hcu_linear_bf16_fp32 as K
        from sglang.kernels.ops.gemm import bf16_fp32 as N
        x,w=self.data(17,1024,512)
        with patch.object(O,'_linear_bf16_fp32_algo','sgl'),patch.object(O,'_linear_bf16_fp32_cublas',side_effect=AssertionError('unexpected fallback')),patch.object(K,'hcu_linear_bf16_fp32',wraps=K.hcu_linear_bf16_fp32) as actual:
            result=N.linear_bf16_fp32(x,w);torch.cuda.synchronize();actual.assert_called_once()
        self.verify(x,w,result)

    def test_auto_preserves_untuned_gfx938_profile(self):
        from hcu_sglang.models.deepseek.current_v4.hcu_backend import bf16_fp32 as O
        x,w=self.data(17,256,128)
        self.assertFalse(O._hcu_auto_profile_enabled())
        before=O._AUTO_SGL_DISABLED
        with patch.object(O,'_get_aiter_tgemm',side_effect=AssertionError('untuned profile must preserve source policy')):
            result=O._auto_dispatch_bf16_fp32(x,w);torch.cuda.synchronize()
        self.assertEqual(O._AUTO_SGL_DISABLED,before)
        self.assertEqual(result.dtype,torch.float32)
        reference=(x.cpu().double()@w.cpu().double().T).bfloat16().float()
        torch.testing.assert_close(result.cpu(),reference,rtol=0,atol=0)

    def test_graph_splitk_reinitializes_output_on_replay(self):
        from hcu_sglang.models.deepseek.current_v4.hcu_backend.hcu_linear_bf16_fp32 import hcu_linear_bf16_fp32
        x,w=self.data(17,1024,512)
        hcu_linear_bf16_fp32(x,w);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):result=hcu_linear_bf16_fp32(x,w)
        for sign in [1.,-1.,.5]:
            x.fill_(sign);w.fill_(1.);graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(result,torch.full_like(result,512*sign),rtol=0,atol=0)

    def test_unsupported_shapes_dtypes_layouts_and_alignment_are_rejected(self):
        from hcu_sglang.models.deepseek.current_v4.hcu_backend.hcu_linear_bf16_fp32 import hcu_linear_bf16_fp32_supported,hcu_linear_bf16_fp32
        pairs=[self.data(0,256,128),self.data(65,256,128),self.data(1,128,128),self.data(1,256,130)]
        x,w=self.data(1,256,128)
        pairs.extend([(x.float(),w),(x,w.float()),(x,torch.empty(256,256,device='cuda',dtype=torch.bfloat16)[:,::2]),(torch.empty(129,device='cuda',dtype=torch.bfloat16)[1:].reshape(1,128),w)])
        for a,b in pairs:
            self.assertFalse(hcu_linear_bf16_fp32_supported(a,b))
            with self.assertRaises(ValueError):hcu_linear_bf16_fp32(a,b)


if __name__=='__main__':unittest.main()
