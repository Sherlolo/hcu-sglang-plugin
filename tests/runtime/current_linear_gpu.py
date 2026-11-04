from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch
from types import SimpleNamespace
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers import current_linear as C,linear_fusion as H,fused_rms_quant as F
from hcu_sglang.layers.quantization import slimquant_w4a8 as Q
from sglang.srt.plugins.hook_registry import HookRegistry


def rms(value,weight,eps):
    return value.float()*torch.rsqrt(value.float().square().mean(-1,keepdim=True)+eps)*weight.float()


class CurrentLinearGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset();H.register_hooks();HookRegistry.apply_hooks()

    def setUp(self):
        self.assertTrue(C.is_hcu());self.assertTrue(C._use_fused_rms_quant)
        self.assertTrue(F.is_lightop_sglang_rms_quant_available())
        self.scope=ExitStack();self.addCleanup(self.scope.close)
        self.scope.enter_context(patch.object(C.N,'get_parallel',return_value=SimpleNamespace(tp_group=None,tp_size=1,tp_rank=0)))
        self.scope.enter_context(patch.object(C.N,'get_forward',return_value=SimpleNamespace(sp_active=False)))
        torch.manual_seed(146)

    def test_real_fused_producer_preserves_quantized_and_residual_consumers(self):
        for dtype in (torch.bfloat16,torch.float16):
            for update in (False,True):
                x=torch.randn(8,256,device='cuda',dtype=dtype);residual=torch.randn_like(x);weight=(torch.randn(256,device='cuda',dtype=dtype)*.1+1).contiguous()
                x0=x.clone();res0=residual.clone();expected=rms(x0.float()+res0.float(),weight,3e-4)
                q,scale=F.fused_rms_norm_per_token_quant(input=x,rms_weight=weight,epsilon=3e-4,residual=residual,update_input=update)
                torch.cuda.synchronize();self.assertEqual(q.dtype,torch.int8);self.assertEqual(scale.dtype,torch.float32);self.assertEqual(scale.shape,(8,1))
                tolerance=.025 if dtype==torch.bfloat16 else .012
                torch.testing.assert_close(q.float()*scale,expected,rtol=tolerance,atol=.025)
                torch.testing.assert_close(residual,(x0.float()+res0.float()).to(dtype),rtol=tolerance,atol=.025)
                if update:torch.testing.assert_close(x,expected.to(dtype),rtol=tolerance,atol=.025)
                else:self.assertTrue(torch.equal(x,x0))

    def make_projection(self):
        layer=C.N.ColumnParallelLinear(256,128,bias=False,eps=3e-4,params_dtype=torch.bfloat16).to('cuda')
        method=object.__new__(Q.SlimQuantW4A8Int8LinearMethod);method.w8a8_strategy=3;layer.quant_method=method
        layer.register_parameter('weight',torch.nn.Parameter(torch.randint(-7,8,(128,256),device='cuda',dtype=torch.int8),requires_grad=False))
        layer.weight_scale=torch.full((128,1),.125,device='cuda')
        method.process_weights_after_loading(layer)
        self.assertEqual(layer.weight.shape,(128,256))
        return layer

    def test_public_projection_fused_and_explicit_paths_use_actual_int8_gemm(self):
        layer=self.make_projection();weight=torch.ones(256,device='cuda',dtype=torch.bfloat16);x0=torch.randn(8,256,device='cuda',dtype=torch.bfloat16)
        q,scale=F.fused_rms_norm_per_token_quant(input=x0,rms_weight=weight,epsilon=layer.eps,update_input=False)
        reference=((q.float()*scale)@(layer.weight.float()*layer.weight_scale).T).to(x0.dtype)
        explicit,_=layer(x0,input_quant_args=(q,scale))
        fused,_=layer(x0.clone(),rms_weight=weight,update_hd=False)
        torch.cuda.synchronize();torch.testing.assert_close(explicit,reference,rtol=.02,atol=.025);torch.testing.assert_close(fused,reference,rtol=.02,atol=.025)
        static=x0.clone()
        for _ in range(3):layer(static,rms_weight=weight,update_hd=False)
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):output,_=layer(static,rms_weight=weight,update_hd=False)
        for shift in (0.,.25):
            static.copy_(x0+shift)
            q2,s2=F.fused_rms_norm_per_token_quant(input=static,rms_weight=weight,epsilon=layer.eps,update_input=False)
            expected=((q2.float()*s2)@(layer.weight.float()*layer.weight_scale).T).to(x0.dtype)
            graph.replay();torch.cuda.synchronize();torch.testing.assert_close(output,expected,rtol=.02,atol=.025)

    def test_packed_mla_normalizes_only_q_kv_and_retains_rope(self):
        self.assertTrue(F.is_lightop_sglang_mla_qkv_a_rms_quant_available())
        packed=torch.randn(8,2112,device='cuda',dtype=torch.bfloat16);before=packed.clone();qw=torch.ones(1536,device='cuda',dtype=packed.dtype);kw=torch.full((512,),1.125,device='cuda',dtype=packed.dtype)
        q,scale=F.fused_mla_qkv_a_rms_norm_per_token_quant(packed_input=packed,q_weight=qw,kv_weight=kw,q_epsilon=1e-6,kv_epsilon=3e-4)
        torch.cuda.synchronize();qref=rms(before[:,:1536],qw,1e-6);kvref=rms(before[:,1536:2048],kw,3e-4)
        torch.testing.assert_close(packed[:,:1536],qref.to(packed.dtype),rtol=.025,atol=.025);torch.testing.assert_close(packed[:,1536:2048],kvref.to(packed.dtype),rtol=.025,atol=.025)
        torch.testing.assert_close(q.float()*scale,qref,rtol=.025,atol=.025);self.assertTrue(torch.equal(packed[:,2048:],before[:,2048:]))


if __name__=='__main__':unittest.main()
