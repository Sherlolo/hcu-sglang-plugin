from __future__ import annotations

import json
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import fp8 as Q,fp8_fallback as C
from sglang.srt.plugins.hook_registry import HookRegistry


def key(q,w,dtype):
    return (q.device.type,q.device.index,q.shape[0],w.shape[1],q.shape[1],q.dtype,dtype)


def reference(q,w,a,b,bias,dtype):
    return ((q.float()@w.float())*a*b.T+bias).to(dtype)


class Fp8FallbackGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset();Q.register_hooks();HookRegistry.apply_hooks()

    def setUp(self):
        self.assertTrue(C.is_hcu());C.TORCH_DEVICE_IDENTITY=None;C._HIPBLASLT_UNSUPPORTED_FP8_SHAPES.clear();torch.manual_seed(708)

    def inputs(self,m,k,n):
        q=(torch.randn(m,k,device='cuda')*.5).to(torch.float8_e4m3fn)
        w=(torch.randn(n,k,device='cuda')*.5).to(torch.float8_e4m3fn).t()
        a=torch.linspace(.25,1.25,m,device='cuda').reshape(m,1)
        b=torch.linspace(.125,.75,n,device='cuda').reshape(n,1)
        bias=torch.linspace(-.25,.5,n,device='cuda')
        return q,w,a,b,bias

    def test_real_public_backend_scaling_bias_padding_and_dtype(self):
        q,w,a,b,bias=self.inputs(32,256,128)
        result=C.N._apply_fallback_scaled_mm(q,w,a,b,(28,256),(2,14,128),bias,torch.bfloat16)
        expected=reference(q,w,a,b,bias,torch.bfloat16)[:28].reshape(2,14,128)
        torch.cuda.synchronize();torch.testing.assert_close(result,expected,rtol=.02,atol=.025)
        print(json.dumps({'shape':[32,128,256],'actual_cached_blaslt_failure':key(q,w,torch.bfloat16) in C._HIPBLASLT_UNSUPPORTED_FP8_SHAPES}),flush=True)

    def test_source_documented_m17_shape_and_live_graph_replay(self):
        q,w,a,b,bias=self.inputs(17,6144,768)
        result=C.N._apply_fallback_scaled_mm(q,w,a,b,q.shape,(17,768),bias,torch.bfloat16)
        torch.cuda.synchronize();torch.testing.assert_close(result,reference(q,w,a,b,bias,torch.bfloat16),rtol=.02,atol=.035)
        for _ in range(3):C.N._apply_fallback_scaled_mm(q,w,a,b,q.shape,(17,768),bias,torch.bfloat16)
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=C.N._apply_fallback_scaled_mm(q,w,a,b,q.shape,(17,768),bias,torch.bfloat16)
        for multiplier in (.5,1.5):
            a.copy_(torch.linspace(.25,1.25,17,device='cuda').reshape(17,1)*multiplier)
            graph.replay();torch.cuda.synchronize();torch.testing.assert_close(out,reference(q,w,a,b,bias,torch.bfloat16),rtol=.02,atol=.035)
        print(json.dumps({'shape':[17,768,6144],'actual_cached_blaslt_failure':key(q,w,torch.bfloat16) in C._HIPBLASLT_UNSUPPORTED_FP8_SHAPES,'graph_replay':True}),flush=True)

    def test_known_shape_cache_exercises_real_triton_math(self):
        q,w,a,b,bias=self.inputs(19,256,128)
        C._HIPBLASLT_UNSUPPORTED_FP8_SHAPES.add(key(q,w,torch.float16))
        result=C.N._apply_fallback_scaled_mm(q,w,a,b,q.shape,(19,128),bias,torch.float16)
        torch.cuda.synchronize();torch.testing.assert_close(result,reference(q,w,a,b,bias,torch.float16),rtol=.005,atol=.005)
        print(json.dumps({'shape':[19,128,256],'cache_fixture':'Preseeded known-unsupported entry only; real source-derived Triton kernel with explicit store-type cast, no math/backend stub.'}),flush=True)


if __name__=='__main__':unittest.main()
