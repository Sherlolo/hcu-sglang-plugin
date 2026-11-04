from __future__ import annotations

import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import fp8_linear as C,w8a8_fp8 as H
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.plugins.hook_registry import HookRegistry
from types import SimpleNamespace


class Fp8LinearGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset();H.register_hooks();HookRegistry.apply_hooks()

    def setUp(self):
        self.assertTrue(C.is_hcu());torch.manual_seed(818)
        self.q=(torch.randn(8,256,device='cuda')*.3).to(torch.float8_e4m3fn)
        self.s=torch.linspace(.25,.5,8,device='cuda').reshape(8,1)
        self.w=(torch.randn(128,256,device='cuda')*.3).to(torch.float8_e4m3fn).t()
        self.ws=torch.linspace(.125,.75,128,device='cuda').reshape(128,1)
        self.bias=torch.linspace(-.25,.5,128,device='cuda')

    def reference(self,dtype,bias=True):
        output=((self.q.float()*self.s)@(self.w.float()*self.ws.T)).to(dtype)
        if bias:output=(output+self.bias).to(dtype)
        return output

    def method(self):
        method=object.__new__(Fp8LinearMethod)
        method.use_marlin=False;method.block_fp8_as_mxfp8=False;method.use_mxfp8=False;method.block_quant=False;method.cutlass_fp8_supported=False;method.use_per_token_if_dynamic=True
        return method

    def test_actual_native_consumer_prequantized_deepgemm_and_output_dtype(self):
        layer=SimpleNamespace(weight=self.w,weight_scale=self.ws,input_scale=None)
        for dtype in [torch.bfloat16,torch.float16]:
            result=self.method().apply(layer,(self.q,self.s,dtype),self.bias)
            torch.cuda.synchronize();self.assertEqual(result.dtype,dtype);self.assertEqual(result.shape,(8,128))
            torch.testing.assert_close(result,self.reference(dtype),rtol=.025,atol=.025)

    def test_direct_multidimensional_carrier_keeps_all_token_rows(self):
        q=self.q.view(2,4,256);scale=self.s.view(2,4,1)
        result=C.N.apply_fp8_linear((q,scale),self.w,self.ws,bias=self.bias,cutlass_fp8_supported=False)
        torch.cuda.synchronize();self.assertEqual(result.shape,(2,4,128));torch.testing.assert_close(result,self.reference(torch.bfloat16).view(2,4,128),rtol=.025,atol=.025)

    def test_dynamic_and_compressed_quantization_match_supplied_quant_reference(self):
        x=torch.randn(8,256,device='cuda',dtype=torch.bfloat16)
        for compressed in (False,True):
            if compressed:q,scale=C.N.scaled_fp8_quant(x,None,num_token_padding=17,use_per_token_if_dynamic=True)
            else:q,scale=C.N.per_token_group_quant_fp8(x,group_size=256)
            ref=((q.float()*scale)@(self.w.float()*self.ws.T))[:8].to(x.dtype)
            ref=(ref+self.bias).to(x.dtype)
            result=C.N.apply_fp8_linear(x,self.w,self.ws,bias=self.bias,cutlass_fp8_supported=False,compressed_tensor_quant=compressed,use_per_token_if_dynamic=True)
            torch.cuda.synchronize();self.assertEqual(result.shape,(8,128));torch.testing.assert_close(result,ref,rtol=.025,atol=.025)

    def test_graph_replay_reads_updated_prequantized_scale(self):
        for _ in range(3):C.N.apply_fp8_linear(self.q,self.w,self.ws,input_scale=self.s,bias=self.bias,cutlass_fp8_supported=False)
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=C.N.apply_fp8_linear(self.q,self.w,self.ws,input_scale=self.s,bias=self.bias,cutlass_fp8_supported=False)
        for multiplier in (.5,1.5):
            self.s.copy_(torch.linspace(.25,.5,8,device='cuda').reshape(8,1)*multiplier)
            graph.replay();torch.cuda.synchronize();torch.testing.assert_close(out,self.reference(torch.bfloat16),rtol=.025,atol=.025)


if __name__=='__main__':unittest.main()
