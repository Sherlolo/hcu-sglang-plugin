from __future__ import annotations
import json
import unittest
from unittest.mock import patch
import torch
from compressed_tensors.quantization import QuantizationArgs
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import compressed_fp8 as C, w8a8_fp8 as H, fp8_block as B
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsLinearMethod
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import CompressedTensorsW8A8Fp8
from sglang.srt.plugins.hook_registry import HookRegistry


class CompressedFp8GpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();H.register_hooks();HookRegistry.apply_hooks()

    def setUp(self):
        self.assertTrue(C.is_hcu());self.assertTrue(C._use_lightop_channel_fp8);torch.manual_seed(4701)
        self.w=(torch.randn(256,512,device='cuda')*.25).to(torch.float8_e4m3fn)
        self.ws=torch.linspace(.125,.75,256,device='cuda').reshape(256,1)
        self.q=(torch.randn(17,512,device='cuda')*.25).to(torch.float8_e4m3fn)
        self.s=torch.linspace(.125,.5,17,device='cuda').reshape(17,1)
        self.bias=torch.linspace(-.25,.5,256,device='cuda')

    def layer(self):
        l=torch.nn.Module();l.scheme=CompressedTensorsW8A8Fp8(QuantizationArgs(num_bits=8,type='float',strategy='channel'),False)
        m=CompressedTensorsLinearMethod(None)
        with torch.device('cuda'):m.create_weights(l,512,[256],512,256,torch.bfloat16,weight_loader=lambda *a,**k:None)
        l.weight.data.copy_(self.w);l.weight_scale.data.copy_(self.ws)
        m.process_weights_after_loading(l)
        return m,l

    def reference(self,q,s,dtype,bias=None):
        out=(q.float()*s)@(self.w.float()*self.ws).T
        if bias is not None:out+=bias.float()
        return out.to(dtype)

    def test_actual_native_linear_optin_prequantized_bf16_fp16_and_bias(self):
        m,l=self.layer();self.assertEqual(l.weight.shape,(256,512));self.assertEqual(l.weight.dtype,torch.float8_e4m3fn)
        for dtype in [torch.bfloat16,torch.float16]:
            with patch.object(C,'hipblaslt_w8a8_channelwise_gemm',wraps=C.hipblaslt_w8a8_channelwise_gemm) as primary,patch.object(C,'hipblaslt_w8a8_channelwise_gemm_kme',wraps=C.hipblaslt_w8a8_channelwise_gemm_kme) as kme:
                out=m.apply(l,(self.q,self.s,dtype),self.bias.to(dtype))
            torch.cuda.synchronize();self.assertEqual(out.dtype,dtype)
            torch.testing.assert_close(out,self.reference(self.q,self.s,dtype,self.bias.to(dtype)),rtol=.025,atol=.025)
            print(json.dumps({'lane':'natural_prequantized','dtype':str(dtype),'primary_calls':primary.call_count,'kme_calls':kme.call_count}))

    def test_dynamic_multidimensional_input_uses_real_lightop_quantization(self):
        m,l=self.layer();x=torch.randn(2,8,512,device='cuda',dtype=torch.bfloat16)
        q,s=C.lightop_per_token_quant_fp8(x.reshape(-1,512),dtype=torch.float8_e4m3fn)
        out=m.apply(l,x,self.bias.to(x.dtype));torch.cuda.synchronize()
        self.assertEqual(out.shape,(2,8,256))
        torch.testing.assert_close(out,self.reference(q,s,x.dtype,self.bias.to(x.dtype)).reshape(2,8,256),rtol=.025,atol=.025)

    def test_optout_native_linear_preserves_requested_tuple_output_dtype(self):
        with patch.object(C,'_use_lightop_channel_fp8',False):
            m,l=self.layer();self.assertEqual(l.weight.shape,(512,256))
            for dtype in [torch.bfloat16,torch.float16]:
                out=m.apply(l,(self.q,self.s,dtype),self.bias.to(dtype))
                torch.cuda.synchronize();self.assertEqual(out.dtype,dtype)
                ref=((self.q.float()*self.s)@(self.w.float()*self.ws).T).to(dtype)
                ref=(ref+self.bias.to(dtype)).to(dtype)
                torch.testing.assert_close(out,ref,rtol=.025,atol=.025)

    def test_actual_kme_backend_after_fixture_primary_miss(self):
        m,l=self.layer()
        with patch.object(C,'hipblaslt_w8a8_channelwise_gemm',side_effect=RuntimeError('fixture primary miss')),patch.object(C,'hipblaslt_w8a8_channelwise_gemm_kme',wraps=C.hipblaslt_w8a8_channelwise_gemm_kme) as kme:
            out=m.apply(l,(self.q,self.s,torch.bfloat16),self.bias.to(torch.bfloat16))
        torch.cuda.synchronize();self.assertEqual(kme.call_count,1)
        torch.testing.assert_close(out,self.reference(self.q,self.s,torch.bfloat16,self.bias.to(torch.bfloat16)),rtol=.025,atol=.025)
        print(json.dumps({'lane':'fixture_primary_miss_real_kme','kme_calls':kme.call_count}))

    def test_native_block_scheme_pairs_hipblaslt_postload_with_real_gemm(self):
        with patch.object(C.Q,'dispatch_w8a8_block_fp8_linear',return_value=B.hipblaslt_w8a8_block_fp8_linear):
            scheme=CompressedTensorsW8A8Fp8(QuantizationArgs(num_bits=8,type='float',strategy='block',block_structure=[128,128]),False)
        l=torch.nn.Module();l.scheme=scheme;l.tp_group=None;m=CompressedTensorsLinearMethod(None)
        with torch.device('cuda'):m.create_weights(l,512,[256],512,256,torch.bfloat16,weight_loader=lambda *a,**k:None)
        bs=torch.tensor([[.25,.5,1.,2.],[2.,1.,.5,.25]],device='cuda')
        l.weight.data.copy_(self.w);l.weight_scale.data.copy_(bs);m.process_weights_after_loading(l)
        self.assertEqual(l.weight.shape,(512,256));self.assertEqual(l.weight_scale.shape,(4,2))
        x=torch.randn(17,512,device='cuda',dtype=torch.bfloat16)
        q,s=B.per_token_group_quant_fp8_hcu(x,128,column_major_scales=False)
        expanded_x=q.float()*s.repeat_interleave(128,1)
        expanded_w=self.w.float()*bs.repeat_interleave(128,0).repeat_interleave(128,1)
        ref=(expanded_x@expanded_w.T).to(x.dtype)
        out=m.apply(l,x);torch.cuda.synchronize()
        torch.testing.assert_close(out,ref,rtol=.025,atol=.025)

    def test_graph_replay_reads_updated_activation_and_weight_scales(self):
        m,l=self.layer()
        for _ in range(3):m.apply(l,(self.q,self.s,torch.bfloat16))
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=m.apply(l,(self.q,self.s,torch.bfloat16))
        for multiplier in [.5,1.75]:
            self.s.copy_(torch.linspace(.125,.5,17,device='cuda').reshape(17,1)*multiplier)
            l.weight_scale.data.copy_(self.ws*multiplier)
            graph.replay();torch.cuda.synchronize()
            ref=((self.q.float()*self.s)@(self.w.float()*self.ws*multiplier).T).to(torch.bfloat16)
            torch.testing.assert_close(out,ref,rtol=.025,atol=.025)


if __name__=='__main__':unittest.main()
