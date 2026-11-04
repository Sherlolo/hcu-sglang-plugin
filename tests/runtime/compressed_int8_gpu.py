from __future__ import annotations
import os
import unittest
from unittest.mock import patch
import torch
from compressed_tensors.quantization import QuantizationStrategy as S
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import compressed_int8 as C
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsLinearMethod
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import CompressedTensorsW8A8Int8
from sglang.srt.plugins.hook_registry import HookRegistry


class CompressedInt8GpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from hcu_sglang import register
        HookRegistry.reset();register();HookRegistry.apply_hooks()

    def setUp(self):
        torch.manual_seed(5001)
        self.w=torch.randint(-31,32,(128,256),device='cuda',dtype=torch.int8)
        self.ws=torch.linspace(.01,.05,128,device='cuda').reshape(128,1).to(torch.bfloat16)
        self.x=torch.randn(17,256,device='cuda',dtype=torch.bfloat16)*.25
        self.bias=torch.linspace(-.25,.5,128,device='cuda',dtype=torch.bfloat16)

    def make(self,strategy):
        with patch.dict(os.environ,{'W8A8_SUPPORT_METHODS':str(strategy)}):scheme=CompressedTensorsW8A8Int8(S.CHANNEL,False,True)
        l=torch.nn.Module();l.scheme=scheme;m=CompressedTensorsLinearMethod(None)
        with torch.device('cuda'):m.create_weights(l,256,[128],256,128,torch.bfloat16,weight_loader=lambda *a,**k:None)
        l.weight.data.copy_(self.w);l.weight_scale=torch.nn.Parameter(self.ws.clone(),requires_grad=False)
        m.process_weights_after_loading(l)
        self.assertEqual(l.weight_scale.dtype,torch.float32)
        return m,l

    def reference(self,q,s,strategy,bias):
        out=(q.float()*s)@(self.w.float()*self.ws.float()).T
        if strategy==3:
            out=out.to(torch.bfloat16)
            if bias is not None:out=(out+bias).to(torch.bfloat16)
            return out
        if bias is not None:out+=bias.float()
        return out.to(torch.bfloat16)

    def test_all_four_real_backends_after_public_postload(self):
        for strategy in [1,2,3,4]:
            with self.subTest(strategy=strategy):
                m,l=self.make(strategy)
                q,s=C.per_token_quant_int8(self.x)
                out=m.apply(l,self.x,self.bias)
                torch.cuda.synchronize();self.assertEqual(out.dtype,torch.bfloat16);self.assertEqual(out.shape,(17,128))
                torch.testing.assert_close(out,self.reference(q,s,strategy,self.bias),rtol=.025,atol=.025)

    def test_prequantized_rms_pair_is_consumed_by_actual_blaslt(self):
        m,l=self.make(3);q,s=C.per_token_quant_int8(self.x)
        with patch.object(C,'_use_fused_rms_quant',True),patch.object(C,'per_token_quant_int8',side_effect=AssertionError('requant')):
            out=m.apply(l,self.x,self.bias,input_quant_args=[q,s])
        torch.cuda.synchronize();torch.testing.assert_close(out,self.reference(q,s,3,self.bias),rtol=.025,atol=.025)

    def test_silu_pair_is_consumed_by_actual_triton(self):
        m,l=self.make(1);q,s=C.per_token_quant_int8(self.x)
        with patch.object(C,'_use_fused_silu_mul_quant',True),patch.object(C,'per_token_quant_int8',side_effect=AssertionError('requant')):
            out=m.apply(l,self.x,None,silu_quant_args=[q,s])
        torch.cuda.synchronize();torch.testing.assert_close(out,self.reference(q,s,1,None),rtol=.025,atol=.025)

    def test_blaslt_graph_reads_updated_activation_scale_and_bias(self):
        m,l=self.make(3);q,s=C.per_token_quant_int8(self.x)
        with patch.object(C,'_use_fused_rms_quant',True):
            for _ in range(3):m.apply(l,self.x,self.bias,input_quant_args=[q,s])
            torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=m.apply(l,self.x,self.bias,input_quant_args=[q,s])
            for factor in [.5,1.75]:
                s.mul_(factor);self.bias.mul_(factor)
                graph.replay();torch.cuda.synchronize()
                torch.testing.assert_close(out,self.reference(q,s,3,self.bias),rtol=.025,atol=.025)


if __name__=='__main__':unittest.main()
