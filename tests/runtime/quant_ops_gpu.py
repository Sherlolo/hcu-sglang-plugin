from __future__ import annotations
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.layers.quantization import quant_ops as C
from lightop import op as O


class SmoothAsmGpuTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5101)
        self.a=torch.randint(-100,101,(17,256),device='cuda',dtype=torch.int8)
        self.b=torch.randint(-31,32,(128,256),device='cuda',dtype=torch.int8).T
        self.sa=torch.linspace(.002,.01,17,device='cuda').reshape(17,1)
        self.sb=torch.linspace(.01,.1,128,device='cuda').reshape(128,1)
        self.assertTrue(C._is_hcu);self.assertTrue(C._w8a8_smooth_asm_supported(self.a.device))

    def reference(self,bias=None):
        out=(self.a.float()*self.sa)@(self.b.float()*self.sb.T)
        if bias is not None:out+=bias.float()
        return out.to(torch.bfloat16)

    def test_actual_small_m_asm_selected_and_numerically_correct(self):
        with patch.object(O,'gemm_w8a8_smooth_asm',wraps=O.gemm_w8a8_smooth_asm) as asm,patch.object(C.quant_ops,'triton_scaled_mm',wraps=C.quant_ops.triton_scaled_mm) as triton:
            out=C.triton_scaled_mm(self.a,self.b,self.sa,self.sb,torch.bfloat16)
        torch.cuda.synchronize();self.assertEqual(asm.call_count,1);self.assertEqual(triton.call_count,0)
        torch.testing.assert_close(out,self.reference(),rtol=.025,atol=.025)

    def test_bias_keeps_actual_triton_backend_and_fused_value(self):
        bias=torch.linspace(-.5,.5,128,device='cuda',dtype=torch.bfloat16)
        with patch.object(O,'gemm_w8a8_smooth_asm',wraps=O.gemm_w8a8_smooth_asm) as asm,patch.object(C.quant_ops,'triton_scaled_mm',wraps=C.quant_ops.triton_scaled_mm) as triton:
            out=C.triton_scaled_mm(self.a,self.b,self.sa,self.sb,torch.bfloat16,bias)
        torch.cuda.synchronize();self.assertEqual(asm.call_count,0);self.assertEqual(triton.call_count,1)
        torch.testing.assert_close(out,self.reference(bias),rtol=.025,atol=.025)

    def test_real_asm_graph_replay_reads_scale_updates(self):
        for _ in range(3):C.triton_scaled_mm(self.a,self.b,self.sa,self.sb,torch.bfloat16)
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=C.triton_scaled_mm(self.a,self.b,self.sa,self.sb,torch.bfloat16)
        for factor in [.5,1.75]:
            self.sa.mul_(factor);self.sb.mul_(factor)
            graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(out,self.reference(),rtol=.025,atol=.025)


if __name__=='__main__':unittest.main()
