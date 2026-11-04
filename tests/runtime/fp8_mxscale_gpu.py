from __future__ import annotations
import unittest
import torch
from hcu_sglang.kernels.rmsnorm_fp8_e8m0 import rmsnorm_fp8_e8m0_quant


class DenseMxscaleProducerGpuTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(4601)

    def verify(self,x,w,eps=1e-6):
        (q,s),y=rmsnorm_fp8_e8m0_quant(x,w,eps)
        torch.cuda.synchronize()
        ref=(x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+eps)*w.float()).to(x.dtype)
        torch.testing.assert_close(y,ref,rtol=.016,atol=.016)
        m,n=x.shape;g=n//128
        self.assertEqual(q.shape,x.shape);self.assertEqual(s.shape,(m,g));self.assertTrue(s.is_contiguous())
        if m:
            logical_bytes=s.view(torch.uint8).reshape(g,m).T
            actual_y=y.float().reshape(m,g,128)
            amax=actual_y.abs().amax(-1)
            exponent=torch.ceil(torch.log2(amax/448))
            expected_scale=torch.exp2(exponent)
            self.assertTrue(bool((expected_scale>0).all()))
            torch.testing.assert_close(logical_bytes.to(torch.int16),exponent.to(torch.int16)+127,rtol=0,atol=0)
            qref=(actual_y/expected_scale.unsqueeze(-1)).clamp(-448,448).to(torch.float8_e4m3fn).reshape(m,n)
            torch.testing.assert_close(q.float(),qref.float(),rtol=0,atol=0)
        return q,s,y

    def test_real_fused_norm_quant_scale_bytes_and_non_power_two_groups(self):
        for m,n in [(1,256),(17,384),(8,7168)]:
            x=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
            w=torch.randn(n,device='cuda',dtype=torch.bfloat16)
            self.verify(x,w)

    def test_empty_and_zero_inputs(self):
        x=torch.zeros(0,384,device='cuda',dtype=torch.bfloat16);w=torch.ones(384,device='cuda',dtype=x.dtype)
        (q,s),y=rmsnorm_fp8_e8m0_quant(x,w,1e-6)
        self.assertEqual(q.shape,(0,384));self.assertEqual(s.shape,(0,3));self.assertEqual(y.shape,x.shape)
        (q,s),y=rmsnorm_fp8_e8m0_quant(torch.zeros(7,384,device='cuda',dtype=x.dtype),w,1e-6)
        torch.cuda.synchronize();self.assertEqual(q.float().abs().sum().item(),0);self.assertEqual(s.view(torch.uint8).sum().item(),0)
        self.assertEqual(y.abs().sum().item(),0)

    def test_graph_replay_reads_current_input_and_norm_weight(self):
        x=torch.randn(17,384,device='cuda',dtype=torch.bfloat16)
        w=torch.randn(384,device='cuda',dtype=x.dtype)
        for _ in range(3):rmsnorm_fp8_e8m0_quant(x,w,1e-6)
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):(q,s),y=rmsnorm_fp8_e8m0_quant(x,w,1e-6)
        for factor in [.5,2.]:
            x.copy_(torch.randn_like(x)*factor);w.copy_(torch.randn_like(w)*factor)
            graph.replay();torch.cuda.synchronize()
            rq,rs,ry=self.verify(x,w)
            torch.testing.assert_close(q.float(),rq.float(),rtol=0,atol=0)
            torch.testing.assert_close(s.view(torch.uint8),rs.view(torch.uint8),rtol=0,atol=0)
            torch.testing.assert_close(y,ry,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
