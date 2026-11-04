from __future__ import annotations
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import fp8_block as C
from sglang.srt.plugins.hook_registry import HookRegistry


class Fp8ReferenceGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset();C.register_hooks();HookRegistry.apply_hooks();torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(4801)
        self.a=(torch.randn(17,257,device='cuda')*.25).to(torch.float8_e4m3fn)
        self.b=(torch.randn(257,129,device='cuda')*.25).to(torch.float8_e4m3fn)
        self.sa=torch.ones(17,3,device='cuda')*.5
        self.sb=torch.tensor([[.25,.5],[1.,2.],[.5,.25]],device='cuda')

    def reference(self):
        a=self.a.float().cpu().double();b=self.b.float().cpu().double()
        sa=self.sa.cpu().double();sb=self.sb.cpu().double();out=torch.zeros(17,129,dtype=torch.float64)
        for k0 in range(0,257,128):
            for n0 in range(0,129,128):
                out[:,n0:n0+128]+=(a[:,k0:k0+128]*sa[:,k0//128:k0//128+1])@(b[k0:k0+128,n0:n0+128]*sb[k0//128,n0//128])
        return out

    def test_ragged_hcu_reference_against_cpu_double_block_products(self):
        for dtype in [torch.float32,torch.bfloat16,torch.float16]:
            out=C.N.torch_fp8_block_gemm(self.a,self.b,self.sa,self.sb,128,dtype)
            torch.cuda.synchronize();self.assertEqual(out.device.type,'cuda');self.assertEqual(out.shape,(17,129))
            tolerance=1e-5 if dtype==torch.float32 else .01
            torch.testing.assert_close(out.cpu(),self.reference().to(dtype),rtol=tolerance,atol=tolerance)

    def test_graph_replay_reads_changed_scale_buffers(self):
        for _ in range(3):C.N.torch_fp8_block_gemm(self.a,self.b,self.sa,self.sb,128)
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=C.N.torch_fp8_block_gemm(self.a,self.b,self.sa,self.sb,128)
        for factor in [.5,2.]:
            self.sa.copy_(torch.full_like(self.sa,factor));self.sb.mul_(2)
            graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(out.cpu(),self.reference().to(out.dtype),rtol=.01,atol=.01)


if __name__=='__main__':unittest.main()
