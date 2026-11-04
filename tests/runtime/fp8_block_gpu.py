from __future__ import annotations
import unittest
import torch
from types import SimpleNamespace
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import fp8_block as C
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.plugins.hook_registry import HookRegistry


class BlockFp8GpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset();C.register_hooks();HookRegistry.apply_hooks()

    def setUp(self):
        self.assertTrue(C.is_hcu());self.assertFalse(C.N._use_aiter);torch.manual_seed(846)
        self.method=object.__new__(Fp8LinearMethod)
        self.method.use_marlin=False;self.method.block_fp8_as_mxfp8=False;self.method.use_mxfp8=False
        self.method.convert_mxfp8_to_block=False;self.method.weight_block_size=[128,128]
        self.method.block_quant=True;self.method.quant_config=SimpleNamespace(weight_block_size=[128,128])
        self.method.w8a8_block_fp8_linear=C.N._dispatch_auto_backend()
        self.assertIs(self.method.w8a8_block_fp8_linear,C.hipblaslt_w8a8_block_fp8_linear)
        self.weight=(torch.randn(256,512,device='cuda')*.25).to(torch.float8_e4m3fn)
        self.scales=torch.tensor([[.25,.5,.75,1.],[1.,.75,.5,.25]],device='cuda')
        self.layer=torch.nn.Module()
        self.layer.weight=torch.nn.Parameter(self.weight.clone(),requires_grad=False)
        self.layer.weight_scale_inv=torch.nn.Parameter(self.scales.clone(),requires_grad=False)
        self.method.process_weights_after_loading_block_quant(self.layer)

    def reference(self,x,bias=None):
        flat=x.reshape(-1,512)
        q,s=C.per_token_group_quant_fp8_hcu(flat,128,column_major_scales=False)
        expanded_x=q.float()*s.repeat_interleave(128,1)
        expanded_w=self.weight.float()*self.scales.repeat_interleave(128,0).repeat_interleave(128,1)
        out=(expanded_x@expanded_w.T).to(x.dtype)
        if bias is not None:out=(out+bias).to(x.dtype)
        return out.reshape(*x.shape[:-1],256)

    def test_public_postload_to_vendor_gemm_bf16_and_fp16(self):
        for dtype in [torch.bfloat16,torch.float16]:
            for rows in [1,17,32]:
                x=torch.randn(rows,512,device='cuda',dtype=dtype)
                bias=torch.linspace(-.25,.5,256,device='cuda',dtype=dtype)
                out=self.method.apply(self.layer,x,bias)
                torch.cuda.synchronize();self.assertEqual(out.dtype,dtype)
                torch.testing.assert_close(out,self.reference(x,bias),rtol=.025,atol=.025)

    def test_multidimensional_input_preserves_leading_token_dimensions(self):
        x=torch.randn(2,8,512,device='cuda',dtype=torch.bfloat16)
        out=self.method.apply(self.layer,x)
        torch.cuda.synchronize();self.assertEqual(out.shape,(2,8,256))
        torch.testing.assert_close(out,self.reference(x),rtol=.025,atol=.025)

    def test_graph_replay_uses_current_input(self):
        x=torch.randn(16,512,device='cuda',dtype=torch.bfloat16)
        for _ in range(3): self.method.apply(self.layer,x)
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=self.method.apply(self.layer,x)
        for multiplier in [.5,1.75]:
            x.copy_(torch.randn_like(x)*multiplier)
            graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(out,self.reference(x),rtol=.025,atol=.025)


if __name__=='__main__':unittest.main()
