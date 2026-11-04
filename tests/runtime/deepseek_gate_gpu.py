from __future__ import annotations
import os
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class DeepseekGateGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def data(self):
        from sglang.srt.models import deepseek_v2 as N
        config=S(model_type='deepseek_v4',n_routed_experts=256,hidden_size=128,topk_method='noaux_tc',router_fp32=False)
        dtype=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device('cuda'):gate=N.MoEGate(config,None,is_hash_moe=True,is_deepseek_v4=True)
        finally:torch.set_default_dtype(dtype)
        with torch.no_grad():gate.weight.copy_(((torch.arange(256*128,device='cuda')%7-3).reshape(256,128)/8).bfloat16())
        x=((torch.arange(24*128,device='cuda')%5-2).reshape(24,128)/4).bfloat16()
        ids=torch.arange(24,device='cuda',dtype=torch.int64)%17
        table=((torch.arange(17*8,device='cuda')*7+5)%256).reshape(17,8).int()
        self.assertLessEqual(gate.tiny_router_gemm_max_tokens,0)
        return gate,x,ids,table

    def verify(self,gate,x,ids,table,logits,weights,experts):
        expected_logits=x.cpu().double()@gate.weight.detach().cpu().double().T
        self.assertEqual(logits.dtype,torch.float32)
        torch.testing.assert_close(logits.cpu().double(),expected_logits,rtol=0,atol=0)
        selected=table[ids]
        values=torch.nn.functional.softplus(expected_logits.gather(1,selected.cpu().long())).sqrt()
        expected=values/(values.sum(1,keepdim=True)+1e-20)
        expected=torch.cat([expected,torch.full((24,1),.4,dtype=torch.float64)],1)
        torch.testing.assert_close(weights.cpu().double(),expected,rtol=3e-6,atol=2e-7)
        expected_ids=torch.cat([selected,torch.full((24,1),256,device='cuda',dtype=torch.int32)],1)
        torch.testing.assert_close(experts,expected_ids,rtol=0,atol=0)

    def test_native_gate_to_public_hash_jit_auto_and_sgl(self):
        from sglang.srt.models import deepseek_v2 as N
        from sglang.kernels.ops.moe.dsv4 import hash_topk
        from hcu_sglang.models.deepseek.current_v4.hcu_backend import bf16_fp32 as O
        for mode in ['auto','sgl']:
            with self.subTest(mode=mode):
                gate,x,ids,table=self.data()
                with patch.dict(os.environ,{'SGLANG_OPT_BF16_FP32_GEMM_ALGO':mode}),patch.object(O,'_linear_bf16_fp32_algo',mode),patch.object(N,'get_exec',return_value=S(deterministic=S(enable_deterministic_inference=False))):
                    logits=gate(x);weights,experts=hash_topk(logits,ids,table,1,2.5)
                    torch.cuda.synchronize();self.verify(gate,x,ids,table,logits,weights,experts)

    def test_graph_replays_native_gate_to_hash_with_changed_input_and_ids(self):
        from sglang.srt.models import deepseek_v2 as N
        from sglang.kernels.ops.moe.dsv4 import hash_topk
        from hcu_sglang.models.deepseek.current_v4.hcu_backend import bf16_fp32 as O
        gate,x,ids,table=self.data()
        with patch.dict(os.environ,{'SGLANG_OPT_BF16_FP32_GEMM_ALGO':'auto'}),patch.object(O,'_linear_bf16_fp32_algo','auto'),patch.object(N,'get_exec',return_value=S(deterministic=S(enable_deterministic_inference=False))):
            hash_topk(gate(x),ids,table,1,2.5);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                logits=gate(x);weights,experts=hash_topk(logits,ids,table,1,2.5)
            x.mul_(-1);ids.copy_((ids+3)%17);graph.replay();torch.cuda.synchronize()
            self.verify(gate,x,ids,table,logits,weights,experts)


if __name__=='__main__':unittest.main()
