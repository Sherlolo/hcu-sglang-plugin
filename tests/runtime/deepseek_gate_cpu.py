from __future__ import annotations
import os
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import register
from hcu_sglang.layers.moe import deepseek_gate as G
from sglang.srt.models import deepseek_v2 as N
from sglang.srt.plugins.hook_registry import HookRegistry


class DeepseekGateCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def gate(self,is_v4=True,model_type='deepseek_v4',**kwargs):
        config=S(model_type=model_type,n_routed_experts=256,hidden_size=128,topk_method='noaux_tc',router_fp32=False,**kwargs)
        dtype=torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with patch.object(N,'tiny_router_gemm_max_tokens',return_value=0),patch.object(N._hip_moe,'router_max_tokens',return_value=-1):
                gate=N.MoEGate(config,None,is_hash_moe=True,is_deepseek_v4=is_v4)
                with torch.no_grad():gate.weight.fill_(.125)
                return gate
        finally:torch.set_default_dtype(dtype)

    def test_native_constructor_uses_source_glm_router_dtype_policy(self):
        with patch.dict(os.environ,{'SGLANG_MOE_ROUTER_USE_CONFIG_DTYPE':'0'}):
            gate=self.gate(model_type='glm5_next',router_dtype='float32')
            self.assertEqual(gate.weight.dtype,torch.bfloat16)
        with patch.dict(os.environ,{'SGLANG_MOE_ROUTER_USE_CONFIG_DTYPE':'1'}):
            gate=self.gate(model_type='glm5_next',router_dtype='float32')
            self.assertEqual(gate.weight.dtype,torch.float32)
        self.assertTrue(gate.is_deepseek_v4)
        self.assertIsNone(gate.e_score_correction_bias)

    def test_dsv4_routes_fp32_producer_before_broad_aiter_in_auto(self):
        gate=self.gate();x=torch.randn(24,128,dtype=torch.float32)
        from sglang.kernels.ops.gemm import bf16_fp32
        sentinel=torch.empty(24,256,dtype=torch.float32)
        with patch.dict(os.environ,{'SGLANG_OPT_BF16_FP32_GEMM_ALGO':'auto'}),patch.object(N,'get_exec',return_value=S(deterministic=S(enable_deterministic_inference=False))),patch.object(N,'_use_aiter',True),patch.object(bf16_fp32,'linear_bf16_fp32',return_value=sentinel) as producer:
            self.assertIs(gate(x),sentinel)
        args=producer.call_args.args
        self.assertEqual(args[0].dtype,gate.weight.dtype)
        torch.testing.assert_close(args[0],x.to(gate.weight.dtype),rtol=0,atol=0)
        self.assertIs(args[1],gate.weight)

    def test_hcu_dsv4_produces_fp32_without_aiter(self):
        gate=self.gate();x=torch.randn(24,128,dtype=torch.bfloat16)
        from sglang.kernels.ops.gemm import bf16_fp32
        def actual(a,b):return torch.nn.functional.linear(a.float(),b.float())
        with patch.object(N,'get_exec',return_value=S(deterministic=S(enable_deterministic_inference=False))),patch.object(N,'_use_aiter',False),patch.object(bf16_fp32,'linear_bf16_fp32',side_effect=actual) as producer:
            y=gate(x)
        producer.assert_called_once();self.assertEqual(y.dtype,torch.float32)
        torch.testing.assert_close(y,actual(x,gate.weight),rtol=0,atol=0)

    def test_legacy_cp_branch_retains_weight_dtype(self):
        gate=self.gate(is_v4=False);x=torch.randn(24,128)
        with patch.object(N,'get_exec',return_value=S(deterministic=S(enable_deterministic_inference=False))),patch.object(G,'enable_cp_v2',return_value=False),patch.object(G,'dsa_use_prefill_cp',return_value=True):
            result=gate(x,forward_batch=object())
        self.assertEqual(result.dtype,torch.bfloat16)
        torch.testing.assert_close(result,torch.nn.functional.linear(x.bfloat16(),gate.weight),rtol=0,atol=0)

    def test_non_hcu_keeps_effective_original_for_both_hooks(self):
        token=object();receiver=object()
        for hook in [G.around_init,G.around_forward]:
            seen=[]
            def original(*a,**kw):seen.append((a,kw));return token
            with patch.object(G,'is_hcu',return_value=False):self.assertIs(hook(original,receiver,token,x=3),token)
            self.assertEqual(seen,[((receiver,token),{'x':3})])


if __name__=='__main__':unittest.main()
