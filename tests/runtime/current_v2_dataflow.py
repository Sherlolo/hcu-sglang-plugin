from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.layers.quantization import slimquant_w4a8 as Q
from hcu_sglang.models.deepseek.deepseek_v2.methods import attention as A, moe_graph as G

class CurrentV2DataflowTests(unittest.TestCase):
    def setUp(self):
        self.method=object.__new__(Q.SlimQuantW4A8Int8LinearMethod)
        self.method.w8a8_strategy=3
        self.layer=SimpleNamespace(weight=torch.ones(16,8,dtype=torch.int8),weight_scale=torch.full((8,1),.5))
        self.x=torch.zeros(2,16,dtype=torch.bfloat16)
        self.q=torch.ones(2,16,dtype=torch.int8)
        self.scale=torch.tensor([[.25],[.5]])

    def test_explicit_quantized_input_survives_projection_and_dense_gemm(self):
        seen=[]
        def gemm(q,w,*,scale_a,scale_b,out_dtype,bias):
            self.assertIs(q,self.q);self.assertIs(scale_a,self.scale)
            seen.append(q)
            return ((q.float()*scale_a)@(w.float()*scale_b.T)).to(out_dtype)
        def projection(x,*,input_quant_args):
            self.assertIs(x,self.x)
            return self.method.apply(self.layer,x,input_quant_args=input_quant_args),None
        carrier=(self.q,self.scale)
        attn=SimpleNamespace(q_lora_rank=16,fused_qkv_a_proj_with_mqa=projection)
        with patch.object(Q,'_use_fused_rms_quant',False,create=True),patch.object(Q,'per_token_quant_int8',side_effect=AssertionError('explicit input was discarded')),patch.object(Q.quant_ops,'blaslt_scaled_mm',side_effect=gemm):
            out=A.prepare_qkv_latent(attn,self.x,None,input_quant_args=carrier)
        self.assertTrue(Q.SlimQuantW4A8Int8LinearMethod.supports_prequantized_input)
        self.assertEqual(len(seen),1)
        self.assertTrue(torch.equal(out,torch.tensor([[2.]*8,[4.]*8],dtype=self.x.dtype)))

    def test_invalid_quantized_carriers_are_rejected(self):
        cases=[(self.q.float(),self.scale),(self.q,self.scale.half()),(self.q[:1],self.scale),(self.q,self.scale[:,0]),(self.q.to('meta'),self.scale)]
        for q,scale in cases:
            with self.subTest(dtype=q.dtype,shape=q.shape,device=q.device),self.assertRaises(ValueError):
                self.method.apply(self.layer,self.x,input_quant_args=(q,scale))

    def test_graph_handoff_preserves_replicated_shared_without_double_add(self):
        routed=torch.full((2,8),2.);shared=torch.full((2,8),3.)
        pieces=G.MergedMoEOutput(routed=routed,shared=shared,shared_is_replicated=True,experts=None,routed_scaling_factor=1.)
        calls=[]
        def forward(x,*,leave_output_sum):
            self.assertIs(x,self.x);self.assertTrue(leave_output_sum);return pieces
        def scoped(**kwargs):calls.append(kwargs);return nullcontext()
        with patch.object(G.N0,'get_tc_piecewise_forward_context',return_value=SimpleNamespace(moe_fusions={3:SimpleNamespace(forward_normal_dual_stream=forward)})),patch.object(G.N0,'get_forward',return_value=SimpleNamespace(scoped=scoped)):
            out,unmerged_shared=G._run_dsv4_dual_stream_outputs(self.x,3,True,True)
            self.assertIs(out,routed);self.assertIs(unmerged_shared,shared)
            with self.assertRaises(AssertionError):G._run_dsv4_dual_stream_outputs(self.x,3,False,False)
        self.assertTrue(calls[0]['mlp_reduce_scatter'])
        self.assertFalse(calls[0]['defer_moe_finalize'])

class QwenFusionGateTests(unittest.TestCase):
    def test_das_default_and_explicit_gate(self):
        from hcu_sglang.models import fused_rotary as F
        for hcu,flag,expected in [(True,False,False),(True,True,True),(False,True,False)]:
            with patch.object(F,'is_dcu',return_value=hcu),patch.object(F,'_use_fused_qwen_bailing_rotary',flag):
                self.assertEqual(F.use_fused_rms_rotary(),expected)

    def test_native_fused_branch_keeps_priority(self):
        from hcu_sglang.models.qwen import qwen3_moe as M
        marker=object()
        owner=SimpleNamespace(use_fused_qk_norm_rope=True,use_fused_qk_norm_rope_cpu=False)
        qkv=torch.ones(1,8,dtype=torch.bfloat16)
        with patch.object(M,'use_fused_rms_rotary',return_value=True),patch.object(M,'get_original_attr',return_value=lambda *args:marker):
            self.assertIs(M.replacement_apply_qk_norm_rope(owner,qkv,None,None),marker)

if __name__=='__main__':unittest.main()

