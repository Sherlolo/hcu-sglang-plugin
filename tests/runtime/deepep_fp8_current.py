from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.moe import deepep_fp8 as H
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8_moe import CompressedTensorsW8A8Fp8MoE

class DeepEPFp8Tests(unittest.TestCase):
    def test_packed_aliases_keep_shape_and_do_not_repeat_packing(self):
        H.register_hooks();scheme=object.__new__(CompressedTensorsW8A8Fp8MoE);layer=torch.nn.Module();layer.w13_weight=torch.nn.Parameter(torch.ones(2,16,128),requires_grad=False);layer.w2_weight=torch.nn.Parameter(torch.ones(2,128,8),requires_grad=False)
        with patch.object(H,'get_deepgemm_marlin_weight_pack_fn',return_value=lambda w:w.clone()*2) as packer:
            H._prepare_dsv4_channel_fp8_deepgemm_weights(scheme,layer);H._prepare_dsv4_channel_fp8_deepgemm_weights(scheme,layer)
        self.assertEqual(packer.call_count,1);self.assertFalse(hasattr(layer,'w13_weight'));self.assertFalse(hasattr(layer,'w2_weight'));self.assertEqual(layer._dsv4_w13_weight_shape,(2,16,128));self.assertEqual(layer._dsv4_w2_weight_shape,(2,128,8));self.assertIn('w13_weight_deepgemm',layer._non_persistent_buffers_set);self.assertEqual(layer.w13_weight_deepgemm.max().item(),2.)

    def test_masked_two_gemm_quant_dataflow_keeps_scales_and_clamp(self):
        import deepgemm
        q=torch.ones(2,3,128);s=torch.ones(2,3,1);ids=torch.zeros(1,2,dtype=torch.int64);weights=torch.full((1,2),.5);masked=torch.tensor([2,1],dtype=torch.int32)
        out=H.D.DeepEPLLDispatchOutput(q,s,ids,weights,masked,2);layer=SimpleNamespace(moe_runner_config=SimpleNamespace(activation='silu',swiglu_limit=3.),w13_weight_deepgemm=torch.ones(2,16,128),w2_weight_deepgemm=torch.ones(2,128,8),w13_weight_scale=torch.ones(2,16,1),w2_weight_scale=torch.ones(2,128,1));calls=[];a2=torch.ones(2,3,8);s2=torch.ones(2,3,1)
        def gemm(a,b,result,m,expected,**kw):
            self.assertIs(m,masked);self.assertEqual(expected,2);calls.append((a,b));result.fill_(len(calls))
        def quant(*,input,fp8type,tokens_per_expert,limit):
            self.assertEqual(input.shape,(2,3,16));self.assertIs(tokens_per_expert,masked);self.assertEqual(limit,3.);return a2,s2
        with patch.object(deepgemm,'m_grouped_fp8_gemm_nt_masked',side_effect=gemm),patch.object(H,'fuse_silu_mul_fp8_quant_ep',side_effect=quant):result=H.forward_groupgemm_w8a8_fp8_masked(layer,out)
        self.assertEqual(len(calls),2);self.assertIs(calls[0][0][0],q);self.assertIs(calls[0][0][1],s);self.assertIs(calls[1][0][0],a2);self.assertIs(calls[1][0][1],s2);self.assertEqual(result.shape,(2,3,128));self.assertEqual(result.max().item(),2.)

    def test_combine_packet_retains_router_ids_and_weights(self):
        q=torch.ones(2,3,8);s=torch.ones(2,3,1);ids=torch.zeros(1,2,dtype=torch.int64);weights=torch.full((1,2),.5);out=H.D.DeepEPLLDispatchOutput(q,s,ids,weights,torch.ones(2,dtype=torch.int32),2);value=torch.ones_like(q,dtype=torch.bfloat16)
        with patch.object(H,'_use_hcu_legacy_channel_fp8_deepep',return_value=True),patch.object(H,'forward_groupgemm_w8a8_fp8_masked',return_value=value):r=H.around_run_moe_core(lambda *a:None,object(),out)
        self.assertIsInstance(r,H.D.DeepEPLLCombineInput);self.assertIs(r.hidden_states,value);self.assertIs(r.topk_ids,ids);self.assertIs(r.topk_weights,weights)

if __name__=='__main__':unittest.main()
