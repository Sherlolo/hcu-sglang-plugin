import unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.layers.attention.fla import chunk_delta_h_source as H, chunk_o_source as O, layernorm_gated_source as L
from sglang.kernels.ops.attention.fla import chunk_delta_h as OH, chunk_o as OO, layernorm_gated as OL

class FlaParityGpuTests(unittest.TestCase):
    def test_chunk_state_update_and_output_match_effective_native(self):
        torch.manual_seed(1)
        k=torch.randn(1,32,2,64,device='cuda',dtype=torch.bfloat16)*.02
        w=torch.randn_like(k)*.02;u=torch.randn_like(k)*.02
        state=torch.randn(1,2,64,64,device='cuda',dtype=torch.float32)*.02
        before=state.clone();left=state.clone();right=state.clone()
        with patch.object(H,'_USE_KDA_HCU_FROM_AITER',False),patch.object(H,'_USE_KDA_HCU_FROM_TRITON',False):
            actual=H.chunk_gated_delta_rule_fwd_h(k,w,u,initial_state=left,initial_state_indices=torch.tensor([0],device='cuda',dtype=torch.int32))
        expected=OH.chunk_gated_delta_rule_fwd_h(k,w,u,initial_state=right,initial_state_indices=torch.tensor([0],device='cuda',dtype=torch.int32))
        for a,b in zip(actual,expected):
            if a is not None:torch.testing.assert_close(a,b,rtol=.02,atol=.002)
        torch.testing.assert_close(left,right,rtol=.02,atol=.002)
        self.assertFalse(torch.equal(left,before))

    def test_boltops_chunk_state_and_output_match_native(self):
        if not hasattr(H,'_triton_fwd_h'):
            self.skipTest('SGLANG_KDA_USE_HCU_OP=1 is required')
        k=torch.randn(1,32,2,64,device='cuda',dtype=torch.bfloat16)*.02
        w=torch.randn_like(k)*.02;u=torch.randn_like(k)*.02
        state=torch.randn(1,2,64,64,device='cuda',dtype=torch.float32)*.02
        left=state.clone();right=state.clone();idx=torch.tensor([0],device='cuda',dtype=torch.int32)
        with patch.object(H,'_USE_KDA_HCU_FROM_AITER',False),patch.object(H,'_USE_KDA_HCU_FROM_TRITON',True):
            actual=H.chunk_gated_delta_rule_fwd_h(k,w,u,initial_state=left,initial_state_indices=idx)
        expected=OH.chunk_gated_delta_rule_fwd_h(k,w,u,initial_state=right,initial_state_indices=idx)
        for a,b in zip(actual,expected):
            if a is not None:torch.testing.assert_close(a,b,rtol=.02,atol=.002)
        torch.testing.assert_close(left,right,rtol=.02,atol=.002)

    def test_chunk_output_matches_native_when_aiter_disabled(self):
        q=torch.randn(1,32,2,64,device='cuda',dtype=torch.bfloat16)*.02
        k=torch.randn_like(q);v=torch.randn_like(q)
        h=torch.randn(1,1,2,64,64,device='cuda',dtype=torch.bfloat16)
        with patch.object(O,'_use_prefill_aiter_linear_attn',False):actual=O.chunk_fwd_o(q,k,v,h)
        expected=OO.chunk_fwd_o(q,k,v,h)
        torch.testing.assert_close(actual,expected,rtol=.02,atol=.002)

    def test_gated_norm_native_matches_reference_without_lightop_opt_in(self):
        x=torch.randn(8,128,device='cuda',dtype=torch.bfloat16)
        weight=torch.randn(128,device='cuda',dtype=torch.bfloat16)
        z=torch.randn_like(x)
        with patch.object(L,'_use_prefill_layer_norm_fwd',False):
            actual=L._layer_norm_fwd(x,weight,None,1e-6,z=z,is_rms_norm=True)
        expected=OL._layer_norm_fwd(x,weight,None,1e-6,z=z,is_rms_norm=True)
        for a,b in zip(actual,expected):
            if a is not None:torch.testing.assert_close(a,b,rtol=.02,atol=.002)

if __name__=='__main__':unittest.main()
