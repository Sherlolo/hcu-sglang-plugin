from __future__ import annotations
from contextlib import ExitStack
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class RoutingGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def fixture(self,deep=False):
        from hcu_sglang.layers.moe import topk_source as T,hash_topk_source as H
        from sglang.srt.layers.moe import utils as U
        stack=ExitStack();exec_config=S(moe=S(enable_waterfill=False,ep_num_redundant_experts=0,enable_eplb=False,init_expert_location='trivial'))
        for module in [T,H]:stack.enter_context(patch.object(module,'get_exec',return_value=exec_config))
        stack.enter_context(patch.object(T,'get_parallel',return_value=S(moe_ep_size=1,moe_ep_rank=0,tp_group=None)))
        stack.enter_context(patch.object(H,'get_parallel',return_value=S(moe_ep_size=1,moe_ep_rank=0,tp_group=None)))
        stack.enter_context(patch.object(T,'get_moe_a2a_backend',return_value=U.MoeA2ABackend.DEEPEP if deep else U.MoeA2ABackend.NONE))
        stack.enter_context(patch.object(U,'get_moe_a2a_backend',return_value=U.MoeA2ABackend.DEEPEP if deep else U.MoeA2ABackend.NONE))
        stack.enter_context(patch.object(T,'get_moe_runner_backend',return_value=U.MoeRunnerBackend.DEEP_GEMM if deep else U.MoeRunnerBackend.TRITON))
        stack.enter_context(patch.object(T,'get_deepep_mode',return_value=U.DeepEPMode.LOW_LATENCY))
        stack.enter_context(patch.object(T,'is_allocation_symmetric',return_value=False))
        recorder=S(on_select_experts=lambda **kwargs:None)
        for module in [T,H]:stack.enter_context(patch.object(module,'get_global_expert_distribution_recorder',return_value=recorder))
        return stack

    def data(self,e=256):
        logits=torch.stack([torch.linspace(-2,2,e,device='cuda').roll(17*r) for r in range(9)])
        hidden=torch.ones(9,128,device='cuda',dtype=torch.bfloat16)
        bias=torch.linspace(-.1,.1,e,device='cuda')
        return hidden,logits,bias

    def verify_softplus(self,logits,bias,result,k,valid=9):
        ids=result.topk_ids[:valid].long();weights=result.topk_weights[:valid]
        scores=torch.nn.functional.softplus(logits[:valid].cpu().double()).sqrt()
        expected_ids=(scores+bias.cpu().double()).topk(k,dim=1).indices
        torch.testing.assert_close(ids.cpu().sort(1).values,expected_ids.sort(1).values,rtol=0,atol=0)
        expected=scores.gather(1,ids.cpu());expected/=expected.sum(1,keepdim=True)+1e-20
        torch.testing.assert_close(weights.cpu().double(),expected,rtol=5e-6,atol=2e-7)

    def test_public_softmax_uses_real_lightop_and_preserves_scores(self):
        from sglang.srt.layers.moe import topk as N
        from hcu_sglang.layers.moe import topk_source as T
        from lightop import op
        self.assertTrue(T._use_fused_topk_softmax)
        hidden,logits,_=self.data()
        with patch.object(op,'topk_softmax',wraps=op.topk_softmax) as actual:
            weights,ids=N.fused_topk(hidden,logits,8,True);torch.cuda.synchronize();actual.assert_called_once()
        values=logits.cpu().double().softmax(1);expected_ids=values.topk(8,dim=1).indices
        torch.testing.assert_close(ids.cpu().long().sort(1).values,expected_ids.sort(1).values,rtol=0,atol=0)
        expected=values.gather(1,ids.cpu().long());expected/=expected.sum(1,keepdim=True)
        torch.testing.assert_close(weights.cpu().double(),expected,rtol=5e-6,atol=2e-7)

    def test_native_topk_selects_real_sqrtsoftplus_lightop_with_native_output_type(self):
        from sglang.srt.layers.moe import topk as N
        from hcu_sglang.layers.moe import topk_source as T
        hidden,logits,bias=self.data()
        with self.fixture(),patch.object(T.op,'moe_fused_gate_sqrtsoftplus',wraps=T.op.moe_fused_gate_sqrtsoftplus) as actual:
            obj=N.TopK(8,layer_id=2,scoring_func='sqrtsoftplus',correction_bias=bias,routed_scaling_factor=2.5,output_format=N.TopKOutputFormat.STANDARD,allow_routed_experts_capture=False)
            result=obj(hidden,logits);torch.cuda.synchronize();actual.assert_called_once()
        self.assertIs(type(result),N.StandardTopKOutput);self.verify_softplus(logits,bias,result,8)

    def test_fused_v41_gate_masks_padding_and_emits_int64_for_deepep(self):
        from sglang.srt.layers.moe import topk as N
        from hcu_sglang.layers.moe import topk_source as T
        hidden,logits,bias=self.data(384);count=torch.tensor(5,device='cuda',dtype=torch.int32)
        cfg=N.TopKConfig(top_k=6,scoring_func='sqrtsoftplus',correction_bias=bias,routed_scaling_factor=2.5,sqrtsoftplus_log1p=True,allow_routed_experts_capture=False)
        self.assertTrue(T._use_lightop_moe_gate_pad_int64_fusion)
        with self.fixture(deep=True),patch.object(T.op,'moe_fused_gate_sqrtsoftplus_v41_hcu_fused',wraps=T.op.moe_fused_gate_sqrtsoftplus_v41_hcu_fused) as actual:
            result=N.select_experts(hidden,logits,cfg,layer_id=2,num_token_non_padded=count);torch.cuda.synchronize();actual.assert_called_once()
        self.assertEqual(result.topk_ids.dtype,torch.int64);self.verify_softplus(logits,bias,result,6,5)
        torch.testing.assert_close(result.topk_ids[5:],torch.full_like(result.topk_ids[5:],-1),rtol=0,atol=0)

    def test_graph_replays_v41_gate_with_changed_logits_and_valid_row_count(self):
        from sglang.srt.layers.moe import topk as N
        hidden,logits,bias=self.data(384);count=torch.tensor(5,device='cuda',dtype=torch.int32)
        cfg=N.TopKConfig(top_k=6,scoring_func='sqrtsoftplus',correction_bias=bias,routed_scaling_factor=2.5,sqrtsoftplus_log1p=True,allow_routed_experts_capture=False)
        with self.fixture(deep=True):
            N.select_experts(hidden,logits,cfg,layer_id=2,num_token_non_padded=count);torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):result=N.select_experts(hidden,logits,cfg,layer_id=2,num_token_non_padded=count)
            logits.copy_(logits.roll(37,1));count.fill_(7);graph.replay();torch.cuda.synchronize()
            self.verify_softplus(logits,bias,result,6,7)
            torch.testing.assert_close(result.topk_ids[7:],torch.full_like(result.topk_ids[7:],-1),rtol=0,atol=0)

    def test_native_and_owned_hash_classes_keep_shared_weights_and_padding(self):
        from sglang.srt.layers.moe import hash_topk as N,topk as K
        from hcu_sglang.layers.moe import hash_topk_source as H
        hidden,logits,_=self.data();input_ids=torch.arange(9,device='cuda',dtype=torch.int64)
        count=torch.tensor(5,device='cuda',dtype=torch.int32)
        for cls in [N.HashTopK,H.HashTopK]:
            with self.fixture(),torch.device('cuda'):
                obj=cls(9,256,1,17,routed_scaling_factor=2.5,layer_id=2)
                obj.tid2eid.copy_((torch.arange(17*8,device='cuda').reshape(17,8)*7%256).int())
                result=obj(hidden,logits,input_ids,num_token_non_padded=count);torch.cuda.synchronize()
            self.assertIs(type(result),K.StandardTopKOutput)
            torch.testing.assert_close(result.topk_ids[:5,:8],obj.tid2eid[input_ids[:5]],rtol=0,atol=0)
            torch.testing.assert_close(result.topk_weights[:5,-1],torch.full_like(result.topk_weights[:5,-1],.4),rtol=1e-6,atol=0)
            torch.testing.assert_close(result.topk_weights[5:],torch.zeros_like(result.topk_weights[5:]),rtol=0,atol=0)


    def synthetic_ids(self):
        result=[]
        for row in range(9):
            result.append([((row*8+slot+3)%8)*32+(row*8+slot+2)%32 for slot in range(8)])
        return torch.tensor(result,device='cuda',dtype=torch.int32)

    def test_synthetic_topk_override_preserves_router_weights(self):
        from sglang.srt.layers.moe import topk as N
        from hcu_sglang.layers.moe import topk_source as T
        hidden,logits,bias=self.data()
        cfg=N.TopKConfig(top_k=8,scoring_func='sqrtsoftplus',correction_bias=bias,routed_scaling_factor=2.5,allow_routed_experts_capture=False)
        with self.fixture(),patch.object(T,'get_parallel',return_value=S(moe_ep_size=8,moe_ep_rank=3,tp_group=None)):
            before=N.select_experts(hidden,logits,cfg,layer_id=2)
            with patch.object(T,'_simulated_expert_balance',True):after=N.select_experts(hidden,logits,cfg,layer_id=2)
            torch.cuda.synchronize()
        torch.testing.assert_close(after.topk_ids,self.synthetic_ids(),rtol=0,atol=0)
        torch.testing.assert_close(after.topk_weights,before.topk_weights,rtol=0,atol=0)

    def test_synthetic_hash_override_preserves_shared_column_and_weights(self):
        import os
        from sglang.srt.layers.moe import hash_topk as N
        from hcu_sglang.layers.moe import hash_topk_source as H
        hidden,logits,_=self.data();input_ids=torch.arange(9,device='cuda',dtype=torch.int64)
        with self.fixture(),patch.object(H,'get_parallel',return_value=S(moe_ep_size=8,moe_ep_rank=3,tp_group=None)),torch.device('cuda'):
            obj=N.HashTopK(9,256,1,17,routed_scaling_factor=2.5,layer_id=2)
            obj.tid2eid.copy_((torch.arange(17*8,device='cuda').reshape(17,8)*7%256).int())
            before=obj(hidden,logits,input_ids)
            with patch.dict(os.environ,{'SGLANG_SIMULATED_EXPERT_BALANCE':'1'}):after=obj(hidden,logits,input_ids)
            torch.cuda.synchronize()
        torch.testing.assert_close(after.topk_ids[:,:8],self.synthetic_ids(),rtol=0,atol=0)
        torch.testing.assert_close(after.topk_ids[:,-1],before.topk_ids[:,-1],rtol=0,atol=0)
        torch.testing.assert_close(after.topk_weights,before.topk_weights,rtol=0,atol=0)


    def test_deepep_shared_slot_remap_occurs_once_before_padded_route_mask(self):
        from sglang.srt.layers.moe import topk as N
        from hcu_sglang.layers.moe import topk_source as T
        hidden,logits,bias=self.data();count=torch.tensor(5,device='cuda',dtype=torch.int32)
        cfg=N.TopKConfig(top_k=9,num_fused_shared_experts=1,scoring_func='sqrtsoftplus',correction_bias=bias,routed_scaling_factor=2.5,allow_routed_experts_capture=False)
        with self.fixture(deep=True),patch.object(T,'get_parallel',return_value=S(moe_ep_size=8,moe_ep_rank=3,tp_group=None)):
            result=N.select_experts(hidden,logits,cfg,layer_id=2,num_token_non_padded=count);torch.cuda.synchronize()
        scores=torch.nn.functional.softplus(logits[:5].cpu().double()).sqrt()
        routed=(scores+bias.cpu().double()).topk(8,dim=1).indices
        expected=routed+routed//32
        torch.testing.assert_close(result.topk_ids[:5,:8].cpu().long().sort(1).values,expected.sort(1).values,rtol=0,atol=0)
        torch.testing.assert_close(result.topk_ids[:5,-1],torch.full_like(result.topk_ids[:5,-1],131),rtol=0,atol=0)
        torch.testing.assert_close(result.topk_weights[:5,-1],torch.full_like(result.topk_weights[:5,-1],.4),rtol=1e-6,atol=0)
        torch.testing.assert_close(result.topk_ids[5:],torch.full_like(result.topk_ids[5:],-1),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
