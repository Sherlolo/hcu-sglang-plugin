from __future__ import annotations
import os
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class RoutingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def test_shared_config_output_checker_and_format_types_keep_native_identity(self):
        from hcu_sglang.layers.moe import topk_source as T
        from sglang.srt.layers.moe import topk as N
        for name in ['TopKConfig','TopKOutputChecker','TopKOutputFormat','TopKOutput','StandardTopKOutput','StandardTopKOutputPacked','TritonKernelTopKOutput','BypassedTopKOutput','PackedTopKOutput']:
            self.assertIs(getattr(T,name),getattr(N,name))

    def test_both_native_and_owned_topk_constructors_keep_native_config_identity(self):
        from hcu_sglang.layers.moe import topk_source as T
        from sglang.srt.layers.moe import topk as N
        with patch.object(T,'get_exec',return_value=S(moe=S(enable_waterfill=False,ep_num_redundant_experts=0))):
            for cls in [N.TopK,T.TopK]:
                obj=cls(4,layer_id=3,num_fused_shared_experts=1)
                self.assertIs(type(obj.topk_config),N.TopKConfig)
                self.assertEqual(obj.layer_id,3);self.assertEqual(obj.topk_config.top_k,4)
                self.assertFalse(obj.enable_waterfill)

    def test_synthetic_routing_rejects_redundant_placement_and_waterfill(self):
        from hcu_sglang.layers.moe import topk_source as T,hash_topk_source as H
        from sglang.srt.layers.moe import topk as N,hash_topk as NH
        for redundant,waterfill in [(1,False),(0,True)]:
            config=S(moe=S(enable_waterfill=waterfill,ep_num_redundant_experts=redundant))
            with patch.object(T,'get_exec',return_value=config),patch.object(H,'get_exec',return_value=config),patch.object(T,'_simulated_expert_balance',True),patch.dict(os.environ,{'SGLANG_SIMULATED_EXPERT_BALANCE':'1'}):
                for cls in [N.TopK,T.TopK]:
                    with self.assertRaises(ValueError):cls(4,layer_id=3,num_fused_shared_experts=1)
                for cls in [NH.HashTopK,H.HashTopK]:
                    with self.assertRaises(ValueError):cls(4,16,1,17)

    def test_synthetic_mode_rejects_eplb_before_routing_compute(self):
        from hcu_sglang.layers.moe import topk_source as T
        from sglang.srt.layers.moe import topk as N
        import torch
        cfg=N.TopKConfig(top_k=4)
        with patch.object(T,'_simulated_expert_balance',True):
            with self.assertRaisesRegex(ValueError,'contiguous expert'):
                N.select_experts(torch.empty(3,8),torch.empty(3,16),cfg,expert_location_dispatch_info=object())


    def test_fused_gate_uses_actual_remapping_predicate_and_rejects_eplb(self):
        import torch
        from hcu_sglang.layers.moe import topk_source as T
        from sglang.srt.layers.moe import utils as U
        config=S(moe=S(enable_eplb=False,init_expert_location='trivial',ep_num_redundant_experts=0))
        with patch.object(T,'get_server_args',return_value=object()),patch.object(T,'get_exec',return_value=config),patch.object(T,'get_moe_a2a_backend',return_value=U.MoeA2ABackend.DEEPEP),patch.object(T,'get_deepep_mode',return_value=U.DeepEPMode.LOW_LATENCY):
            self.assertTrue(T._can_use_lightop_moe_gate_pad_int64_fusion(torch.empty(3,384),6,0,False))
            config.moe.enable_eplb=True
            self.assertFalse(T._can_use_lightop_moe_gate_pad_int64_fusion(torch.empty(3,384),6,0,False))

    def test_non_hcu_around_preserves_effective_prior_callable(self):
        from hcu_sglang.layers.moe import routing as R
        token=object();seen=[]
        def implementation(*a,**k):raise AssertionError('HCU implementation must not execute')
        def original(*a,**k):seen.append((a,k));return token
        with patch.object(R,'is_hcu',return_value=False):self.assertIs(R._around(implementation)(original,token,value=7),token)
        self.assertEqual(seen,[((token,),{'value':7})])


if __name__=='__main__':unittest.main()
