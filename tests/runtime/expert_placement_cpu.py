from __future__ import annotations
import argparse
from collections import Counter
import os
import random
import subprocess
import sys
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.eplb import expert_location as N
BASE_MAP=[N._compute_logical_to_all_physical_map]
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class ExpertPlacementCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def config(self,algorithm):return S(moe=S(moe_a2a_backend='deepep',ep_dispatch_algorithm=algorithm,ep_join_mode='static'))

    def test_load_aware_lp_and_dynamic_preserve_complete_replica_candidates(self):
        physical=torch.tensor([[0,1,2,3,0,1,2,3]])
        expected=torch.tensor([[[0,4],[1,5],[2,6],[3,7]]])
        for algorithm in ['load_aware','lp','dynamic']:
            with patch.object(N,'get_exec',return_value=self.config(algorithm)),patch.object(N,'get_parallel',return_value=S(ep_size=4,nnodes=1)),patch.object(N,'_prefer_same_node_experts',return_value=True),patch.dict(os.environ,{'HCU_PLUGIN_EP_STATIC_DISPATCH_POLICY':'nearest'}):
                actual=N._compute_logical_to_all_physical_map(physical,4,4,0)
                torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                if algorithm=='load_aware':
                    before=BASE_MAP[0](physical,4,4,0)
                    self.assertEqual(int((before[0,0]>=0).sum()),1)
                    self.assertEqual(int((actual[0,0]>=0).sum()),2)


    def test_native_metadata_factory_keeps_num_valid_copies_for_load_aware(self):
        physical=torch.tensor([[0,1,2,3,0,1,2,3]])
        common={'num_physical_experts':8,'base_num_physical_experts':8,'ep_size':4,'model_config_for_expert_location':S(num_logical_experts=4,num_layers=1)}
        with patch.object(N,'get_exec',return_value=self.config('load_aware')),patch.object(N,'get_device',return_value=S(device='cpu')),patch.object(N.ExpertLocationMetadata,'_init_common',return_value=common),patch.dict(os.environ,{'HCU_PLUGIN_EP_STATIC_DISPATCH_POLICY':'nearest'}):
            metadata=N.ExpertLocationMetadata.init_by_mapping(object(),physical,moe_ep_rank=0)
        self.assertIs(type(metadata),N.ExpertLocationMetadata)
        torch.testing.assert_close(metadata.logical_to_all_physical_map_num_valid,torch.full((1,4),2),rtol=0,atol=0)
        self.assertEqual(tuple(metadata.logical_to_all_physical_map.shape),(1,4,8))
        self.assertIsNone(metadata.logical_to_rank_dispatch_physical_map)


    def test_native_metadata_to_real_solver_retains_replicated_expert_count(self):
        from hcu_sglang.eplb.lplb_solver import LoadAwareReplicaSolver
        physical=torch.tensor([[0,1,2,3,0,1,2,3]])
        common={'num_physical_experts':8,'base_num_physical_experts':8,'ep_size':4,'model_config_for_expert_location':S(num_logical_experts=4,num_layers=1)}
        with patch.object(N,'get_exec',return_value=self.config('load_aware')),patch.object(N,'get_device',return_value=S(device='cpu')),patch.object(N.ExpertLocationMetadata,'_init_common',return_value=common),patch.dict(os.environ,{'HCU_PLUGIN_EP_STATIC_DISPATCH_POLICY':'nearest'}):
            metadata=N.ExpertLocationMetadata.init_by_mapping(object(),physical,moe_ep_rank=0)
        solver=LoadAwareReplicaSolver(metadata.physical_to_logical_map[0],metadata.logical_to_all_physical_map[0],4,logical_to_all_physical_map_num_valid=metadata.logical_to_all_physical_map_num_valid[0])
        self.assertEqual(solver.num_replicated,4)
        torch.testing.assert_close(solver.num_valid_copies,torch.full((4,),2,dtype=torch.int32),rtol=0,atol=0)

    def test_static_nearest_matches_original_and_fair_keeps_candidates(self):
        physical=torch.tensor([[0,1,2,3,0,1,2,3]])
        with patch.object(N,'get_exec',return_value=self.config('static')),patch.object(N,'get_parallel',return_value=S(ep_size=4,nnodes=1)),patch.object(N,'_prefer_same_node_experts',return_value=True):
            with patch.dict(os.environ,{'HCU_PLUGIN_EP_STATIC_DISPATCH_POLICY':'nearest'}):
                torch.testing.assert_close(N._compute_logical_to_all_physical_map(physical,4,4,0),BASE_MAP[0](physical,4,4,0),rtol=0,atol=0)
            with patch.dict(os.environ,{'HCU_PLUGIN_EP_STATIC_DISPATCH_POLICY':'locality_fair'}):
                actual=N._compute_logical_to_all_physical_map(physical,4,4,0)
                self.assertTrue(bool(((actual>=0).sum(-1)==2).all()))

    def test_fair_assignment_prioritizes_local_gpu_then_node(self):
        from hcu_sglang.eplb.placement import _assign_locality_fair_experts as assign
        candidates=[0,4,12]
        a=assign(candidates,8,2,2,4,random.Random(17))
        b=assign(candidates,8,2,2,4,random.Random(17));self.assertEqual(a,b)
        for rank,choice in enumerate(a):
            cost=lambda expert:0 if expert//2==rank else (1 if expert//4==rank//2 else 2)
            self.assertEqual(cost(choice),min(map(cost,candidates)))
        counts=Counter(a);self.assertLessEqual(max(counts.values())-min(counts.values()),1)

    def test_fair_assignment_rejects_invalid_topology_and_expert_ids(self):
        from hcu_sglang.eplb.placement import _assign_locality_fair_experts as assign
        for candidates,ep,local,gpn,node in [([],8,2,None,None),([0],0,2,None,None),([0],8,0,None,None),([0],8,2,2,None),([0],8,2,3,6),([0],8,2,2,3),([-1],8,2,None,None),([16],8,2,None,None)]:
            with self.assertRaises(ValueError):assign(candidates,ep,local,gpn,node,random.Random(0))

    def test_cli_sidecar_updates_process_policy_before_unchanged_seed_constructor(self):
        from sglang.srt.server_args import ServerArgs
        from hcu_sglang.config import ep_static_dispatch_policy
        parser=argparse.ArgumentParser();ServerArgs.add_cli_args(parser)
        args=parser.parse_args(['--model-path','fixture-model','--ep-static-dispatch-policy','locality_fair'])
        with patch.dict(os.environ,{'HCU_PLUGIN_EP_STATIC_DISPATCH_POLICY':'nearest'}),patch.object(ServerArgs,'__post_init__',lambda self:None):
            seed=ServerArgs.from_cli_args(args)
            self.assertEqual(ep_static_dispatch_policy(),'locality_fair')
            result=subprocess.check_output([sys.executable,'-c','from hcu_sglang.config import ep_static_dispatch_policy; print(ep_static_dispatch_policy())'],text=True).strip()
            self.assertEqual(result,'locality_fair')
            self.assertFalse(hasattr(seed,'hcu_plugin_ep_static_dispatch_policy'))

    def test_fair_policy_requires_static_and_non_hcu_keeps_original(self):
        from hcu_sglang.eplb import placement as P
        with patch.dict(os.environ,{'HCU_PLUGIN_EP_STATIC_DISPATCH_POLICY':'locality_fair'}),patch.object(P.P,'resolving_view',return_value=S(ep_dispatch_algorithm='load_aware')):
            with self.assertRaises(ValueError):P.around_validation(lambda _:7,object())
        token=object()
        with patch.object(P,'is_hcu',return_value=False):self.assertIs(P._around(lambda *a:None)(lambda *a:token,object()),token)


if __name__=='__main__':unittest.main()
