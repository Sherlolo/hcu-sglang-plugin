from __future__ import annotations
from contextlib import ExitStack
import os
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class ExpertPlacementGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def test_real_native_metadata_factory_to_solver_and_public_dispatch(self):
        from sglang.srt.eplb import expert_location as N,lplb_solver as NL,expert_location_dispatch as ND
        from sglang.srt import runtime_context as R
        from sglang.srt.model_executor.model_runner_components import moe_ep_setup as F
        from hcu_sglang.eplb import integration as I,lplb_solver as L,expert_location_dispatch as D
        cfg=S(moe=S(moe_a2a_backend='deepep',ep_dispatch_algorithm='load_aware',ep_join_mode='static'))
        parallel=S(moe_ep_size=4,moe_ep_rank=0,moe_ep_group=None,ep_size=4,nnodes=1)
        common={'num_physical_experts':8,'base_num_physical_experts':8,'ep_size':4,'model_config_for_expert_location':S(num_logical_experts=4,num_layers=1)}
        resources=S(lplb_solvers={});physical=torch.tensor([[0,1,2,3,0,1,2,3]],device='cuda')
        with ExitStack() as stack:
            for module,name,value in [(N,'get_exec',cfg),(N,'get_device',S(device='cuda')),(N,'get_parallel',parallel),(R,'get_exec',cfg),(R,'get_parallel',parallel),(L,'get_resources',resources),(NL,'get_resources',resources),(D,'get_exec',cfg),(D,'get_parallel',parallel)]:stack.enter_context(patch.object(module,name,return_value=value))
            stack.enter_context(patch.object(N.ExpertLocationMetadata,'_init_common',return_value=common))
            metadata=N.ExpertLocationMetadata.init_by_mapping(object(),physical,moe_ep_rank=0)
            torch.testing.assert_close(metadata.logical_to_all_physical_map_num_valid,torch.full((1,4),2,device='cuda',dtype=torch.int64),rtol=0,atol=0)
            stack.enter_context(patch.object(I,'get_global_expert_location_metadata',return_value=metadata));stack.enter_context(patch.object(D,'get_global_expert_location_metadata',return_value=metadata))
            F.init_lplb_solvers(model_config=S(hf_config=S(architectures=['MiniMaxM2ForCausalLM'])))
            solver=NL.get_global_lplb_solver(0);self.assertEqual(solver.num_replicated,4)
            info=ND.ExpertLocationDispatchInfo.init_new(0)
            ids=torch.tensor([[0,1],[2,3],[0,2]],device='cuda',dtype=torch.int32)
            count=torch.tensor(2,device='cuda',dtype=torch.int32)
            probabilities=solver.solve(ids,count)
            actual=ND.topk_ids_logical_to_physical(ids,info,probabilities);torch.cuda.synchronize()
            torch.testing.assert_close(physical[0,actual.long()],ids.long(),rtol=0,atol=0)
            torch.testing.assert_close(solver._local_counts,torch.ones(4,device='cuda'),rtol=0,atol=0)

    def test_gpu_metadata_keeps_all_candidates_for_locality_fair_static_policy(self):
        from sglang.srt.eplb import expert_location as N
        cfg=S(moe=S(moe_a2a_backend='deepep',ep_dispatch_algorithm='static',ep_join_mode='static'))
        parallel=S(ep_size=4,nnodes=1)
        common={'num_physical_experts':8,'base_num_physical_experts':8,'ep_size':4,'model_config_for_expert_location':S(num_logical_experts=4,num_layers=1)}
        physical=torch.tensor([[0,1,2,3,0,1,2,3]],device='cuda')
        with patch.object(N,'get_exec',return_value=cfg),patch.object(N,'get_device',return_value=S(device='cuda')),patch.object(N,'get_parallel',return_value=parallel),patch.object(N,'_prefer_same_node_experts',return_value=True),patch.object(N.ExpertLocationMetadata,'_init_common',return_value=common),patch.dict(os.environ,{'HCU_PLUGIN_EP_STATIC_DISPATCH_POLICY':'locality_fair'}):
            metadata=N.ExpertLocationMetadata.init_by_mapping(object(),physical,moe_ep_rank=0)
        torch.testing.assert_close(metadata.logical_to_all_physical_map_num_valid,torch.full((1,4),2,device='cuda',dtype=torch.int64),rtol=0,atol=0)
        self.assertEqual(metadata.logical_to_rank_dispatch_physical_map.device,physical.device)
        torch.testing.assert_close(metadata.logical_to_rank_dispatch_physical_map[0,:2],torch.tensor([0,1],device='cuda'),rtol=0,atol=0)
        self.assertTrue(bool((metadata.logical_to_rank_dispatch_physical_map>=0).all()))


if __name__=='__main__':unittest.main()
