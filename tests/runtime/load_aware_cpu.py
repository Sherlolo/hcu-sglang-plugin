from __future__ import annotations
import argparse
from contextlib import ExitStack
import os
import pickle
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class LoadAwareCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def metadata(self):
        physical=torch.tensor([[0,1,2,3,0,1,2,3],[0,1,2,3,0,1,2,3]])
        mapping=torch.tensor([[[0,4],[1,5],[2,6],[3,7]]]*2)
        return S(physical_to_logical_map=physical,physical_to_logical_map_cpu=physical,logical_to_all_physical_map=mapping,logical_to_all_physical_map_num_valid=torch.full((2,4),2),ep_size=2,num_layers=2)

    def test_public_factory_instantiates_actual_load_aware_solver_and_publishes_native_registry(self):
        from hcu_sglang.eplb import integration as I,lplb_solver as L
        from sglang.srt.eplb import lplb_solver as N
        from sglang.srt.model_executor.model_runner_components import moe_ep_setup
        from sglang.srt import runtime_context as R
        resources=S(lplb_solvers={});metadata=self.metadata();config=S(moe=S(ep_dispatch_algorithm='load_aware'))
        with patch.object(I,'get_global_expert_location_metadata',return_value=metadata),patch.object(R,'get_exec',return_value=config),patch.object(R,'get_parallel',return_value=S(moe_ep_group=None)),patch.object(L,'get_resources',return_value=resources),patch.object(N,'get_resources',return_value=resources),patch.dict(os.environ,{},clear=False):
            os.environ.pop('SGLANG_EXPERIMENTAL_LPLB_STATIC_PROBS',None)
            moe_ep_setup.init_lplb_solvers(model_config=S(hf_config=S(architectures=['MiniMaxM2ForCausalLM'])))
            self.assertEqual(set(resources.lplb_solvers),{0,1})
            for layer,solver in resources.lplb_solvers.items():
                self.assertIs(type(solver),L.LoadAwareReplicaSolver)
                self.assertIs(N.get_global_lplb_solver(layer),solver)
                torch.testing.assert_close(solver.copy_ranks,torch.tensor([[0,1]]*4,dtype=torch.int32),rtol=0,atol=0)

    def test_no_replica_solver_returns_identity_without_count_or_collective(self):
        from hcu_sglang.eplb.lplb_solver import LoadAwareReplicaSolver
        class Group:
            def all_reduce(self,*a):raise AssertionError('load-aware must not collect')
        solver=LoadAwareReplicaSolver(torch.arange(4),torch.arange(4).reshape(4,1),2,Group())
        expected=torch.ones(4,1)
        torch.testing.assert_close(solver.solve(torch.empty(0,3,dtype=torch.int32)),expected,rtol=0,atol=0)
        self.assertIsNone(solver._local_counts)

    def test_static_lp_solver_keeps_existing_probability_path(self):
        from hcu_sglang.eplb.lplb_solver import LPLBSolver
        mapping=torch.tensor([[0,2],[1,3]])
        probabilities=torch.tensor([[.25,.75],[.6,.4]])
        solver=LPLBSolver(torch.tensor([0,1,0,1]),mapping,2,static_probabilities=probabilities,static_max_copies=2)
        result=solver.solve(torch.tensor([[0,1]]))
        self.assertIs(result,solver._static_probabilities)
        self.assertEqual(result.data_ptr(),probabilities.data_ptr())
        torch.testing.assert_close(solver.static_dispatch_map,mapping.int(),rtol=0,atol=0)

    def test_cli_accepts_load_aware_and_preserves_seed_value_through_pickle(self):
        from sglang.srt.server_args import ServerArgs
        parser=argparse.ArgumentParser();ServerArgs.add_cli_args(parser)
        args=parser.parse_args(['--model-path','fixture-model','--ep-dispatch-algorithm','load_aware'])
        self.assertEqual(args.ep_dispatch_algorithm,'load_aware')
        with patch.object(ServerArgs,'__post_init__',lambda self:None):seed=ServerArgs.from_cli_args(args)
        clone=pickle.loads(pickle.dumps(seed))
        self.assertEqual(clone.ep_dispatch_algorithm,'load_aware')
        self.assertIs(type(clone),ServerArgs)

    def test_load_aware_validation_requires_transport_redundancy_and_hip(self):
        from hcu_sglang.eplb import integration as I
        for a2a,redundant,hip in [('none',2,True),('deepep',0,True),('deepep',2,False),('deepep',2,True)]:
            cfg=S(ep_dispatch_algorithm='load_aware',ep_num_redundant_experts=redundant)
            with patch.object(I.P,'resolving_view',return_value=cfg),patch.object(I.P,'resolved_view',return_value=S(moe_a2a_backend=a2a)),patch.object(I.P,'get_platform',return_value=S(is_hip=hip)):
                if a2a=='deepep' and redundant>0 and hip:self.assertEqual(I.around_dispatch_validation(lambda _:7,object()),7)
                else:
                    with self.assertRaises(ValueError):I.around_dispatch_validation(lambda _:7,object())


    def test_rebalance_reinitializes_load_aware_and_runner_skips_draft(self):
        from hcu_sglang.eplb import integration as I
        from sglang.srt.eplb import eplb_manager as N
        from sglang.srt.model_executor.model_runner import ModelRunner
        events=[]
        updater=S(update=lambda *a,**k:[])
        with patch.object(I.M,'get_parallel',return_value=S(nnodes=1)):
            N.update_expert_location_with_recovery(expert_location_updater=updater,model=S(routed_experts_weights_of_layer=object()),new_expert_location_metadata=object(),update_layer_ids=[0],tp_rank=0,expert_backup_client=None,update_weights_from_disk_callable=lambda **kw:None,ep_dispatch_algorithm='load_aware',init_lplb_solvers_callable=lambda:events.append('rebalance'))
        self.assertEqual(events,['rebalance'])
        runner=S(is_draft_worker=False,model_config=object())
        with patch.object(I,'get_exec',return_value=S(moe=S(ep_dispatch_algorithm='load_aware'))),patch.object(I,'init_lplb_solvers',side_effect=lambda **kw:events.append('init')):
            ModelRunner.maybe_init_lplb_solvers(runner)
            runner.is_draft_worker=True;ModelRunner.maybe_init_lplb_solvers(runner)
        self.assertEqual(events,['rebalance','init'])

    def test_native_dispatch_info_type_and_missing_probabilities_contract(self):
        from hcu_sglang.eplb import expert_location_dispatch as D
        from sglang.srt.eplb import expert_location_dispatch as N
        self.assertTrue(issubclass(D.ExpertLocationDispatchInfo,N.ExpertLocationDispatchInfo))
        info=N.ExpertLocationDispatchInfo('load_aware',None,torch.tensor([[0,1]]),torch.tensor([2]),2)
        with self.assertRaisesRegex(RuntimeError,'log2phy_prob is None'):N.topk_ids_logical_to_physical(torch.tensor([[0]]),info)


if __name__=='__main__':unittest.main()
