from __future__ import annotations
import unittest
import torch
from hcu_sglang.eplb import load_aware_dispatch as D
from hcu_sglang.eplb.lplb_solver import LoadAwareReplicaSolver


class LoadAwareGpuTests(unittest.TestCase):
    def test_count_masks_padded_rows_and_reuses_buffer(self):
        ids=torch.tensor([[0,1],[2,2],[3,3]],device='cuda',dtype=torch.int32)
        counts=torch.full((4,),99.,device='cuda')
        result=D.count_logical_experts(ids,counts,torch.tensor(2,device='cuda',dtype=torch.int32));torch.cuda.synchronize()
        self.assertIs(result,counts)
        torch.testing.assert_close(counts,torch.tensor([1.,1.,2.,0.],device='cuda'),rtol=0,atol=0)
        ids.fill_(1);D.count_logical_experts(ids,counts);torch.cuda.synchronize()
        torch.testing.assert_close(counts,torch.tensor([0.,6.,0.,0.],device='cuda'),rtol=0,atol=0)

    def test_count_ignores_drop_and_shared_ids_without_touching_adjacent_storage(self):
        storage=torch.full((6,),123.,device='cuda');counts=storage[1:5]
        ids=torch.tensor([[0,-1],[4,2]],device='cuda',dtype=torch.int32)
        D.count_logical_experts(ids,counts);torch.cuda.synchronize()
        torch.testing.assert_close(storage,torch.tensor([123.,1.,0.,1.,0.,123.],device='cuda'),rtol=0,atol=0)

    def test_count_strided_routed_columns_excludes_shared_column(self):
        storage=torch.tensor([[0,1,3],[2,1,3]],device='cuda',dtype=torch.int64)
        ids=storage[:,:2];self.assertFalse(ids.is_contiguous())
        counts=torch.empty(4,device='cuda');D.count_logical_experts(ids,counts);torch.cuda.synchronize()
        torch.testing.assert_close(counts,torch.tensor([1.,2.,1.,0.],device='cuda'),rtol=0,atol=0)

    def solver(self):
        physical=torch.tensor([0,1,2,3,0,1,2,3],device='cuda')
        mapping=torch.tensor([[0,4],[1,5],[2,6],[3,7]],device='cuda')
        class Group:
            def all_reduce(self,*a):raise AssertionError('Source load-aware is local per rank')
        return LoadAwareReplicaSolver(physical,mapping,2,Group()),mapping

    def test_real_solver_matches_cpu_waterfill_and_probability_invariants(self):
        solver,mapping=self.solver()
        ids=torch.tensor([[0,0],[0,1],[2,3],[0,1],[2,2]],device='cuda',dtype=torch.int32)
        actual=solver.solve(ids);torch.cuda.synchronize()
        counts=torch.bincount(ids.flatten().long(),minlength=4).float()
        expected=D.build_load_aware_probabilities_cpu_reference(counts,mapping,2)
        torch.testing.assert_close(actual.cpu(),expected,rtol=2e-4,atol=2e-5)
        torch.testing.assert_close(actual.sum(1),torch.ones(4,device='cuda'),rtol=1e-6,atol=1e-6)
        self.assertTrue(bool((actual>=0).all()))
        empty=solver.solve(torch.empty(0,2,device='cuda',dtype=torch.int32));torch.cuda.synchronize()
        torch.testing.assert_close(empty,torch.full_like(empty,.5),rtol=0,atol=0)

    def test_sampling_strided_routed_columns_respects_logical_ownership(self):
        storage=torch.tensor([[0,1,3],[2,0,3]],device='cuda',dtype=torch.int64);ids=storage[:,:2]
        mapping=torch.tensor([[0,4],[1,5],[2,6],[3,7]],device='cuda')
        probabilities=torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]],device='cuda')
        actual=D.dispatch_load_aware(ids,probabilities,mapping,seed=123);torch.cuda.synchronize()
        torch.testing.assert_close(actual,torch.tensor([[0,5],[2,0]],device='cuda'),rtol=0,atol=0)


    def test_sampling_drops_invalid_logical_ids_without_reading_guard_rows(self):
        map_storage=torch.full((6,2),99,device='cuda',dtype=torch.int64)
        mapping=map_storage[1:5];mapping.copy_(torch.tensor([[0,4],[1,5],[2,6],[3,7]],device='cuda'))
        prob_storage=torch.ones(6,2,device='cuda');probabilities=prob_storage[1:5]
        probabilities.copy_(torch.tensor([[1.,0.],[0.,1.],[1.,0.],[0.,1.]],device='cuda'))
        ids=torch.tensor([[-1,4],[0,1]],device='cuda',dtype=torch.int32)
        actual=D.dispatch_load_aware(ids,probabilities,mapping,seed=123);torch.cuda.synchronize()
        torch.testing.assert_close(actual,torch.tensor([[-1,-1],[0,5]],device='cuda',dtype=torch.int32),rtol=0,atol=0)


    def test_full_padded_256_by_264_layout_matches_cpu_reference(self):
        physical=torch.cat([torch.arange(256),torch.arange(8)]).to('cuda')
        mapping=torch.full((256,264),-1,device='cuda',dtype=torch.int64)
        mapping[:,0]=torch.arange(256,device='cuda');mapping[:8,1]=torch.arange(256,264,device='cuda')
        solver=LoadAwareReplicaSolver(physical,mapping,8)
        ids=(torch.arange(512,device='cuda')*7%256).reshape(64,8).int()
        actual=solver.solve(ids);torch.cuda.synchronize()
        counts=torch.bincount(ids.flatten().long(),minlength=256).float()
        expected=D.build_load_aware_probabilities_cpu_reference(counts,mapping,8)
        torch.testing.assert_close(actual.cpu(),expected,rtol=2e-4,atol=2e-5)
        torch.testing.assert_close(actual.sum(1),torch.ones(256,device='cuda'),rtol=1e-6,atol=1e-6)

    def test_graph_counts_solves_and_samples_changed_inputs(self):
        solver,mapping=self.solver();ids=torch.tensor([[0,1],[2,3],[0,0]],device='cuda',dtype=torch.int32)
        count=torch.tensor(2,device='cuda',dtype=torch.int32)
        p=solver.solve(ids,count);D.dispatch_load_aware(ids,p,mapping,seed=123);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            p=solver.solve(ids,count);out=D.dispatch_load_aware(ids,p,mapping,seed=123)
        ids.copy_(torch.tensor([[2,1],[3,3],[0,2]],device='cuda',dtype=torch.int32));count.fill_(3)
        graph.replay();torch.cuda.synchronize()
        self.assertTrue(bool(((out>=0)&(out<8)).all()))
        physical_to_logical=torch.tensor([0,1,2,3,0,1,2,3],device='cuda')
        torch.testing.assert_close(physical_to_logical[out.long()],ids.long(),rtol=0,atol=0)
        expected=D.build_load_aware_probabilities_cpu_reference(torch.bincount(ids.flatten().long(),minlength=4).float(),mapping,2)
        torch.testing.assert_close(p.cpu(),expected,rtol=2e-4,atol=2e-5)


if __name__=='__main__':unittest.main()
