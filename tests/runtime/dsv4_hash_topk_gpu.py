from __future__ import annotations
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class Dsv4HashTopkGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def inputs(self,n=9,k=8):
        logits=torch.linspace(-20,20,n*128,device='cuda').reshape(n,128)
        ids=(torch.arange(n,device='cuda',dtype=torch.int64)*3)%17
        table=((torch.arange(17*k,device='cuda').reshape(17,k)*7+5)%128).to(torch.int32)
        return logits,ids,table

    def verify(self,inputs,result,shared=2,scale=2.5):
        logits,ids,table=inputs;weights,experts=result
        routed=table[ids.long()]
        selected=logits.gather(1,routed.long()).double()
        value=torch.nn.functional.softplus(selected).sqrt()
        expected=value/(value.sum(1,keepdim=True)+1e-20)
        expected=torch.cat([expected,torch.full((logits.shape[0],shared),1/scale,device='cuda',dtype=torch.float64)],1)
        expected_ids=torch.cat([routed,torch.arange(128,128+shared,device='cuda',dtype=torch.int32).expand(logits.shape[0],shared)],1)
        torch.testing.assert_close(experts,expected_ids,rtol=0,atol=0)
        torch.testing.assert_close(weights.double(),expected,rtol=3e-6,atol=2e-7)
        self.assertTrue(bool(torch.isfinite(weights).all()))

    def test_public_hcu_dispatch_uses_real_jit_and_all_routing_lanes(self):
        from sglang.kernels.ops.moe import dsv4,triton_hash_topk
        from hcu_sglang.layers.moe import dsv4_ops as O
        for k,shared in [(1,0),(8,2),(32,0),(31,1)]:
            data=self.inputs(9,k)
            with patch.object(triton_hash_topk,'hash_topk_triton',side_effect=AssertionError('HCU must use owned JIT')),patch.object(O,'_jit_hash_topk_module',wraps=O._jit_hash_topk_module) as jit:
                result=dsv4.hash_topk(*data,shared,2.5)
                torch.cuda.synchronize();self.verify(data,result,shared)
                self.assertEqual(jit.call_count,1)

    def test_all_underflow_rows_remain_finite(self):
        from sglang.kernels.ops.moe.dsv4 import hash_topk
        data=self.inputs();data[0].fill_(-1000.)
        result=hash_topk(*data,2,2.5);torch.cuda.synchronize()
        weights,ids=result
        self.assertTrue(bool(torch.isfinite(weights).all()))
        torch.testing.assert_close(weights[:,:8],torch.zeros_like(weights[:,:8]),rtol=0,atol=0)
        torch.testing.assert_close(weights[:,8:],torch.full_like(weights[:,8:],.4),rtol=1e-6,atol=0)
        torch.testing.assert_close(ids[:,:8],data[2][data[1]],rtol=0,atol=0)

    def test_graph_replays_changed_logits_and_token_ids(self):
        from sglang.kernels.ops.moe.dsv4 import hash_topk
        data=self.inputs();hash_topk(*data,2,2.5);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):result=hash_topk(*data,2,2.5)
        data[0].copy_(data[0].flip(1));data[1].copy_((data[1]+1)%17)
        graph.replay();torch.cuda.synchronize();self.verify(data,result)

    def test_source_logical_warp_bound_is_rejected(self):
        from sglang.kernels.ops.moe.dsv4 import hash_topk
        with self.assertRaisesRegex(Exception,'topk_fused <= warp size'):
            hash_topk(*self.inputs(9,32),1,2.5)

    def test_empty_token_routing_returns_empty_outputs(self):
        from sglang.kernels.ops.moe.dsv4 import hash_topk
        data=self.inputs(0)
        weights,ids=hash_topk(*data,2,2.5);torch.cuda.synchronize()
        self.assertEqual(tuple(weights.shape),(0,10))
        self.assertEqual(tuple(ids.shape),(0,10))
        from sglang.kernels.ops.moe.dsv4 import mask_topk_ids
        mask_topk_ids(ids,torch.tensor(0,dtype=torch.int32))
        torch.cuda.synchronize()

    def test_mask_strided_topk_with_cpu_and_gpu_valid_token_scalars(self):
        from sglang.kernels.ops.moe.dsv4 import mask_topk_ids
        for device,dtype in [('cpu',torch.int32),('cpu',torch.int64),('cuda',torch.int32)]:
            storage=torch.arange(7*71,device='cuda',dtype=torch.int32).reshape(7,71)
            before=storage.clone();view=storage[:,:32]
            mask_topk_ids(view,torch.tensor(3,device=device,dtype=dtype));torch.cuda.synchronize()
            torch.testing.assert_close(view[:3],before[:3,:32],rtol=0,atol=0)
            torch.testing.assert_close(view[3:],torch.full_like(view[3:],-1),rtol=0,atol=0)
            torch.testing.assert_close(storage[:,32:],before[:,32:],rtol=0,atol=0)

    def test_mask_graph_reads_updated_device_token_count(self):
        from sglang.kernels.ops.moe.dsv4 import mask_topk_ids
        values=torch.zeros(7,8,device='cuda',dtype=torch.int32)
        count=torch.tensor(3,device='cuda',dtype=torch.int32)
        mask_topk_ids(values,count);torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):mask_topk_ids(values,count)
        values.fill_(11);count.fill_(5);graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(values[:5],torch.full_like(values[:5],11),rtol=0,atol=0)
        torch.testing.assert_close(values[5:],torch.full_like(values[5:],-1),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
