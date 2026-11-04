from __future__ import annotations
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class ExpertRecordingGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def test_auto_normal_and_low_latency_counts_share_gpu_buffer(self):
        from sglang.srt.eplb import expert_distribution as N
        cfg=S(moe=S(expert_distribution_recorder_mode='stat',moe_a2a_backend='deepep',deepep_mode='auto',elastic_ep_backend=None))
        metadata=S(num_layers=1,num_local_physical_experts=2,num_physical_experts=4)
        with patch.object(N,'get_exec',return_value=cfg),patch.object(N,'get_device',return_value='cuda'):
            gatherer=N._SinglePassGatherer.init_new(metadata,0)
        gatherer.on_deepep_dispatch_normal(0,[7,3],None,None,None)
        counts=torch.tensor([2,5],device='cuda',dtype=torch.int32)
        gatherer.on_deepep_dispatch_low_latency(0,counts)
        gatherer.on_select_experts(0,torch.zeros(128,8,device='cuda',dtype=torch.int32))
        torch.cuda.synchronize()
        torch.testing.assert_close(gatherer._data,torch.tensor([[9,8]],device='cuda',dtype=torch.int32),rtol=0,atol=0)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):gatherer.on_deepep_dispatch_low_latency(0,counts)
        gatherer.reset();counts.copy_(torch.tensor([4,1],device='cuda',dtype=torch.int32));graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(gatherer._data,torch.tensor([[4,1]],device='cuda',dtype=torch.int32),rtol=0,atol=0)

    def test_real_graph_capture_skips_python_collection_and_host_append(self):
        from sglang.srt.eplb import expert_distribution as N
        events=[];rec=object.__new__(N._ExpertDistributionRecorderReal)
        rec._recording=True;rec._record_current_pass=True
        rec._single_pass_gatherers={'gpu':S(collect=lambda:events.append('collect') or {'value':1})}
        rec._accumulator=S(append=lambda *args:events.append('append'))
        marker=torch.zeros(1,device='cuda');torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            marker.add_(1);rec._on_forward_pass_end(9,{})
        self.assertEqual(events,[])
        graph.replay();torch.cuda.synchronize();self.assertEqual(events,[])
        torch.testing.assert_close(marker,torch.ones_like(marker),rtol=0,atol=0)
        rec._on_forward_pass_end(9,{})
        self.assertEqual(events,['collect','append'])


if __name__=='__main__':unittest.main()
