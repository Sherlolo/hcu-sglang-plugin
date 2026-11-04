from __future__ import annotations
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class OwnedJitAmaxGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def test_real_owned_jit_varlen_values_and_skip_rows(self):
        from hcu_sglang.models.deepseek.current_v4.hcu_backend.candidate_blocks import amax8_varlen
        for dtype in [torch.float32,torch.bfloat16]:
            for out_dtype in [torch.float32,torch.bfloat16]:
                scores=torch.arange(128,device='cuda').reshape(4,32).to(dtype)
                lens=torch.tensor([0,1,17,32],device='cuda',dtype=torch.int32)
                out=torch.full((4,4),-999.,device='cuda',dtype=out_dtype)
                result=amax8_varlen(scores,lens,out=out)
                torch.cuda.synchronize();self.assertIs(result,out)
                expected=torch.full((4,4),-999.,dtype=out_dtype)
                for row,length in enumerate([0,1,17,32]):
                    blocks=(length+7)//8
                    for block in range(max(0,blocks-1)):expected[row,block]=scores[row,8*block:8*(block+1)].float().cpu().max()
                    if blocks:expected[row,blocks-1]=torch.inf
                torch.testing.assert_close(out.cpu(),expected,rtol=0,atol=0)
                out.fill_(-777.);amax8_varlen(scores,lens,topk=4,out=out);torch.cuda.synchronize()
                torch.testing.assert_close(out.cpu(),torch.full((4,4),-777.,dtype=out_dtype),rtol=0,atol=0)

    def test_actual_compiled_kernel_graph_reads_updated_scores_and_lengths(self):
        from hcu_sglang.models.deepseek.current_v4.hcu_backend.candidate_blocks import amax8_varlen
        scores=torch.randn(2,32,device='cuda');lens=torch.tensor([17,32],device='cuda',dtype=torch.int32)
        out=torch.full((2,4),-999.,device='cuda')
        for _ in range(3):amax8_varlen(scores,lens,out=out)
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):amax8_varlen(scores,lens,out=out)
        for length in [9,25]:
            scores.copy_(torch.randn_like(scores));lens.fill_(length);out.fill_(-999.)
            graph.replay();torch.cuda.synchronize()
            blocks=(length+7)//8
            for row in range(2):
                for block in range(blocks-1):self.assertEqual(out[row,block].item(),scores[row,8*block:8*(block+1)].max().item())
                self.assertTrue(torch.isinf(out[row,blocks-1]).item())
                if blocks<4:self.assertTrue(bool((out[row,blocks:]==-999.).all()))


if __name__=='__main__':unittest.main()
