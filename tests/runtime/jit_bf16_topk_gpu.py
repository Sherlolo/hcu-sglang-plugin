from __future__ import annotations
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class Bf16ShortTopkGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def data(self):
        scores=torch.stack([torch.arange(-128,128).roll(i*37) for i in range(4)]).to(device='cuda',dtype=torch.bfloat16)
        lens=torch.tensor([0,3,129,256],device='cuda',dtype=torch.int32)
        pages=(torch.arange(64,device='cuda').reshape(4,16)+11).to(torch.int32)
        out=torch.full((4,16),-777,device='cuda',dtype=torch.int32)
        return scores,lens,pages,out

    def verify(self,scores,lens,pages,out):
        for row,length in enumerate(lens.cpu().tolist()):
            count=min(out.shape[1],length)
            selected=scores[row,:length].float().topk(count,sorted=False).indices
            slots=(pages[row].long()[selected//16]*16+selected%16).to(torch.int32)
            actual=out[row]
            self.assertEqual(int((actual<0).sum()),out.shape[1]-count)
            torch.testing.assert_close(actual[actual>=0].sort().values,slots.sort().values,rtol=0,atol=0)

    def test_cold_owned_jit_selects_exact_bf16_values_and_page_slots(self):
        from hcu_sglang.models.deepseek.current_v4.hcu_backend.topk import topk_transform_bf16_small
        scores,lens,pages,out=self.data()
        topk_transform_bf16_small(scores,lens,pages,out,16);torch.cuda.synchronize()
        self.verify(scores,lens,pages,out)

    def test_maximum_width_and_k_with_positive_and_negative_bf16_bins(self):
        from hcu_sglang.models.deepseek.current_v4.hcu_backend.topk import topk_transform_bf16_small
        values=(torch.arange(16384,device='cuda',dtype=torch.int32)+0x2000).to(torch.int16).view(torch.bfloat16)
        scores=torch.stack([values,-values.flip(0)])
        lens=torch.tensor([16384,8173],device='cuda',dtype=torch.int32)
        pages=(torch.arange(512,device='cuda').reshape(2,256).flip(1)+17).to(torch.int32)
        out=torch.full((2,2048),-777,device='cuda',dtype=torch.int32)
        topk_transform_bf16_small(scores,lens,pages,out,64);torch.cuda.synchronize()
        for row,length in enumerate(lens.cpu().tolist()):
            selected=scores[row,:length].float().topk(2048,sorted=False).indices
            expected=(pages[row].long()[selected//64]*64+selected%64).to(torch.int32)
            torch.testing.assert_close(out[row].sort().values,expected.sort().values,rtol=0,atol=0)

    def test_ties_and_signed_zero_return_valid_unique_slots(self):
        from hcu_sglang.models.deepseek.current_v4.hcu_backend.topk import topk_transform_bf16_small
        scores=torch.zeros(2,256,device='cuda',dtype=torch.bfloat16)
        scores[0,::2]=-0.;scores[1,:128]=-1.;scores[1,128:]=1.
        lens=torch.tensor([200,256],device='cuda',dtype=torch.int32)
        pages=torch.arange(32,device='cuda',dtype=torch.int32).reshape(2,16)
        out=torch.full((2,128),-777,device='cuda',dtype=torch.int32)
        topk_transform_bf16_small(scores,lens,pages,out,16);torch.cuda.synchronize()
        for row,length in enumerate(lens.cpu().tolist()):
            slots=out[row].long();self.assertEqual(slots.unique().numel(),128)
            relative=slots-row*256
            self.assertTrue(bool(((relative>=0)&(relative<length)).all()))
            threshold=scores[row,:length].float().topk(128).values[-1]
            self.assertTrue(bool((scores[row,relative].float()>=threshold).all()))

    def test_finite_bf16_subnormal_bins_preserve_exact_order(self):
        from hcu_sglang.models.deepseek.current_v4.hcu_backend.topk import topk_transform_bf16_small
        bits=torch.cat([torch.arange(1,128),torch.tensor([0]),torch.arange(0x8001,0x8080),torch.tensor([0x8000])]).to(torch.int16)
        scores=bits.view(torch.bfloat16).reshape(1,256).to('cuda')
        lens=torch.tensor([256],device='cuda',dtype=torch.int32)
        pages=torch.arange(16,device='cuda',dtype=torch.int32).reshape(1,16)
        out=torch.full((1,64),-777,device='cuda',dtype=torch.int32)
        topk_transform_bf16_small(scores,lens,pages,out,16);torch.cuda.synchronize()
        raw=bits.to(torch.int32)&0xFFFF
        ordered=raw^torch.where((raw&0x8000)!=0,0xFFFF,0x8000)
        expected=ordered.topk(64).indices.to(torch.int32)
        torch.testing.assert_close(out[0].cpu().sort().values,expected.sort().values,rtol=0,atol=0)

    def test_actual_kernel_graph_updates_lengths_scores_and_page_table(self):
        from hcu_sglang.models.deepseek.current_v4.hcu_backend.topk import topk_transform_bf16_small
        scores,lens,pages,out=self.data()
        for _ in range(3):topk_transform_bf16_small(scores,lens,pages,out,16)
        torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):topk_transform_bf16_small(scores,lens,pages,out,16)
        for length in [17,200]:
            scores.copy_(scores.roll(13,-1));lens.fill_(length);pages.add_(7);out.fill_(-777)
            graph.replay();torch.cuda.synchronize();self.verify(scores,lens,pages,out)


if __name__=='__main__':unittest.main()
