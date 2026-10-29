from contextlib import ExitStack, contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import sys
import unittest

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat

bootstrap_aiter_compat()

from sglang.srt.layers.moe import topk as S
from sglang.srt.plugins.hook_registry import HookRegistry
from hcu_sglang.layers.moe import topk as plugin


class TopKDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = staticmethod(S.fused_topk)
        HookRegistry.reset()
        plugin.register()
        HookRegistry.apply_hooks()

    @classmethod
    def tearDownClass(cls):
        S.fused_topk = cls.original
        HookRegistry.reset()

    @contextmanager
    def kernels(self, *, fused=False, hcu=True, aiter=False):
        calls=[]
        def fill(weights,ids,logits,renormalize):
            values,index=logits.float().softmax(-1).topk(weights.shape[-1],dim=-1)
            if renormalize:
                values=values/values.sum(-1,keepdim=True)
            weights.copy_(values)
            ids.copy_(index.to(torch.int32))
        def native(weights,ids,logits,renormalize):
            calls.append(('native',))
            fill(weights,ids,logits,renormalize)
        def lightop(weights,ids,unused,logits,renormalize):
            self.assertIsNone(unused)
            calls.append(('lightop',))
            fill(weights,ids,logits,renormalize)
        def aiter_kernel(hidden,logits,k,renormalize,*,topk_ids,topk_weights):
            calls.append(('aiter',))
            fill(topk_weights,topk_ids,logits,renormalize)
            return topk_weights,topk_ids
        def sigmoid(weights,ids,logits,renormalize,bias,scale,shared):
            calls.append(('sigmoid',bias,scale,shared))
            scores=logits.float().sigmoid()
            choice=(scores+bias if bias is not None else scores).topk(weights.shape[-1],dim=-1).indices
            values=scores.gather(-1,choice)
            if renormalize:
                values=values/values.sum(-1,keepdim=True)
            weights.copy_(values*scale)
            ids.copy_(choice.to(torch.int32))
        def packed(weights,ids,storage,logits,renormalize,*,num_token_non_padded):
            calls.append(('packed',storage,num_token_non_padded))
            fill(weights,ids,logits,renormalize)
            storage.fill_(17)
        light=ModuleType('lightop');light.op=SimpleNamespace(topk_softmax=lightop)
        pack=ModuleType('sglang.kernels.ops.lora.moe.trtllm_lora_temp.topk_softmax_pack');pack.topk_softmax_pack=packed
        with ExitStack() as stack:
            stack.enter_context(patch.object(plugin,'is_dcu',return_value=hcu,create=True))
            stack.enter_context(patch.object(plugin,'_use_fused_topk_softmax',fused,create=True))
            stack.enter_context(patch.object(S,'_use_aiter',aiter))
            stack.enter_context(patch.object(S,'_is_cuda',False))
            stack.enter_context(patch.object(S,'topk_softmax',native))
            stack.enter_context(patch.object(S,'topk_sigmoid',sigmoid))
            stack.enter_context(patch.object(S,'aiter_fused_topk',aiter_kernel,create=True))
            stack.enter_context(patch.dict(sys.modules,{'lightop':light,pack.__name__:pack}))
            yield calls

    def tensors(self):
        return torch.zeros(2,8),torch.tensor([[1.,3.,2.,0.],[4.,1.,2.,3.]])

    def test_public_hook_accepts_current_keywords(self):
        hidden,gates=self.tensors()
        with self.kernels() as calls:
            weights,ids=S.fused_topk(hidden,gates,2,True,num_fused_shared_experts=0,routed_scaling_factor=1.,apply_routed_scaling_factor_on_output=False,packed_out=None,num_token_non_padded=None)
        self.assertEqual(calls,[('native',)])
        self.assertEqual(ids.tolist(),[[1,2],[0,3]])
        self.assertTrue(torch.allclose(weights.sum(-1),torch.ones(2)))
        self.assertEqual(weights.dtype,torch.float32)
        self.assertEqual(ids.dtype,torch.int32)

    def test_das_flag_enables_lightop(self):
        hidden,gates=self.tensors()
        with self.kernels(fused=True) as calls:
            weights,ids=S.fused_topk(hidden,gates,2,True,num_fused_shared_experts=0)
        self.assertEqual(calls,[('lightop',)])
        self.assertEqual(ids.tolist(),[[1,2],[0,3]])
        self.assertTrue(torch.allclose(weights.sum(-1),torch.ones(2)))

    def test_aiter_keeps_priority_over_lightop(self):
        with self.kernels(fused=True,aiter=True) as calls:
            S.fused_topk(*self.tensors(),2,True)
        self.assertEqual(calls,[('aiter',)])

    def test_sigmoid_keeps_bias_scaling_and_shared_count(self):
        hidden,gates=self.tensors();bias=torch.tensor([0.,1.,0.,0.])
        with self.kernels(fused=True) as calls:
            S.fused_topk(hidden,gates,2,True,correction_bias=bias,scoring_func='sigmoid',routed_scaling_factor=2.5,apply_routed_scaling_factor_on_output=True,num_fused_shared_experts=1)
        self.assertEqual(calls[0][0],'sigmoid')
        self.assertIs(calls[0][1],bias)
        self.assertEqual(calls[0][2:],(2.5,1))

    def test_packed_output_and_non_padded_count_reach_native(self):
        storage=torch.empty(2,2,dtype=torch.int32);count=torch.tensor(1)
        with self.kernels(fused=False) as calls:
            S.fused_topk(*self.tensors(),2,True,packed_out=storage,num_token_non_padded=count)
        self.assertEqual(calls[0][0],'packed')
        self.assertIs(calls[0][1],storage)
        self.assertIs(calls[0][2],count)
        self.assertTrue(torch.equal(storage,torch.full_like(storage,17)))

    def test_non_hcu_and_empty_batch_keep_native(self):
        with self.kernels(fused=True,hcu=False) as calls:
            weights,ids=S.fused_topk(torch.empty(0,8),torch.empty(0,4),2,True)
        self.assertEqual(calls,[('native',)])
        self.assertEqual(weights.shape,(0,2))
        self.assertEqual(ids.shape,(0,2))


if __name__ == '__main__':
    unittest.main()
