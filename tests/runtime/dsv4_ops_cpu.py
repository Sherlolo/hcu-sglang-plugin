from __future__ import annotations
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.layers.moe import dsv4_ops as O


class Dsv4OpsContractTests(unittest.TestCase):
    def test_empty_strided_input_is_normalized_and_buffers_preserved(self):
        x=torch.empty(3,16)[:0,::2]
        self.assertNotEqual(x.stride(0),x.shape[1])
        buffers=[torch.empty(0) for _ in range(6)]
        sf=torch.empty(0)
        seen=[]
        class Module:
            def run(self,*args):seen.append(args)
        with patch.object(O,'_jit_mega_moe_pre_dispatch_module',return_value=Module()) as factory:
            O.mega_moe_pre_dispatch(x,*buffers,quant_group_size=64,shared_x_sf=sf,shared_block_m=128)
        factory.assert_called_once_with(64,True)
        args=seen[0]
        self.assertEqual(args[0].shape,x.shape)
        self.assertEqual(args[0].stride(0),x.shape[1])
        self.assertEqual(args[0].dtype,x.dtype)
        for actual,expected in zip(args[1:7],buffers):self.assertIs(actual,expected)
        self.assertIs(args[7],sf);self.assertEqual(args[8],128)

    def test_nonempty_input_identity_and_no_shared_layout(self):
        x=torch.empty(3,16)[:,::2];seen=[]
        class Module:
            def run(self,*args):seen.append(args)
        with patch.object(O,'_jit_mega_moe_pre_dispatch_module',return_value=Module()) as factory:
            O.mega_moe_pre_dispatch(x,*[torch.empty(0) for _ in range(6)])
        factory.assert_called_once_with(32,False)
        self.assertIs(seen[0][0],x)

    def test_non_hcu_around_retains_effective_prior_wrapper(self):
        token=object()
        for hook in [O.around_hash_topk,O.around_mega_moe_pre_dispatch]:
            seen=[]
            def original(*a,**kw):seen.append((a,kw));return token
            with patch.object(O,'is_hcu',return_value=False):
                self.assertIs(hook(original,token,value=13),token)
            self.assertEqual(seen,[((token,),{'value':13})])

    def test_full_registration_rewrites_preimported_consumer_alias(self):
        from hcu_sglang import register
        from hcu_sglang.layers.moe import mega_moe
        from sglang.kernels.ops.moe import dsv4
        from sglang.srt.plugins.hook_registry import HookRegistry
        HookRegistry.reset();register();HookRegistry.apply_hooks()
        self.assertIs(mega_moe.mega_moe_pre_dispatch,dsv4.mega_moe_pre_dispatch)
        for name in ['hash_topk','mega_moe_pre_dispatch']:
            self.assertIn('sglang.kernels.ops.moe.dsv4.'+name,HookRegistry._patched)


if __name__=='__main__':unittest.main()
