import os,unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.models.deepseek.deepseek_v2.methods import mlp as M
from hcu_sglang.models.deepseek.deepseek_v2.hooks import mlp as H
from sglang.srt.plugins.hook_registry import HookRegistry

class MLPConstructorTests(unittest.TestCase):
    def test_real_public_constructor_matches_das_hcu_and_hip_gates(self):
        HookRegistry.reset();H.register_hooks();HookRegistry.apply_hooks()
        for hcu,enabled,expected in [(True,'1',False),(False,'1',True),(False,'0',False)]:
            with self.subTest(hcu=hcu,enabled=enabled),patch.object(M.C,'_is_dcu',hcu),patch.object(M.N0,'_is_hip',True),patch.object(M.N0,'MergedColumnParallelLinear',return_value=torch.nn.Identity()),patch.object(M.N0,'RowParallelLinear',return_value=torch.nn.Identity()),patch.dict(os.environ,{'SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL':enabled}):
                layer=M.N0.DeepseekV2MLP(8,16,'silu',swiglu_limit=8.,parallel_group='replicated')
                self.assertIsInstance(layer,torch.nn.Module);self.assertEqual(layer.use_fused_clamp_act_mul,expected);self.assertTrue(layer.is_replicated);self.assertEqual(layer.swiglu_limit,8.);self.assertFalse(layer._fused_clamp_fp8_checked)

if __name__=='__main__':unittest.main()
