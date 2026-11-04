from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.mem_cache import memory_pool as N
_BASE = [N.MambaPool.__init__, N.MambaPool.State, N.MambaPool.SpeculativeState]
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry
HookRegistry.reset()
register()
HookRegistry.apply_hooks()


class MambaPoolTests(unittest.TestCase):
    def params(self, dtype=torch.float32):
        from sglang.srt.configs.mamba_utils import KimiLinearStateShape, KimiLinearCacheParams, Mamba2StateDType
        shape=KimiLinearStateShape.create(tp_world_size=1,num_heads=2,head_dim=4,conv_kernel_size=4,num_spec=2)
        return KimiLinearCacheParams(shape=shape,layers=[2,3],dtype=Mamba2StateDType(conv=torch.bfloat16,temporal=dtype))

    def options(self,params):
        return dict(size=3,spec_state_size=2,cache_params=params,mamba_layer_ids=params.layers,
                    device='cpu',enable_memory_saver=False,speculative_num_draft_tokens=2,
                    speculative_eagle_topk=1,enable_linear_replayssm_spec=True,
                    linear_replayssm_cache_len=4)

    def test_real_config_to_public_pool_source_replayssm_layout(self):
        params=self.params()
        self.assertTrue(params.use_hcu_kda)
        pool=N.MambaPool(**self.options(params))
        state=pool.get_speculative_mamba2_params_all_layers()
        self.assertIs(type(state),_BASE[2])
        self.assertIs(N.MambaPool.State,_BASE[1])
        self.assertTrue(pool.use_hcu_kda)
        self.assertTrue(pool.use_hcu_kda_replayssm)
        self.assertIsNone(state.intermediate_ssm)
        self.assertIsNone(state.replayssm_d)
        self.assertIsNone(state.replayssm_k)
        self.assertEqual(state.replayssm_g.shape,(2,4,2,2,4))
        self.assertEqual(state.replayssm_rawv.shape,(2,4,2,2,4))
        self.assertEqual(state.replayssm_rawk.shape,(2,4,2,2,4))
        self.assertEqual(state.replayssm_beta.shape,(2,4,2,2))
        ring_bytes=sum(t.nbytes for t in (state.replayssm_g,state.replayssm_rawv,state.replayssm_rawk,state.replayssm_beta))
        self.assertEqual(ring_bytes,params.replayssm_ring_bytes_per_req(2)*(pool.size+1))
        self.assertEqual(state.conv[0].shape,(2,4,3,24))
        self.assertEqual(pool.get_state_dim_per_tensor(),[24,24,2,2])
        pointers,spans,items=pool.get_contiguous_buf_infos()
        self.assertEqual(len(pointers),4)
        self.assertNotIn(state.replayssm_rawv.data_ptr(),pointers)
        self.assertEqual(pool.get_state_layer_ids(),[2,3,2,3])

    def test_copy_clear_and_host_roundtrip_keep_checkpoint_owner(self):
        pool=N.MambaPool(**self.options(self.params()))
        state=pool.mamba_cache
        state.temporal[:,1].fill_(3)
        state.conv[0][:,1].fill_(5)
        state.replayssm_rawv[:,1].fill_(7)
        pool.copy_from(torch.tensor([1]),torch.tensor([2]))
        self.assertTrue(torch.equal(state.temporal[:,1],state.temporal[:,2]))
        self.assertTrue(torch.equal(state.conv[0][:,1],state.conv[0][:,2]))
        self.assertEqual(state.replayssm_rawv[:,2].count_nonzero().item(),0)
        snapshot=pool.get_cpu_copy(torch.tensor([2]))
        pool.clear_slots(torch.tensor([2]))
        self.assertEqual(state.temporal[:,2].count_nonzero().item(),0)
        pool.load_cpu_copy(snapshot,torch.tensor([2]))
        self.assertTrue(torch.equal(state.temporal[:,1],state.temporal[:,2]))

    def test_fp32_gate_rejects_native_constructor_counterexample(self):
        params=self.params(torch.bfloat16)
        before=N.MambaPool.__new__(N.MambaPool)
        _BASE[0](before,**self.options(params))
        self.assertEqual(before.mamba_cache.temporal.dtype,torch.bfloat16)
        with self.assertRaisesRegex(ValueError,'float32'):
            N.MambaPool(**self.options(params))

    def test_hcu_uses_dense_conv_and_preserves_non_hcu_effective_original(self):
        from hcu_sglang.mem_cache import mamba_pool as M
        self.assertFalse(N.conv_window_dedup_enabled(False,False,1,False))
        original=Mock(return_value='previous')
        with patch('hcu_sglang.utils.detection.is_hcu',return_value=False):
            self.assertEqual(M.around_conv_window_dedup(original,False,False,1,False),'previous')
        original.assert_called_once_with(False,False,1,False)


if __name__=='__main__':unittest.main()
