from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.mem_cache import allocation as A


class CacheAllocationTests(unittest.TestCase):
    def allocator(self, pages, staged=()):
        owner = SimpleNamespace(
            sglang_kvalloc_kernel=True, lightop_kvalloc_kernel=False,
            page_size=4, free_pages=torch.tensor(pages, dtype=torch.int64),
            release_pages=torch.tensor(staged, dtype=torch.int64),
            debug_mode=False, need_sort=False, device='cpu',
        )
        def merge():
            owner.free_pages = torch.cat((owner.free_pages, owner.release_pages)).sort().values
            owner.release_pages = torch.empty(0, dtype=torch.int64)
        owner.merge_and_sort_free = Mock(side_effect=merge)
        return owner

    def call_extend(self, owner, prefix, seq, count=None):
        prefix = torch.tensor(prefix, dtype=torch.int64)
        seq = torch.tensor(seq, dtype=torch.int64)
        last = prefix - 1
        return A.around_alloc_extend(Mock(), owner, prefix, prefix, seq, seq, last,
                                    int((seq-prefix).sum()), count)

    def test_extend_rejects_capacity_before_native_launch(self):
        owner = self.allocator([1])
        native = Mock(side_effect=AssertionError('launched with insufficient capacity'))
        with patch.object(A, 'is_hcu', return_value=True), patch.object(A, '_load_kvcacheio', return_value=SimpleNamespace(hcu_alloc_extend_kernel=native)):
            self.assertIsNone(self.call_extend(owner, [0], [8]))
        native.assert_not_called()
        self.assertEqual(owner.free_pages.tolist(), [1])

    def test_decode_rejects_capacity_before_native_launch(self):
        owner = self.allocator([1])
        seq = torch.tensor([1, 1], dtype=torch.int32)
        native = Mock(side_effect=AssertionError('launched with insufficient capacity'))
        with patch.object(A, 'is_hcu', return_value=True), patch.object(A, '_load_kvcacheio', return_value=SimpleNamespace(hcu_alloc_decode_kernel=native)):
            self.assertIsNone(A.around_alloc_decode(Mock(), owner, seq, seq, seq-1))
        native.assert_not_called()

    def test_unsorted_allocator_reclaims_staged_pages_before_launch(self):
        owner = self.allocator([1], [2])
        def kernel(**kwargs):
            self.assertEqual(kwargs['free_page_ptr'].tolist(), [1,2])
            kwargs['out_indices'].copy_(torch.arange(4,12))
        with patch.object(A, 'is_hcu', return_value=True), patch.object(A, '_load_kvcacheio', return_value=SimpleNamespace(hcu_alloc_extend_kernel=kernel)):
            self.assertEqual(self.call_extend(owner, [0], [8]).tolist(), list(range(4,12)))
        owner.merge_and_sort_free.assert_called_once()
        self.assertEqual(owner.free_pages.numel(), 0)

    def test_lightop_and_missing_lightop_fallback(self):
        for available in (True, False):
            owner = self.allocator([1,2,3,4])
            owner.lightop_kvalloc_kernel=True
            selected=Mock(side_effect=lambda **kw: kw['out_indices'].fill_(4))
            fallback=Mock(side_effect=lambda **kw: kw['out_indices'].fill_(8))
            with patch.object(A, 'is_hcu', return_value=True), patch.object(A, '_load_kvcacheio', return_value=SimpleNamespace(hcu_alloc_extend_kernel=fallback)), patch.object(A, '_get_lightop_alloc_extend_kernel', return_value=selected if available else None):
                self.assertEqual(self.call_extend(owner, [0], [4]).tolist(), [4 if available else 8]*4)
            self.assertEqual(selected.call_count, int(available))
            self.assertEqual(fallback.call_count, int(not available))

    def test_non_hcu_preserves_effective_original(self):
        owner=self.allocator([])
        original=Mock(return_value='effective')
        values=torch.tensor([0])
        with patch.object(A,'is_hcu',return_value=False):
            self.assertEqual(A.around_alloc_decode(original,owner,values,values,values),'effective')
        original.assert_called_once_with(owner,values,values,values)

class PublicAllocatorTests(unittest.TestCase):
    def test_real_paged_constructor_free_and_deferred_group(self):
        from hcu_sglang.environ import register_envs
        from sglang.srt.plugins.hook_registry import HookRegistry
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
        register_envs()
        HookRegistry.reset()
        with patch.object(A, 'is_hcu', return_value=True):
            import sglang.srt.mem_cache.allocation
            A.register_hooks()
            HookRegistry.apply_hooks()
            owner = PagedTokenToKVPoolAllocator(16, 4, torch.bfloat16, 'cpu', None, True)
            allocated = owner.alloc(8)
            owner.free(allocated)
            owner.free(allocated)
            owner.free(torch.tensor([0,1,2,3]))
            self.assertEqual(sorted(owner.get_all_free_pages().tolist()), [1,2,3,4])
            allocated = owner.alloc(4)
            owner.free_group_begin()
            owner.free(allocated)
            self.assertEqual(owner.available_size(), 12)
            owner.free_group_end()
            self.assertEqual(owner.available_size(), 16)
            self.assertTrue(owner.sglang_kvalloc_kernel)
            self.assertIsInstance(owner.lightop_kvalloc_kernel, bool)

    def test_swa_ring_uses_current_public_owner(self):
        from hcu_sglang.mem_cache import swa
        owner=SimpleNamespace(_swa_req_ring=True)
        original=Mock(return_value='ring')
        with patch.object(swa,'is_hcu',return_value=True):
            self.assertEqual(swa.around_free_swa(original,owner,torch.tensor([1])), 'ring')
        original.assert_called_once()

    def test_prefill_full_pool_env_and_capacity(self):
        from hcu_sglang import disaggregation as PD
        from hcu_sglang.environ import register_envs
        from sglang.srt.environ import envs
        register_envs()
        pool_cls=type('DeepSeekV4TokenToKVPool',(),{})
        runner=SimpleNamespace(full_max_total_num_tokens=256, max_token_pool_size=999)
        worker=SimpleNamespace(is_hybrid_swa=True,model_runner=runner)
        queue=SimpleNamespace(scheduler=SimpleNamespace(tp_worker=worker),token_to_kv_pool=pool_cls(),max_total_num_tokens=128)
        with patch.object(PD,'is_hcu',return_value=True), patch.object(envs.SGLANG_DSV4_PD_PREFILL_USE_FULL_TOKEN_POOL,'get',return_value=True):
            PD.after_prefill_bootstrap_queue_init(None,queue)
        self.assertEqual(queue.max_total_num_tokens,256)


    def test_non_hcu_dsa_registration_keeps_native_methods(self):
        from hcu_sglang.mem_cache import dsa
        from sglang.srt.mem_cache.memory_pool import DSATokenToKVPool
        before=dict(vars(DSATokenToKVPool))
        with patch.object(dsa, 'is_hcu', return_value=False):
            dsa.register_hooks()
        self.assertEqual(dict(vars(DSATokenToKVPool)), before)

    def test_non_hcu_dsv4_registration_keeps_native_class(self):
        from hcu_sglang.mem_cache import dsv4
        from sglang.srt.mem_cache import deepseek_v4_memory_pool as N
        from sglang.srt.plugins.hook_registry import HookRegistry
        HookRegistry.reset()
        before=dict(vars(N.DeepSeekV4SingleKVPool))
        with patch.object(dsv4, 'is_hcu', return_value=False):
            dsv4.register_hooks()
            self.assertFalse(dsv4.token_is_bf16_attention_kv_cache(SimpleNamespace(_unified_kv=True)))
        self.assertEqual(dict(vars(N.DeepSeekV4SingleKVPool)), before)

class MlaPaddingTests(unittest.TestCase):
    def test_public_configurator_padding_and_effective_fallback(self):
        from hcu_sglang.mem_cache import mla
        from sglang.srt.mem_cache import kv_cache_configurator as K
        from sglang.srt.plugins.hook_registry import HookRegistry
        from sglang.srt import runtime_context
        HookRegistry.reset()
        mla.register_hooks()
        HookRegistry.apply_hooks()
        model=SimpleNamespace(hf_config=SimpleNamespace(model_type='deepseek_v2'),kv_lora_rank=512,qk_rope_head_dim=0)
        for backend, dtype, expected in [('hcu_mla',torch.float8_e4m3fn,576),('hcu_mla',torch.bfloat16,512),('triton',torch.float8_e4m3fn,512)]:
            kernel=SimpleNamespace(attention_backend=backend,prefill_attention_backend=None,decode_attention_backend=None)
            with patch.object(mla,'is_hcu',return_value=True),patch.object(runtime_context,'get_exec',return_value=SimpleNamespace(kernel=kernel)):
                self.assertEqual(K.calculate_mla_kv_cache_dim(model_config=model,kv_cache_dtype=dtype),expected)
        original=Mock(return_value=321)
        with patch.object(mla,'is_hcu',return_value=False):
            self.assertEqual(mla.around_calculate_mla_kv_cache_dim(original,model_config=model,kv_cache_dtype=torch.float8_e4m3fn),321)
        original.assert_called_once()


if __name__ == '__main__':
    unittest.main()
