from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.mem_cache import hicache_storage as N
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.srt.mem_cache.storage.file import lru_file_evictor as E

BASE = [N.HiCacheFile, E.LRUFileEvictor]
NATIVE = '--native' in sys.argv
if NATIVE:
    sys.argv.remove('--native')
else:
    from hcu_sglang import register
    from sglang.srt.plugins.hook_registry import HookRegistry
    HookRegistry.reset()
    register()
    HookRegistry.apply_hooks()


class FileBackendCpuTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {'SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR': ''})
        self.env.start()
        self.addCleanup(self.env.stop)

    def backend(self, rank=0, mla=True, cap=0, pp_rank=0, pp_size=1, cp_rank=0, cp_size=1):
        config = N.HiCacheStorageConfig(
            tp_rank=rank, tp_size=2, pp_rank=pp_rank, pp_size=pp_size,
            attn_cp_rank=cp_rank, attn_cp_size=cp_size, is_mla_model=mla,
            enable_storage_metrics=False, is_page_first_layout=True, model_name='org/model',
            extra_config={'file_storage_path': str(self.root), 'enable_metadata_cache': True,
                          'metadata_ttl': -1, 'max_size': cap, 'min_free_space': 0, 'eviction_ratio': 1.0})
        return StorageBackendFactory.create_backend('file', config, None)

    def test_mla_mamba_distinct_rank_payloads_and_shared_kv(self):
        zero, one = self.backend(), self.backend(1)
        a, b = torch.arange(8, dtype=torch.uint8), torch.arange(8, 16, dtype=torch.uint8)
        self.assertTrue(zero.set('page.mamba', a))
        self.assertTrue(one.set('page.mamba', b))
        self.assertTrue(zero.set('page', a))
        self.assertEqual(zero._get_suffixed_key('page'), one._get_suffixed_key('page'))
        for backend, expected in [(zero, a), (one, b)]:
            output = torch.empty_like(expected)
            self.assertIs(backend.get('page.mamba', output), output)
            self.assertTrue(torch.equal(output, expected))
        output = torch.empty_like(a)
        self.assertIs(one.get('page', output), output)
        self.assertTrue(torch.equal(output, a))
        self.assertEqual(len(list(self.root.glob('*.bin'))), 3)

    def test_nonzero_rank_capped_mamba_write_is_admitted(self):
        one = self.backend(1, cap=8)
        self.assertTrue(one.set('page.mamba', torch.arange(8, dtype=torch.uint8)))
        self.assertEqual(one._evictor._total_bytes, 8)
        self.assertFalse(one.set('unowned', torch.arange(8, dtype=torch.uint8)))
        self.assertEqual(one._evictor._total_bytes, 8)

    def test_eviction_preserves_other_rank_and_shared_files(self):
        zero, one = self.backend(cap=16), self.backend(1, cap=8)
        data = torch.arange(8, dtype=torch.uint8)
        self.assertTrue(zero.set('shared', data))
        self.assertTrue(zero.set('zero.mamba', data))
        self.assertTrue(one.set('old.mamba', data))
        old = one._get_suffixed_key('old.mamba')
        self.assertTrue(one.metadata_cache.contains(old))
        self.assertTrue(one.set('new.mamba', data + 1))
        self.assertFalse((self.root / (old + '.bin')).exists())
        self.assertFalse(one.metadata_cache.contains(old))
        for backend, key in [(zero, 'shared'), (zero, 'zero.mamba'), (one, 'new.mamba')]:
            self.assertTrue(backend.exists(key))
        self.assertEqual(len(list(self.root.glob('*.bin'))), 3)
        self.assertEqual(one._evictor._total_bytes, 8)

    def test_restart_metadata_includes_shared_and_own_state_only(self):
        data = torch.arange(8, dtype=torch.uint8)
        zero, one = self.backend(), self.backend(1)
        for backend, key in [(zero, 'shared'), (zero, 'zero.mamba'), (one, 'one.mamba')]:
            self.assertTrue(backend.set(key, data))
        restarted = self.backend(1, cap=8)
        expected = {zero._get_suffixed_key('shared'), one._get_suffixed_key('one.mamba')}
        self.assertEqual(set(restarted.metadata_cache.cache), expected)
        self.assertEqual(set(restarted._evictor._lru), {one._get_suffixed_key('one.mamba')})
        self.assertEqual(restarted._evictor._total_bytes, 8)
        out = torch.empty_like(data)
        self.assertIs(restarted.get('shared', out), out)
        self.assertEqual(set(restarted._evictor._lru), {one._get_suffixed_key('one.mamba')})
        self.assertTrue(torch.equal(out, data))

    def test_rank_zero_startup_eviction_owns_shared_and_tp_state(self):
        zero, one = self.backend(), self.backend(1)
        data = torch.arange(8, dtype=torch.uint8)
        for backend, key in [(zero, 'old'), (zero, 'state.mamba'), (one, 'foreign.mamba')]:
            self.assertTrue(backend.set(key, data))
        old = self.root / (zero._get_suffixed_key('old') + '.bin')
        os.utime(old, (1, 1))
        restarted = self.backend(cap=8)
        self.assertFalse(old.exists())
        self.assertFalse(restarted.metadata_cache.contains(zero._get_suffixed_key('old')))
        self.assertTrue(restarted.exists('state.mamba'))
        self.assertTrue(one.exists('foreign.mamba'))
        self.assertEqual(set(restarted._evictor._lru), {zero._get_suffixed_key('state.mamba')})
        self.assertEqual(restarted._evictor._total_bytes, 8)

    def test_pp_cp_and_non_mla_key_contract(self):
        for mla in [True, False]:
            for rank in [0, 1]:
                backend = self.backend(rank, mla, pp_rank=1, pp_size=2, cp_rank=2, cp_size=4)
                shard = f'_org-model_{rank}_2_2_1_cp2_4'
                self.assertEqual(backend._get_component_key('p', N.PoolName.MAMBA), 'p.mamba' + shard)
                suffix = '_org-model_2_1_cp2_4' if mla else shard
                self.assertEqual(backend._get_component_key('p', N.PoolName.KV), 'p' + suffix)
                self.assertEqual(backend._get_component_key('p', N.PoolName.SWA), 'p.swa' + suffix)

    def test_pending_write_is_not_evicted_and_abort_refunds(self):
        backend = self.backend(1, cap=8)
        evictor = backend._evictor
        pending = backend._get_suffixed_key('pending.mamba')
        self.assertTrue(evictor.reserve(pending, 8))
        self.assertFalse(backend.set('blocked.mamba', torch.arange(8, dtype=torch.uint8)))
        self.assertEqual(evictor._pending_writes, {pending})
        self.assertEqual(evictor._total_bytes, 8)
        evictor.abort(pending)
        self.assertEqual(evictor._total_bytes, 0)
        self.assertEqual(evictor._pending_writes, set())
        self.assertTrue(backend.set('after.mamba', torch.arange(8, dtype=torch.uint8)))
        self.assertEqual(list(self.root.glob('*.tmp')), [])

    def test_failed_atomic_write_releases_reservation_and_tempfile(self):
        backend = self.backend(1, cap=8)
        with patch.object(N.os, 'replace', side_effect=OSError('injected write failure')):
            self.assertFalse(backend.set('fail.mamba', torch.arange(8, dtype=torch.uint8)))
        self.assertEqual(backend._evictor._total_bytes, 0)
        self.assertEqual(backend._evictor._pending_writes, set())
        self.assertEqual(list(self.root.iterdir()), [])
        self.assertTrue(backend.set('ok.mamba', torch.arange(8, dtype=torch.uint8)))

    def test_multi_pool_public_batch_io_and_exists(self):
        class HostPool:
            page_size = 1
            def __init__(self, data):
                self.data = data
            def get_data_page(self, offset, flat=False):
                return self.data[offset]
            def get_dummy_flat_data_page(self):
                return torch.empty_like(self.data[0])
            def set_from_flat_data_page(self, offset, value):
                self.data[offset].copy_(value)
        backend = self.backend(1, cap=32)
        kv = HostPool(torch.arange(16, dtype=torch.uint8).reshape(2, 8))
        mamba = HostPool(torch.arange(32, 48, dtype=torch.uint8).reshape(2, 8))
        zero = self.backend(cap=32)
        for key, row in zip(['a', 'b'], kv.data):
            self.assertTrue(zero.set(key, row))
        for name, pool in [(N.PoolName.KV, kv), (N.PoolName.MAMBA, mamba)]:
            backend.register_mem_host_pool_v2(pool, name)
        transfers = [N.PoolTransfer(name=N.PoolName.MAMBA, host_indices=torch.tensor([0, 1]), keys=['a', 'b'])]
        self.assertEqual(backend.batch_set_v2(transfers), {N.PoolName.MAMBA: [True, True]})
        result = backend.batch_exists_v2(['a', 'b'], transfers)
        self.assertEqual(result.kv_hit_pages, 2)
        expected = mamba.data.clone()
        mamba.data.zero_()
        self.assertEqual(backend.batch_get_v2(transfers), {N.PoolName.MAMBA: [True, True]})
        self.assertTrue(torch.equal(mamba.data, expected))

    def test_factory_preserves_native_base_and_preimport_alias(self):
        backend = self.backend()
        self.assertIsInstance(backend, BASE[0])
        self.assertIs(type(backend), N.HiCacheFile)
        self.assertIs(type(backend._evictor), E.LRUFileEvictor)
        if not NATIVE:
            self.assertIsNot(type(backend), BASE[0])
            self.assertIsNot(type(backend._evictor), BASE[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
