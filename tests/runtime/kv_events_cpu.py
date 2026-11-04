from __future__ import annotations

from array import array
import gc
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.mem_cache import events as N
from sglang.srt.mem_cache.radix_cache import TreeNode, RadixKey
from sglang.srt.disaggregation.kv_events import BlockStored, BlockRemoved, AllBlocksCleared, StorageMedium
from sglang.srt.managers.scheduler import Scheduler

BASE = [N.LoRANameTable, N.KVCacheEventRecorder, Scheduler.handle_generate_request, Scheduler.handle_embedding_request]
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry
HookRegistry.reset()
register()
HookRegistry.apply_hooks()
from hcu_sglang.mem_cache import kv_events as H, kv_event_admission as A


class KvEventsCpuTests(unittest.TestCase):
    def names(self):
        return H.KvEventLoraNames.from_lora_refs([S(lora_id='a' * 32, lora_name='adapter-A')])

    def node(self, extra=None, salt=None, parent=None, tokens=(1, 2, 3, 4)):
        if parent is None:
            parent = TreeNode()
            parent.key = RadixKey(array('q'))
            parent.hash_value = []
        node = TreeNode()
        node.key = RadixKey(array('q', tokens), extra, cache_salt=salt)
        node.parent = parent
        return node

    def register_req(self, names, extra, name='adapter-A'):
        req = S(lora_id='a' * 32, extra_key=extra)
        names.register(req, name)
        return req

    def test_default_lora_names_separate_ids_and_store_remove_agree(self):
        names = self.names()
        for key, name in [('one', 'first'), ('two', 'second')]:
            self.register_req(names, key, name)
        ids = []
        for extra in ['one', 'two']:
            recorder = N.KVCacheEventRecorder(enabled=True, page_size=2, lora_names=names)
            node = self.node(extra)
            recorder.record_store(node)
            recorder.record_remove(node)
            events = recorder.take()
            self.assertEqual(len(events), 2)
            self.assertIsInstance(events[0], BlockStored)
            self.assertIsInstance(events[1], BlockRemoved)
            self.assertEqual(events[0].block_hashes, events[1].block_hashes)
            self.assertEqual(events[0].lora_name, names.get(extra))
            ids.append(events[0].block_hashes)
        self.assertNotEqual(ids[0], ids[1])
        old_ids = []
        for extra in ['one', 'two']:
            old = BASE[1](enabled=True, page_size=2, lora_names=names)
            old.record_store(self.node(extra))
            old_ids.append(old.take()[0].block_hashes)
        self.assertEqual(old_ids[0], old_ids[1])

    def test_native_coalesces_different_lora_names_but_owned_does_not(self):
        def events():
            return [BlockStored(block_hashes=[i], parent_block_hash=i-1 if i > 1 else None,
                                token_ids=[i], block_size=1, lora_id=None, medium=StorageMedium.GPU,
                                cache_salt=None, session_id=None, lora_name=name)
                    for i, name in [(1, 'A'), (2, 'B')]]
        old = BASE[1](enabled=True, page_size=1)
        new = N.KVCacheEventRecorder(enabled=True, page_size=1)
        for event in events():
            old.enqueue(event)
        for event in events():
            new.enqueue(event)
        self.assertEqual(len(old.take()), 1)
        self.assertEqual(len(new.take()), 2)

    def test_plain_namespace_keeps_native_hash_and_queue_take_clears(self):
        for salt in [None, 'tenant-salt']:
            old = BASE[1](enabled=True, page_size=2)
            new = N.KVCacheEventRecorder(enabled=True, page_size=2)
            old.record_store(self.node(salt=salt))
            new.record_store(self.node(salt=salt))
            self.assertEqual(old.take(), new.take())
            self.assertEqual(new.take(), [])
        disabled = N.KVCacheEventRecorder(enabled=False, page_size=2)
        node = self.node()
        disabled.record_store(node)
        disabled.record_remove(node)
        disabled.record_all_cleared()
        self.assertEqual(disabled.take(), [])

    def test_parent_link_and_host_writeback_use_same_persistent_namespace(self):
        names = self.names()
        req = self.register_req(names, 'elastic|adapter')
        recorder = N.KVCacheEventRecorder(enabled=True, page_size=2, lora_names=names)
        parent = self.node(req.extra_key, tokens=(1, 2))
        child = self.node(req.extra_key, parent=parent, tokens=(3, 4))
        recorder.record_store(parent)
        first = recorder.take()[0]
        recorder.record_remove(parent)
        self.assertEqual(recorder.take()[0].block_hashes, first.block_hashes)
        del req
        gc.collect()
        recorder.record_store(child, medium=StorageMedium.CPU, session_id='session')
        event = recorder.take()[0]
        self.assertEqual(event.parent_block_hash, first.block_hashes[-1])
        self.assertEqual(event.lora_name, 'adapter-A')
        self.assertEqual(event.medium, StorageMedium.CPU)
        self.assertEqual(event.session_id, 'session')

    def test_dynamo_format_keeps_native_namespace_and_resolves_adapter_suffix(self):
        names = self.names()
        key = 'tenant-' + 'a' * 32
        self.assertIsNone(names.get(key))
        self.assertEqual(names.resolve(key), 'adapter-A')
        old = BASE[1](enabled=True, page_size=2, dynamo_format=True, lora_names=names)
        new = N.KVCacheEventRecorder(enabled=True, page_size=2, dynamo_format=True, lora_names=names)
        old.record_store(self.node(key, salt='s'))
        new.record_store(self.node(key, salt='s'))
        self.assertEqual(old.take(), new.take())
        self.register_req(names, key)
        old.record_remove(self.node(key, salt='s'))
        new.record_remove(self.node(key, salt='s'))
        self.assertEqual(old.take(), new.take())

    def test_names_fallback_old_ipc_and_legacy_table_get(self):
        names = self.names()
        self.register_req(names, 'final-after-elastic', None)
        self.assertEqual(names.get('final-after-elastic'), 'adapter-A')
        self.assertIsNone(names.get('unregistered'))
        names.register(S(lora_id=None, extra_key='base'), 'unused')
        self.assertIsNone(names.get('base'))
        legacy = N.LoRANameTable.from_lora_refs([S(lora_id='a' * 32, lora_name='legacy')])
        self.assertEqual(legacy.get('prefix-' + 'a' * 32), 'legacy')

    def scheduler_fixture(self, enabled=True, sharing=True):
        class Fixture:
            init_kv_event_lora_names = Scheduler.init_kv_event_lora_names
            handle_generate_request = Scheduler.handle_generate_request
            handle_embedding_request = Scheduler.handle_embedding_request
            def _maybe_namespace_elastic_radix_cache(self, req):
                def namespace(receiver, request):
                    request.extra_key += '|elastic_ep_size=2'
                return A.around_namespace(namespace, self, req)
            def _handle_generate_request(self, recv_req, cleanup, **kwargs):
                self.admit(recv_req)
            def _handle_embedding_request(self, recv_req, cleanup, **kwargs):
                self.admit(recv_req)
            def admit(self, recv_req):
                self.req = S(lora_id='a' * 32, extra_key='namespace')
                self._maybe_namespace_elastic_radix_cache(self.req)
                self.observed = self.kv_event_lora_names.get(self.req.extra_key)
                if getattr(recv_req, 'fail', False):
                    raise ValueError('after namespace')
        fixture = Fixture()
        fixture.kv_events_publisher = S(enable_kv_cache_events=enabled)
        fixture.tree_cache = S(supports_prefix_sharing=lambda:sharing)
        with patch.object(A, 'get_lora', return_value=S(lora_paths=[S(lora_id='a' * 32, lora_name='fallback')])):
            fixture.init_kv_event_lora_names()
        return fixture

    def test_public_generate_and_embedding_record_after_final_namespace(self):
        for method in ['handle_generate_request', 'handle_embedding_request']:
            fixture = self.scheduler_fixture()
            getattr(fixture, method)(S(mm_inputs=None, lora_name='explicit'))
            self.assertEqual(fixture.req.extra_key, 'namespace|elastic_ep_size=2')
            self.assertEqual(fixture.observed, 'explicit')
            self.assertIsNone(fixture.kv_event_lora_names.get('namespace'))
            self.assertIsNone(A._ADMISSION.get())

    def test_registered_native_namespace_hook_runs_in_public_handler(self):
        from types import MethodType
        from sglang.srt.managers import scheduler as SN
        fixture = self.scheduler_fixture()
        fixture._maybe_namespace_elastic_radix_cache = MethodType(Scheduler._maybe_namespace_elastic_radix_cache, fixture)
        with patch.object(SN, 'get_exec', return_value=S(moe=S(elastic_ep_backend=None))):
            fixture.handle_generate_request(S(mm_inputs=None, lora_name='native-seam'))
        self.assertEqual(fixture.req.extra_key, 'namespace')
        self.assertEqual(fixture.observed, 'native-seam')
        self.assertIsNone(A._ADMISSION.get())

    def test_admission_disabled_events_or_nonsharing_do_not_name_namespace(self):
        for enabled, sharing in [(False, True), (True, False)]:
            fixture = self.scheduler_fixture(enabled, sharing)
            fixture.handle_generate_request(S(mm_inputs=None, lora_name='explicit'))
            self.assertIsNone(fixture.observed)
        fixture = self.scheduler_fixture()
        fixture.handle_embedding_request(S(mm_inputs=None))
        self.assertEqual(fixture.observed, 'fallback')

    def test_admission_context_restored_on_exception_and_nested_call(self):
        fixture = self.scheduler_fixture()
        token = A._ADMISSION.set(('outer', 'request'))
        try:
            with self.assertRaisesRegex(ValueError, 'after namespace'):
                fixture.handle_generate_request(S(mm_inputs=None, lora_name='name', fail=True))
            self.assertEqual(A._ADMISSION.get(), ('outer', 'request'))
        finally:
            A._ADMISSION.reset(token)
        self.assertIsNone(A._ADMISSION.get())

    def test_native_tree_insert_and_eviction_publish_owned_recorder(self):
        import torch
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
        from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as CT
        names = self.names()
        self.register_req(names, 'actual-tree')
        params = CacheInitParams(False, None, None, 2, enable_kv_cache_events=True, kv_event_lora_names=names)
        cache = S(page_size=2, token_to_kv_pool_allocator=None, enable_session_radix_cache=False)
        tree = UnifiedTreeCore(params, {CT.FULL:FullComponent(cache, params)})
        cache.tree_core = tree
        self.assertIs(type(tree.kv_events), N.KVCacheEventRecorder)
        tree.kv_events.take()
        key = RadixKey(array('q', [1, 2, 3, 4]), 'actual-tree')
        step = tree.begin_insert(InsertParams(key=key, value=torch.arange(4)))
        while step.result is None:
            self.assertEqual(step.actions, [])
            step = tree.resume_insert()
        self.assertEqual(tree.end_insert(), [])
        stored = tree.kv_events.take()
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].lora_name, 'adapter-A')
        result = tree.evict_device_leaf(step.result.last_device_node, is_write_back=False)
        removed = tree.kv_events.take()
        self.assertEqual(len(removed), 1)
        self.assertIsInstance(removed[0], BlockRemoved)
        self.assertEqual(removed[0].block_hashes, stored[0].block_hashes)
        self.assertEqual(result.tracker[CT.FULL], 4)
        self.assertTrue(torch.equal(torch.cat(result.device_frees.pop(CT.FULL)), torch.arange(4)))
        self.assertEqual(dict(result.device_frees), {})
        self.assertEqual(dict(result.host_frees), {})

    def test_non_hcu_admission_preserves_effective_original(self):
        calls = []
        with patch.object(A, 'is_hcu', return_value=False):
            result = A.around_admission(lambda receiver, req, **kw:calls.append((receiver, req, kw)) or 17, 'self', 'req', flag=3)
        self.assertEqual(result, 17)
        self.assertEqual(calls, [('self', 'req', {'flag': 3})])
        self.assertIsNone(A._ADMISSION.get())


if __name__ == '__main__':
    unittest.main(verbosity=2)
