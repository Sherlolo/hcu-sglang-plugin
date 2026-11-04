from __future__ import annotations
from array import array
from dataclasses import replace, fields
from types import SimpleNamespace as NS
import pickle
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from sglang.srt.mem_cache import base_prefix_cache as B
BASE_MATCH=B.MatchPrefixParams
BASE_INSERT=B.InsertParams
from hcu_sglang.mem_cache import prefix_protocol as P
P.register_hooks()
from sglang.srt.mem_cache.unified_cache import unified_tree_core as T
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
CT=T.ComponentType


def fixture(with_swa=True):
    freed=[]
    allocator=object.__new__(SWATokenToKVPoolAllocator)
    allocator.device=torch.device('cpu');allocator._swa_req_ring=False
    allocator.free=lambda indices:freed.extend(indices.tolist())
    allocator.free_segment=lambda indices,**kw:freed.extend(indices.tolist())
    allocator.free_full_segment=lambda indices,**kw:freed.extend(indices.tolist())
    params=CacheInitParams(disable=False,req_to_token_pool=None,token_to_kv_pool_allocator=allocator,page_size=2,sliding_window_size=8 if with_swa else None)
    component_cache=NS(token_to_kv_pool_allocator=allocator,page_size=2,enable_session_radix_cache=False,is_mamba_enabled=False,is_swa_enabled=with_swa)
    components={CT.FULL:FullComponent(component_cache,params)}
    if with_swa:components[CT.SWA]=SWAComponent(component_cache,params)
    tree=T.UnifiedTreeCore(params,components);component_cache.tree_core=tree
    cache=object.__new__(UnifiedRadixCache);cache.tree_core=tree
    cache.components=components;cache.tree_components=tuple(components)
    cache.token_to_kv_pool_allocator=allocator;cache.enable_session_radix_cache=False
    cache.disable=False;cache.enable_hicache=False;cache.host_memory_mode='buffer_only'
    cache.cache_controller=None;cache.linker=None;cache.buffer_pipeline=None
    cache.update_eviction_metrics=lambda *args:None
    return cache,tree,freed


class PrefixProtocolTests(unittest.TestCase):
    def test_parameter_schema_additions_preserve_identity_copy_and_pickle(self):
        self.assertIs(B.MatchPrefixParams,BASE_MATCH);self.assertIs(B.InsertParams,BASE_INSERT)
        match=B.MatchPrefixParams(None,return_full_match=True)
        self.assertTrue(replace(match).return_full_match)
        insert=B.InsertParams(force_leaf_creation=True,mamba_keep_idx=3)
        self.assertTrue(replace(insert,prev_prefix_len=2).force_leaf_creation)
        self.assertEqual(pickle.loads(pickle.dumps(insert)).mamba_keep_idx,3)
        self.assertIn('return_full_match',{f.name for f in fields(match)})

    def test_real_tree_full_only_match_can_reuse_tombstoned_swa_prefix(self):
        cache,tree,freed=fixture()
        key=T.RadixKey(array('q',range(8)),None)
        node=tree._new_node();node.key=key;node.parent=tree.root_node
        tree.root_node.children[key.child_key(2)]=node
        node.component_data[CT.FULL].value=torch.arange(8)
        node.component_data[CT.SWA].value=None
        before=node.component_data[CT.FULL].value.clone()
        ordinary=tree.match_prefix(B.MatchPrefixParams(key))
        full=tree.match_prefix(B.MatchPrefixParams(key,return_full_match=True))
        self.assertEqual(ordinary.device_prefix_len,0)
        self.assertEqual(full.device_prefix_len,8)
        self.assertEqual(full.last_device_node,node.id)
        self.assertTrue(torch.equal(node.component_data[CT.FULL].value,before))
        self.assertIsNone(node.component_data[CT.SWA].value)
        self.assertEqual(freed,[])

    def test_source_force_leaf_metadata_and_real_insert_preserve_source_inert_semantics(self):
        cache,tree,freed=fixture()
        req=NS(kv=NS(get_evicted_seqlen=lambda ct:0,cache_protected_len=0),force_radix_leaf_creation=True,swa_branching_seqlen=None)
        params=B.InsertParams()
        tree.components[1].prepare_for_caching_req(req,params,8,False)
        self.assertTrue(params.force_leaf_creation)
        results=[]
        for force in (False,True):
            actual_cache,actual,_=fixture(with_swa=False)
            key=T.RadixKey(array('q',range(8)),None)
            result=actual_cache.insert(B.InsertParams(key=key,value=torch.arange(8),force_leaf_creation=force))
            node=next(iter(actual.root_node.children.values()))
            results.append((result.total_len,node.component_data[CT.FULL].value.clone()))
        self.assertEqual(results[0][0],results[1][0])
        self.assertTrue(torch.equal(results[0][1],results[1][1]))

    def test_joint_reclaim_callback_stops_real_tree_eviction_after_first_sufficient_leaf(self):
        cache,tree,freed=fixture(with_swa=False)
        for start in (0,4):
            cache.insert(B.InsertParams(key=T.RadixKey(array('q',range(start,start+4)),None),value=torch.arange(start,start+4)))
        self.assertEqual(tree.full_evictable_size(),8)
        result=cache.evict_for_alloc(B.EvictParams(num_tokens=8),allocation_reclaim_satisfied=lambda:len(freed)>=4)
        self.assertEqual(result.num_tokens_evicted,4)
        self.assertEqual(len(freed),4);self.assertEqual(tree.full_evictable_size(),4)
        survivor=next(iter(tree.root_node.children.values()))
        self.assertEqual(len(survivor.component_data[CT.FULL].value),4)
        self.assertTrue(set(survivor.component_data[CT.FULL].value.tolist()).isdisjoint(freed))

    def test_base_cache_joint_reclaim_is_explicitly_unsupported(self):
        self.assertEqual(B.BasePrefixCache.evict_for_alloc(NS(evict=lambda params:'native'),B.EvictParams()),'native')
        with self.assertRaisesRegex(NotImplementedError,'joint reclaim'):
            B.BasePrefixCache.evict_for_alloc(NS(),B.EvictParams(),allocation_reclaim_satisfied=lambda:True)

    def test_streaming_session_reuses_own_live_record_without_changing_abort_state(self):
        from sglang.srt.session.streaming_session import StreamingSession
        session=object.__new__(StreamingSession)
        session.slots={'session':NS(kv=NS(holds_kv=True))}
        req=NS(session=NS(session_id='session'),to_finish=None)
        from unittest.mock import patch
        from sglang.srt.session import streaming_session as S
        with patch.object(S,'_is_streaming',return_value=True):
            self.assertTrue(session.will_lend_record(req))
            req.to_finish=object();self.assertFalse(session.will_lend_record(req))
            self.assertIsNotNone(req.session)

if __name__=='__main__':unittest.main()
