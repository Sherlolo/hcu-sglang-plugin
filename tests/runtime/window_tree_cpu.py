from __future__ import annotations
from array import array
from dataclasses import replace
from types import SimpleNamespace as S
from unittest.mock import patch
import pickle,sys,unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.mem_cache.unified_cache import unified_tree_core as N
ORIGINAL=[N.UnifiedTreeCore.swa_tombstone_ranges]
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class WindowTreeCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def tree(self,locked=False,device="cpu"):
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
        from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
        from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
        class Auxiliary(SWAComponent):component_type=N.ComponentType.AUXILIARY_SWA
        allocator=object.__new__(SWATokenToKVPoolAllocator);allocator.device=torch.device(device);allocator._swa_req_ring=False
        params=CacheInitParams(disable=False,req_to_token_pool=None,token_to_kv_pool_allocator=allocator,page_size=2,sliding_window_size=8)
        cache=S(token_to_kv_pool_allocator=allocator,page_size=2,enable_session_radix_cache=False)
        comps={N.ComponentType.FULL:FullComponent(cache,params),N.ComponentType.SWA:SWAComponent(cache,params),N.ComponentType.AUXILIARY_SWA:Auxiliary(cache,params)}
        tree=N.UnifiedTreeCore(params,comps);cache.tree_core=tree
        key=N.RadixKey(array('q',range(8)),None);node=tree._new_node();node.key=key;node.parent=tree.root_node;tree.root_node.children[key.child_key(2)]=node;node.component_data[N.ComponentType.FULL].value=torch.arange(8,device=device);node.component_data[N.ComponentType.SWA].value=torch.arange(100,108,device=device)
        tree.lru_lists[N.ComponentType.SWA].insert_mru(node);tree.component_evictable_size_[N.ComponentType.SWA]=8
        if locked:node.component_data[N.ComponentType.AUXILIARY_SWA].lock_ref=1
        tree._update_evictable_leaf_sets(node)
        return tree,key,node

    def test_real_tree_ctor_old_signature_fails_new_component_query_is_independent(self):
        tree,key,node=self.tree()
        with self.assertRaisesRegex(TypeError,'component_type'):ORIGINAL[0](tree,key,0,8,component_type=N.ComponentType.AUXILIARY_SWA)
        self.assertEqual(tree.swa_tombstone_ranges(key,0,8),[])
        self.assertEqual(tree.swa_tombstone_ranges(key,0,8,component_type=N.ComponentType.AUXILIARY_SWA),[(0,8)])

    def test_partial_aux_attach_uses_real_split_and_preserves_swa_and_full(self):
        tree,key,node=self.tree();source=torch.tensor([202,203,204,205])
        tree.attach_swa_window(key,2,6,source,component_type=N.ComponentType.AUXILIARY_SWA);source.fill_(-1)
        nodes=[child for child,_,_ in tree._walk_span(key,8)]
        self.assertEqual([len(n.key) for n in nodes],[2,4,2])
        self.assertEqual(torch.cat([n.component_data[N.ComponentType.FULL].value for n in nodes]).tolist(),list(range(8)))
        self.assertEqual(torch.cat([n.component_data[N.ComponentType.SWA].value for n in nodes]).tolist(),list(range(100,108)))
        self.assertEqual(nodes[1].component_data[N.ComponentType.AUXILIARY_SWA].value.tolist(),[202,203,204,205])
        self.assertEqual(tree.swa_tombstone_ranges(key,0,8,component_type=N.ComponentType.AUXILIARY_SWA),[(0,2),(6,8)])
        self.assertEqual(tree.component_evictable_size_[N.ComponentType.AUXILIARY_SWA],4)
        self.assertEqual(tree.component_evictable_size_[N.ComponentType.SWA],8)
        self.assertTrue(tree.lru_lists[N.ComponentType.AUXILIARY_SWA].in_list(nodes[1]))

    def test_locked_attach_is_protected_and_pending_backup_splits_return_actions(self):
        tree,key,node=self.tree(locked=True);node.write_through_pending_id=19
        actions=tree.attach_swa_window(key,2,6,torch.arange(4),component_type=N.ComponentType.AUXILIARY_SWA)
        self.assertEqual(len(actions),2);self.assertTrue(all(a.ack_id==19 for a in actions));self.assertEqual(tree.component_protected_size_[N.ComponentType.AUXILIARY_SWA],4);self.assertEqual(tree.component_evictable_size_[N.ComponentType.AUXILIARY_SWA],0)

    def test_live_overlap_and_size_mismatch_fail_without_replacing_data(self):
        tree,key,node=self.tree()
        with self.assertRaisesRegex(AssertionError,'size mismatch'):tree.attach_swa_window(key,0,4,torch.zeros(3))
        with self.assertRaisesRegex(AssertionError,'over live SWA'):tree.attach_swa_window(key,0,8,torch.zeros(8))
        self.assertEqual(node.component_data[N.ComponentType.SWA].value.tolist(),list(range(100,108)))

    def test_default_swa_window_and_adjacent_tombstones_merge(self):
        tree,key,node=self.tree();node.component_data[N.ComponentType.SWA].value=None;tree.lru_lists[N.ComponentType.SWA].remove_node(node);tree.component_evictable_size_[N.ComponentType.SWA]=0
        tree._split_node(key,node,4);self.assertEqual(tree.swa_tombstone_ranges(key,0,8),[(0,8)])
        tree.attach_swa_window(key,2,6,torch.arange(4));self.assertEqual(tree.swa_tombstone_ranges(key,0,8),[(0,2),(6,8)])
        self.assertEqual(tree.swa_tombstone_ranges(key,0,8,component_type=N.ComponentType.AUXILIARY_SWA),[(0,8)])

    def test_plan_counts_request_readout_copy_and_pickle_keep_components_separate(self):
        from hcu_sglang.mem_cache.window_plan import StagedPrefetchPlan,WindowLoadBackPlan
        from sglang.srt.managers.schedule_batch import Req
        ct=N.ComponentType;plan=StagedPrefetchPlan(3,None,8,0,{ct.SWA:WindowLoadBackPlan(0,8,((0,2),(6,8))),ct.AUXILIARY_SWA:WindowLoadBackPlan(2,8,((2,6),))})
        self.assertEqual(plan.window_tokens(ct.SWA),4);self.assertEqual(plan.window_tokens(ct.AUXILIARY_SWA),4);self.assertEqual(plan.window_tokens(ct.MAMBA),0)
        self.assertEqual(pickle.loads(pickle.dumps(replace(plan,operation_id=4))).operation_id,4)
        req=object.__new__(Req);req.staged_prefetch_plan=plan;req.host_hit_length=req.swa_host_hit_length=req.mamba_host_hit_length=0
        self.assertEqual(req.auxiliary_swa_host_hit_length,4);self.assertTrue(req.needs_host_load_back())

    def test_rust_factory_is_lazy_preserves_custom_factory_and_source_aux_guard(self):
        from hcu_sglang.mem_cache import window_rust as O
        token=object();factory=lambda *args:token;seen=[]
        self.assertIs(O.around_get_factory(lambda name:factory,'python'),factory)
        self.assertIs(O.around_get_factory(lambda name:factory,'rust'),factory)
        wrapped=O.around_get_factory(lambda name:seen.append(name) or O._NATIVE_FACTORY[0],'rust');self.assertEqual(seen,['rust']);self.assertTrue(callable(wrapped))
        key=N.RadixKey(array('q',[1,2]),None)
        with self.assertRaisesRegex(NotImplementedError,'one SWA window'):O.swa_tombstone_ranges(S(),key,0,2,component_type=N.ComponentType.AUXILIARY_SWA)
        with self.assertRaises(NotImplementedError):O.attach_swa_window(S(),key,0,2,torch.arange(2),component_type=N.ComponentType.AUXILIARY_SWA)
        with patch.object(O,'is_hcu',return_value=False):self.assertIs(O.around_get_factory(lambda name:factory,'rust'),factory)

    def test_non_hcu_window_wrappers_preserve_effective_original(self):
        from hcu_sglang.mem_cache import window_tree as O
        for fn in [O.around_swa_tombstone_ranges,O.around_attach_swa_window,O.around__attach_swa_segment]:
            seen=[];token=object()
            with patch.object(O,'is_hcu',return_value=False):self.assertIs(fn(lambda *a,**k:seen.append((a,k)) or token,token,1,tag=2),token)
            self.assertEqual(seen,[((token,1),{'tag':2})])


if __name__=='__main__':unittest.main()
