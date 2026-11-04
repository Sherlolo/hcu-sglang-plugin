from __future__ import annotations
from types import SimpleNamespace as S
from array import array
from unittest.mock import patch,Mock
import unittest,torch,pickle
import window_tree_cpu as T
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.mem_cache.buffer_mode import pipeline as N
from sglang.srt.mem_cache.hicache_storage import PoolName as PN,PoolTransfer,PoolHitPolicy
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle,InitLoadBackParams,InsertResult
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as CT
BASE=[N.BufferModePipeline,PN,PN.SWA]
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class Fixture:
    def __init__(self,device='cpu',fail_load=False,bad_publish=False):
        self.device=device;self.fail_load=fail_load;self.bad_publish=bad_publish;self.events=[];self.pending=[];self.loaded=[];self.contexts=[]
        self.tree,self.key,node=T.WindowTreeCpuTests().tree(device=device)
        self.tree._split_node(node.key,node,2);self.tree._split_node(node.key,node,2);self.tree._split_node(node.key,node,2)
        self.nodes=[n for n,_,_ in self.tree._walk_span(self.key,8)]
        for i,n in enumerate(self.nodes):
            n.component_data[CT.FULL].value=torch.arange(80+2*i,82+2*i,device=device)
            n.component_data[CT.SWA].value=torch.arange(10+2*i,12+2*i,device=device) if i in (0,2) else None
            n.component_data[CT.AUXILIARY_SWA].value=torch.arange(30+2*i,32+2*i,device=device) if i in (0,1) else None
        tail=self.nodes[-1];tail.component_data[CT.FULL].value=None;tail.component_data[CT.FULL].host_value=torch.tensor([6,7])
        for ct in [CT.SWA,CT.AUXILIARY_SWA]:
            for n in self.nodes:
                if self.tree.lru_lists[ct].in_list(n):self.tree.lru_lists[ct].remove_node(n)
            self.tree.component_evictable_size_[ct]=0
            for n in self.nodes:
                value=n.component_data[ct].value
                if value is not None:self.tree.lru_lists[ct].insert_mru(n);self.tree.component_evictable_size_[ct]+=len(value)
        self.host={PN.KV:torch.arange(1000,1300),PN.SWA:torch.arange(2000,2300),PN.AUXILIARY_SWA:torch.arange(3000,3300)}
        if device=='cuda':self.host={k:v.pin_memory() for k,v in self.host.items()}
        self.data={pool:torch.full((128,),-1,dtype=torch.int64,device=device) for pool in self.host}
        self.mapping={CT.SWA:{0:10,1:11,4:14,5:15},CT.AUXILIARY_SWA:{0:30,1:31,2:32,3:33}}
        self.entries={pool:S(host_pool=S(page_size=2,size=128,size_per_token=8,pool_label=pool.value,free=lambda indices,pool=pool:self.free(pool,indices)),device_free_fn=lambda indices,pool=pool:self.events.append(('device_free',pool,indices.cpu().tolist())),device_pool=None) for pool in self.host}
        group=S(anchor_entry=self.entries[PN.KV],entry_map=self.entries,size=128,free=lambda indices:self.free(PN.KV,indices))
        self.cc=S(mem_pool_host=group,load=self.load,prefetch_tokens_occupied=8,write_policy='write_through')
        self.cache=S(token_to_kv_pool_allocator=S(get_kvcache=lambda:S(size=128),size_full=128,full_available_size=lambda:100),cache_controller=self.cc,tree_core=self.tree,page_size=2,supports_swa=lambda:True,prefix_device_indices=lambda req:self.tree.collect_full_device_indices(req.last_node,self.tree.root_node.id)[:req.prefix_len],_settle_storage_prefetch_hit=lambda *a,**k:self.events.append(('settle',k)),_log_storage_prefetch_deferred=lambda *a:self.events.append(('deferred',a)),_apply_cache_action=lambda action:self.events.append(('action',action)),insert=self.insert,match_prefix=lambda params:self.tree.match_prefix(params),path_device_indices=lambda node:self.tree.collect_full_device_indices(node,self.tree.root_node.id),_finish_storage_prefetch=lambda *a,**k:self.events.append(('finished',k)),sidecar_pool_specs=[],prefetch_threshold=2)
        self.swa=self.component(CT.SWA,PN.SWA);self.aux=self.component(CT.AUXILIARY_SWA,PN.AUXILIARY_SWA)
        self.cache.components={CT.FULL:S(prepare_buffer_load_back=lambda *a:self.context(),buffer_mode_host_pool_entries=lambda group:()),CT.SWA:self.swa,CT.AUXILIARY_SWA:self.aux}
        self.pipe=N.BufferModePipeline(self.cache,self.swa,128,auxiliary_swa_component=self.aux)
        self.handle=CacheRequestHandle(rid='window',attempt_id=0)
        from sglang.srt.managers.schedule_batch import Req
        self.req=object.__new__(Req)
        for k,v in dict(rid='window',cache_request_handle=self.handle,prefix_len=2,last_node=self.nodes[0].id,kv=S(cache_protected_len=2),extra_key=None,cache_salt=None,host_hit_length=0,swa_host_hit_length=0,mamba_host_hit_length=0,host_loaded_length=0,host_hit_is_storage=False,storage_hit_length=0,storage_hit_start=None,staged_prefetch_plan=None).items():setattr(self.req,k,v)
        self.staged=N._StagedPrefetch(request=self.handle,key_tokens=array('q',range(8)),extra_key=None,cache_salt=None,matched_len=0,num_tokens=8,occupied_tokens=8,host_indices=torch.arange(8),aux_xfers=[PoolTransfer(PN.SWA,host_indices=torch.arange(100,108)),PoolTransfer(PN.AUXILIARY_SWA,host_indices=torch.arange(200,208))],hash_values=['a','b','c','d'],operation_id=9)
        self.pipe.staged_prefetches[self.handle]=self.staged

    def component(self,ct,pool):
        def restore(full,window):
            self.mapping[ct].update({int(f)-80:int(w) for f,w in zip(full.cpu(),window.cpu())});self.events.append(('restore',ct,full.cpu().tolist(),window.cpu().tolist()))
        return S(component_type=ct,pool_name=pool,full_window_pages=4,_swa_kv_pool_host=self.entries[pool].host_pool,restore_buffer_mapping=restore,prepare_buffer_load_back=lambda *a:self.context(),buffer_mode_host_pool_entries=lambda group:())

    def context(self):
        owner=self
        class Context:
            load_xfers=()
            def finalize_allocation(self,success):owner.events.append(('allocation',success))
            def get_insert_fields(self):return {}
            def get_redundant_device_slots(self,result):return []
        obj=Context();self.contexts.append(obj);return obj

    def free(self,pool,indices):
        assert not self.pending,'staging freed before copy completion'
        self.events.append(('host_free',pool,indices.cpu().tolist()))

    def load(self,host_indices,node_id,extra_pools):
        self.loaded=extra_pools
        if self.fail_load:return None
        full=torch.arange(86,86+len(host_indices),device=self.device);self.pending.append((PN.KV,host_indices.clone(),full))
        for transfer in extra_pools:
            start={PN.SWA:20,PN.AUXILIARY_SWA:40}[transfer.name]
            transfer.device_indices=torch.arange(start,start+len(transfer.host_indices),device=self.device);self.pending.append((transfer.name,transfer.host_indices.clone(),transfer.device_indices))
        return full

    def insert(self,params):
        self.events.append(('insert',params))
        self.nodes[-1].component_data[CT.FULL].value=(torch.tensor([98,99],device=self.device) if self.bad_publish else params.value[6:].clone())
        for ct in [CT.SWA,CT.AUXILIARY_SWA]:
            if self.nodes[-1].component_data[ct].value is None:self.tree.set_component_device_value(self.nodes[-1].id,ct,torch.tensor([self.mapping[ct][6],self.mapping[ct][7]],device=self.device))
        return InsertResult(prefix_len=params.prev_prefix_len,total_len=8,last_device_node=self.nodes[-1].id)

    def drain(self):
        if self.device=='cuda':
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream());self.pinned_payloads=[]
            with torch.cuda.stream(stream):
                for pool,host,dev in self.pending:
                    payload=self.host[pool][host]
                    if not payload.is_pinned():payload=payload.pin_memory()
                    self.pinned_payloads.append(payload);self.data[pool].index_copy_(0,dev,payload.to(self.device,non_blocking=True))
                self.copy_complete_event=torch.cuda.Event();self.copy_complete_event.record(stream)
            self.copy_complete_event.synchronize()
        else:
            for pool,host,dev in self.pending:self.data[pool][dev]=self.host[pool][host]
        self.pending.clear()



class WindowPipelineCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def test_pool_enum_preserves_legacy_string_keys_and_pickle(self):
        self.assertEqual(BASE[2],PN.SWA);self.assertEqual(hash(BASE[2]),hash(PN.SWA));self.assertEqual({BASE[2]:17}[PN.SWA],17)
        self.assertIs(pickle.loads(pickle.dumps(BASE[2])),PN.SWA)
        self.assertIs(pickle.loads(pickle.dumps(PN.AUXILIARY_SWA)),PN.AUXILIARY_SWA)

    def test_public_prepare_load_and_ack_trim_each_window_preserve_bounce_and_bytes(self):
        from hcu_sglang.mem_cache.window_plan import StagedPrefetchPlan
        f=Fixture();self.assertIsInstance(f.pipe,BASE[0]);self.assertTrue(f.pipe.prepare_staged_prefetch(f.req));p=f.req.staged_prefetch_plan;self.assertIsInstance(p,StagedPrefetchPlan)
        self.assertEqual((p.device_prefix_len,p.full_tokens,p.window_tokens(CT.SWA),p.window_tokens(CT.AUXILIARY_SWA)),(6,2,4,4));self.assertEqual(p.windows[CT.SWA].missing_ranges,((2,4),(6,8)));self.assertEqual(p.windows[CT.AUXILIARY_SWA].missing_ranges,((4,6),(6,8)));self.assertEqual(f.req.auxiliary_swa_host_hit_length,4)
        result=f.pipe.init_load_back(InitLoadBackParams(best_match_node=f.req.last_node,host_hit_length=2,req=f.req));self.assertEqual(result[0],2)
        self.assertEqual([t.host_indices.tolist() for t in f.loaded],[[102,103,106,107],[204,205,206,207]])
        self.assertEqual([len(t.host_indices) for t in f.staged.aux_xfers],[8,8]);self.assertFalse(any(e[0]=='host_free' for e in f.events));self.assertEqual(f.data[PN.KV][86:88].tolist(),[-1,-1])
        f.drain();self.assertEqual(f.data[PN.KV][86:88].tolist(),[1006,1007]);self.assertEqual(f.data[PN.SWA][20:24].tolist(),[2102,2103,2106,2107]);self.assertEqual(f.data[PN.AUXILIARY_SWA][40:44].tolist(),[3204,3205,3206,3207])
        self.assertTrue(f.pipe.try_finish_load_back(-10));self.assertEqual(f.cc.prefetch_tokens_occupied,0);self.assertEqual([len(e[2]) for e in f.events if e[0]=='host_free'],[8,8,8]);self.assertFalse(f.pipe.try_finish_load_back(-10))

    def test_failed_allocation_retains_staging_and_rolls_contexts_without_publication(self):
        f=Fixture(fail_load=True);f.pipe.prepare_staged_prefetch(f.req);self.assertIsNone(f.pipe.init_load_back(InitLoadBackParams(best_match_node=f.req.last_node,host_hit_length=2,req=f.req)))
        self.assertIn(f.handle,f.pipe.staged_prefetches);self.assertEqual(f.pipe.ongoing_buffer_load_back,{});self.assertFalse(any(e[0] in ('host_free','insert','restore') for e in f.events));self.assertEqual([e for e in f.events if e[0]=='allocation'],[('allocation',False)]*3)

    def test_ownership_violation_is_fail_stop_and_pending_buffers_stay_owned(self):
        f=Fixture(bad_publish=True);f.pipe.prepare_staged_prefetch(f.req)
        with self.assertRaisesRegex(RuntimeError,'ownership violation'):f.pipe.init_load_back(InitLoadBackParams(best_match_node=f.req.last_node,host_hit_length=2,req=f.req))
        self.assertIn(-10,f.pipe.ongoing_buffer_load_back);self.assertTrue(f.pending);self.assertFalse(any(e[0]=='host_free' for e in f.events))

    def test_covered_window_drops_its_sidecar_without_touching_staging_owner(self):
        f=Fixture();f.nodes[-1].component_data[CT.FULL].value=torch.tensor([86,87])
        for i,node in enumerate(f.nodes):
            if node.component_data[CT.SWA].value is None:f.tree.set_component_device_value(node.id,CT.SWA,torch.arange(10+2*i,12+2*i))
        f.mapping[CT.SWA]={i:i+10 for i in range(8)}
        f.staged.aux_xfers.append(PoolTransfer(PN.SWA,host_indices=None,indices_from_pool=PN.SWA))
        f.pipe.prepare_staged_prefetch(f.req);self.assertEqual(f.req.staged_prefetch_plan.window_tokens(CT.SWA),0)
        f.pipe.init_load_back(InitLoadBackParams(best_match_node=f.req.last_node,host_hit_length=0,req=f.req));self.assertEqual([t.name for t in f.loaded],[PN.AUXILIARY_SWA]);self.assertEqual(len(f.staged.aux_xfers),3)

    def test_tail_backup_trims_keys_and_device_rows_together_and_extends_prefix(self):
        from hcu_sglang.mem_cache.window_component import extend_storage_prefix
        f=Fixture();f.pipe._swa_tail_only=True;t=PoolTransfer(PN.SWA,device_indices=torch.arange(16));keys=list('abcdefgh')
        self.assertEqual(f.pipe._trim_swa_backup_transfer(t,keys),list('efgh'));self.assertEqual(t.device_indices.tolist(),list(range(8,16)));self.assertEqual(t.hit_policy,PoolHitPolicy.TRAILING_PAGES)
        self.assertEqual(extend_storage_prefix(S(),['p'],['a','b']),['p','a','b']);self.assertIsNone(extend_storage_prefix(S(),None,['a']))
        token=object();self.assertIs(extend_storage_prefix(S(extend_storage_prefix=lambda p,h:token),[],[]),token)

    def test_native_swa_view_uses_source_mapping_and_preserves_custom_callback(self):
        from hcu_sglang.mem_cache.window_component import adapt_window_component
        seen=[];base=S(component_type=CT.SWA,cache=S(_apply_cache_action=lambda action:seen.append(action)))
        view=adapt_window_component(base);self.assertEqual(view.pool_name,PN.SWA);view.restore_buffer_mapping(torch.tensor([4]),torch.tensor([8]));self.assertEqual(len(seen),1)
        custom=S(component_type=CT.SWA,pool_name=PN.SWA,restore_buffer_mapping=lambda *a:7);self.assertIs(adapt_window_component(custom),custom)

    def test_public_cache_init_and_base_component_factory_use_new_pipeline(self):
        from hcu_sglang.mem_cache import window_cache_init as O
        from sglang.srt.mem_cache import unified_radix_cache as U
        from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as A
        from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
        from sglang.srt.mem_cache.cache_init_params import CacheInitParams
        f=Fixture();c=f.cache;c.enable_session_radix_cache=False;c.tree_components=(CT.FULL,CT.SWA,CT.AUXILIARY_SWA);c.metrics_collector=None;c.shutdown=lambda:None;c.host_pool_group=f.cc.mem_pool_host
        params=CacheInitParams(disable=False,req_to_token_pool=None,token_to_kv_pool_allocator=None,page_size=2);c.components[CT.FULL]=FullComponent(c,params)
        with patch.object(O.N,'get_memory',return_value=S(hicache_host_memory_mode='buffer_only',hicache_storage_backend=None,hicache_write_policy='write_through',hicache_storage_prefetch_policy='wait_complete')),patch.object(O.N,'get_observability',return_value=S(extra_metric_labels=None)),patch.object(O.N,'get_model',return_value=S(context_length=8)),patch.object(O.N,'StorageAttachment',return_value=S(resolve_prefetch_threshold=lambda threshold:threshold)),patch.object(A,'attach_hybrid_pool_to_unified_cache'),patch.object(O.N.atexit,'register'):
            U.UnifiedRadixCache.init_hicache(c,S(file_storage_path=None),params)
        self.assertIsInstance(c.buffer_pipeline,N.BufferModePipeline);self.assertIs(c.buffer_pipeline._auxiliary_swa_component,f.aux);self.assertTrue(c.tree_core.is_host_memory_buffer_only)

    def test_shared_host_fence_reserves_full_and_both_windows_twice(self):
        from hcu_sglang.mem_cache.window_pipeline import validate_buffer_only_stack
        from sglang.srt.mem_cache.pool_host.base import HostKVCache
        calls=[];sizes={'full':16,'swa':4,'aux':8};domain=S(capacity_bytes=8384)
        def fits(a,b,empty):calls.append((a,b));return sum(sizes[label]*tokens for label,tokens in a+b)<=domain.capacity_bytes
        domain.can_fit_many_then=fits
        def pool(label):
            p=Mock(spec=HostKVCache);p.shared_allocation_domain=domain;p.pool_label=label;p.size_per_token=sizes[label];p.page_size=2;return p
        full,swa,aux=pool('full'),pool('swa'),pool('aux');group=S(entry_map={},anchor_entry=S(host_pool=full))
        kwargs=dict(sidecar_pool_specs=[],host_pool_group=group,swa_component=S(_swa_kv_pool_host=swa,full_window_pages=4),auxiliary_swa_component=S(_swa_kv_pool_host=aux,full_window_pages=4),storage_prefetch_threshold=256)
        validate_buffer_only_stack(**kwargs);self.assertEqual(calls[0][0],[('full',256),('swa',8),('aux',8)]);domain.capacity_bytes=8300
        with self.assertRaisesRegex(ValueError,'two minimum'):validate_buffer_only_stack(**kwargs)


if __name__=='__main__':unittest.main()
