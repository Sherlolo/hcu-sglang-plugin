from array import array
from types import SimpleNamespace as NS
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.disagg import decode_hicache as D
from sglang.srt.disaggregation import decode_hicache_mixin as N
from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.mem_cache.base_prefix_cache import MatchResult, CacheRequestOutcome

class Pool:
    def __init__(self):
        self.row=torch.full((1,8),-1,dtype=torch.int64)
        self.row[0,:2]=torch.tensor([10,11])
        self.writes=[]
    def write(self,index,value):
        self.writes.append((index,value.clone()))
        self.row[index]=value

class Cache:
    def __init__(self):
        self.req_to_token_pool=Pool()
        self.old=NS(name='old');self.host=NS(name='host');self.restored=NS(name='restored')
        self.path=torch.tensor([10,11,50,51,52],dtype=torch.int64)
        self.device_payload=torch.zeros(64,2,dtype=torch.float32)
        self.host_payload=torch.tensor([[1.,2.],[3.,4.],[5.,6.]])
        self.prefetch_done=False;self.event_done=False;self.load_size=3;self.free_slot=True
        self.unlocked=[];self.loads=[];self.finished=[];self.prefetched=[];self.dmas=0
        self.hicache_storage_pass_prefix_keys=True
    def swa_reprefill_tail_tokens(self,*args):return 0
    def match_prefix(self,params):
        self.match_params=params
        return MatchResult(2,self.old,self.host,self.host,host_hit_length=3)
    def prefix_device_indices(self,req):return self.path[:2]
    def is_backuped(self,node):return True
    def is_root(self,node):return False
    def get_last_hash_value(self,node):return 'hash'
    def get_prefix_hash_values(self,node):return ['hash']
    def query_storage_hit_length(self,*args,**kw):
        self.storage_query=(args,kw);return 1
    def prefetch_from_storage(self,*args,**kw):self.prefetched.append((args,kw))
    def has_ongoing_prefetch(self,handle):return bool(self.prefetched)
    def check_prefetch_progress(self,handle):return self.prefetch_done
    def pop_prefetch_loaded_tokens(self,handle):return 1
    def has_free_load_back_slot(self):return self.free_slot
    def init_load_back(self,params):
        self.loads.append(params)
        return self.load_size,self.restored
    def path_device_indices(self,node):return self.path
    def lock(self,node):return NS(node=node)
    def unlock(self,lock):
        if lock is not None:self.unlocked.append(lock)
    def is_load_back_event_done(self,index):return self.event_done
    def ready_to_load_host_cache(self):
        self.device_payload[self.path[2:]]=self.host_payload
        self.dmas+=1;return 4
    def finish(self,handle,outcome):self.finished.append((handle,outcome))

class Queue(D.DecodeHiCachePreallocMixin,D.DecodeHiCacheTransferMixin):
    def __init__(self,cache):
        self.tree_cache=cache;self.scheduler=NS(enable_decode_hicache=True)
        self.transfer_queue=NS(queue=[])

class DecodeHiCacheLifecycleTests(unittest.TestCase):
    def make(self):
        cache=Cache();q=Queue(cache)
        req=NS(origin_input_ids=array('q',range(7)),output_ids=array('q'),rid='r',cache_request_handle='handle',extra_key='adapter',cache_salt='salt',kv=NS(req_pool_idx=0),lock=NS(node=cache.old),last_node=cache.old,_compute_max_prefix_len=lambda n:n)
        pm=D.DecodePrefixMatch(cache.path[:2],2,1,cache.old,cache.host,True)
        dr=NS(req=req,prefix_match=pm,hicache_restore_lock=None,hicache_restore_status=D.HiCacheRestoreResult.PENDING,hicache_restored_kv_indices=None,hicache_load_consumer_index=None,kv_receiver=NS(poll=lambda:KVPoll.Success))
        q.transfer_queue.queue=[dr]
        return cache,q,req,pm,dr

    def test_bridge_preserves_all_native_type_and_enum_identity(self):
        for name in ('DecodeHiCachePreallocMixin','DecodeHiCacheTransferMixin','DecodePrefixMatch','HiCacheRestoreGatedKVReceiver','HiCacheRestoreResult'):
            self.assertIs(getattr(D,name),getattr(N,name))
        self.assertIsNone(D.register_hooks())

    def test_prefetch_wait_dma_ready_commit_changes_only_reserved_request_span(self):
        cache,q,req,pm,dr=self.make();old_lock=req.lock
        gate=D.HiCacheRestoreGatedKVReceiver(dr)
        self.assertEqual(gate.poll(),KVPoll.Transferring)
        self.assertEqual(q._hicache_pending_restore_tokens(),3)
        q._process_hicache_local_restores([dr])
        self.assertEqual(cache.loads,[]);self.assertEqual(cache.dmas,0)
        cache.prefetch_done=True
        q._process_hicache_local_restores([dr])
        self.assertEqual(cache.dmas,1);self.assertEqual(dr.hicache_load_consumer_index,4)
        self.assertIs(dr.hicache_restore_status,D.HiCacheRestoreResult.PENDING)
        self.assertIs(req.last_node,pm.last_device_node)
        self.assertIs(cache.loads[0].req,req)
        self.assertEqual(q._hicache_pending_restore_tokens(),0)
        torch.testing.assert_close(cache.device_payload[cache.path[2:]],cache.host_payload)
        self.assertEqual(gate.poll(),KVPoll.Transferring)
        cache.event_done=True
        q._process_hicache_local_restores([dr])
        self.assertIs(dr.hicache_restore_status,D.HiCacheRestoreResult.READY)
        self.assertEqual(gate.poll(),KVPoll.Success)
        restore_lock=dr.hicache_restore_lock
        q._commit_hicache_local_restore_to_req(dr)
        self.assertTrue(torch.equal(cache.req_to_token_pool.row[0],torch.tensor([10,11,50,51,52,-1,-1,-1])))
        self.assertEqual(req.prefix_len,5);self.assertIs(req.last_node,cache.restored)
        self.assertIs(req.lock,restore_lock);self.assertIsNone(dr.hicache_restore_lock)
        self.assertEqual(cache.unlocked,[old_lock])
        q._clean_hicache_prefetch_resources(dr)
        self.assertEqual(cache.unlocked,[old_lock])
        self.assertEqual(cache.finished,[('handle',CacheRequestOutcome.ABORT)])

    def test_failed_coverage_does_not_publish_request_row_or_restore_lock(self):
        cache,q,req,pm,dr=self.make();cache.prefetch_done=True;cache.load_size=1
        before=cache.req_to_token_pool.row.clone()
        q._process_hicache_local_restores([dr])
        self.assertIs(dr.hicache_restore_status,D.HiCacheRestoreResult.FAILED)
        self.assertEqual(D.HiCacheRestoreGatedKVReceiver(dr).poll(),KVPoll.Failed)
        self.assertIsNone(dr.hicache_restore_lock);self.assertEqual(cache.dmas,0)
        self.assertTrue(torch.equal(cache.req_to_token_pool.row,before))
        self.assertIs(req.last_node,pm.last_device_node)

    def test_prealloc_queries_only_suffix_and_preserves_request_namespace(self):
        cache,q,req,pm,dr=self.make()
        result=MatchResult(2,cache.old,cache.host,cache.host,host_hit_length=2)
        prefix=q._build_decode_prefix_match(req,result,max_prefix_len=6)
        self.assertEqual(prefix.decode_prefix_len,5)
        args,kw=cache.storage_query
        self.assertEqual(list(args[1]),[4,5]);self.assertEqual(kw,{'extra_key':'adapter','cache_salt':'salt'})
        q._start_hicache_prefetch(req,prefix)
        args,kw=cache.prefetched[0]
        self.assertEqual(list(args[2]),[4]);self.assertTrue(prefix.prefetch_registered)
        self.assertEqual(kw,{'extra_key':'adapter','cache_salt':'salt'})

    def test_prefetch_error_falls_back_to_l2_and_empty_restore_becomes_ready(self):
        cache,q,req,pm,dr=self.make()
        def fail(*args,**kw):raise OSError('storage offline')
        cache.prefetch_from_storage=fail
        q._start_hicache_prefetch(req,pm)
        self.assertEqual(pm.l3_storage_hit_length,0);self.assertFalse(pm.prefetch_registered)
        dr.prefix_match=D.DecodePrefixMatch(torch.tensor([1,2]),0,0,cache.old)
        q._process_hicache_local_restores([dr])
        self.assertIs(dr.hicache_restore_status,D.HiCacheRestoreResult.READY)
        self.assertEqual(cache.loads,[]);self.assertEqual(cache.dmas,0)

if __name__=='__main__':unittest.main()
