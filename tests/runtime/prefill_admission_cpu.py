from __future__ import annotations
from types import SimpleNamespace as S
from unittest.mock import patch
import unittest
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.managers import schedule_policy as N,schedule_batch as R
from sglang.srt.mem_cache import prefill_budget as B
from sglang.srt.mem_cache.storage_prefetch import StagedPrefetchPlan
BASE=[N.PrefillAdder,R.Req,B.PrefillBudget.check_prefill]
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class PrefillAdmissionCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def adder(self,chunk=150,capacity=10000,cls=None):
        events=[];cache=S(supports_mamba=lambda:False,supports_prefix_sharing=lambda:False,evictable_size=lambda:0,full_evictable_size=lambda:0,swa_evictable_size=lambda:0,sliding_window_size=128,buffer_pipeline=None,disable=False,lock=lambda node:object(),unlock=lambda lock:None,finish_storage_prefetch_admission=lambda *a,**kw:events.append(('finish',kw)),discard_storage_prefetch_accounting=lambda *a:events.append(('discard',{})))
        allocator=S(page_size=16,available_size=lambda:capacity,full_available_size=lambda:capacity,swa_available_size=lambda:capacity,size_swa=capacity,can_reserve=lambda full,swa,**kw:full<capacity and swa<capacity,request_slot_reserve=lambda **kw:0)
        allocator.create_prefill_budget=lambda cache,**kw:B.PrefillBudget(allocator,cache,**kw)
        with patch.object(N,'_use_exact_chunk_fill',return_value=False):obj=(cls or N.PrefillAdder)(16,cache,allocator,None,1.,1000,chunk)
        obj._check_prefill_tile_budget=lambda tokens:None
        return obj,events

    def req(self,prefix=128,extend=25):
        return S(prefix_len=prefix,full_untruncated_fill_ids=list(range(prefix+extend)),sampling_params=S(max_new_tokens=0,ignore_eos=False),output_ids=[],kv=S(req_pool_idx=None,holds_mamba=False),is_retracted=False,retracted_stain=False,host_hit_length=0,swa_host_hit_length=0,auxiliary_swa_host_hit_length=0,mamba_host_hit_length=0,last_node=object(),needs_host_load_back=lambda:False,materialized_host_hit_len=lambda:0,storage_hit_length=0,storage_hit_start=None,host_hit_is_storage=False,fulfilled_storage_hit_len=lambda p:0,cache_request_handle=object(),host_loaded_length=0)

    def test_real_constructor_and_public_admission_reserve_replay_compute_not_kv(self):
        obj,_=self.adder();req=self.req()
        self.assertIsInstance(obj,BASE[0]);self.assertIsInstance(obj.memory_budget,B.PrefillBudget)
        with patch.object(N,'get_exec',return_value=S(features=S(enable_encoder_swa_bounded_replay=True))):result=obj.add_one_req(req,False,None)
        self.assertIs(result,N.AddReqResult.CONTINUE);self.assertEqual(req.extend_end,144)
        self.assertEqual(obj.rem_chunk_tokens,6);self.assertEqual(obj.rem_input_tokens,856);self.assertEqual(obj.memory_budget.total_offset,32);self.assertEqual(obj.log_replay_tokens,128);self.assertIs(obj.new_chunked_req,req)

    def test_old_public_admission_overdraw_and_exact_first_page_boundary(self):
        old,_=self.adder(cls=BASE[0]);req=self.req()
        with patch.object(N,'get_exec',return_value=S(features=S(enable_encoder_swa_bounded_replay=True))):old.add_one_req(req,False,None)
        self.assertEqual(old.rem_chunk_tokens,-10);self.assertEqual(req.extend_end,153)
        obj,_=self.adder(chunk=144);req=self.req()
        with patch.object(N,'get_exec',return_value=S(features=S(enable_encoder_swa_bounded_replay=True))):result=obj.add_one_req(req,False,None)
        self.assertIs(result,N.AddReqResult.OTHER);self.assertEqual(obj.can_run_list,[req]);self.assertEqual(req.extend_end,144);self.assertEqual(obj.rem_chunk_tokens,0)

    def test_replay_only_chunk_does_not_commit_or_mutate_compute_budget(self):
        for chunk in [127,128,143]:
            obj,_=self.adder(chunk=chunk);req=self.req()
            with patch.object(N,'get_exec',return_value=S(features=S(enable_encoder_swa_bounded_replay=True))):result=obj.add_one_req(req,False,None)
            self.assertIs(result,N.AddReqResult.OTHER);self.assertEqual(obj.can_run_list,[]);self.assertEqual(obj.rem_chunk_tokens,chunk);self.assertEqual(obj.memory_budget.total_offset,0)

    def test_compute_rounding_replay_and_dllm_charge_are_paired(self):
        for exact,charge in [(False,160),(True,153)]:
            obj,_=self.adder(chunk=300);obj.exact_chunk_fill=exact
            obj._update_prefill_budget(128,25,0,False,replay_len=128)
            self.assertEqual(obj.rem_input_tokens,1000-charge);self.assertEqual(obj.rem_chunk_tokens,300-charge);self.assertEqual(obj.memory_budget.total_offset,48)
        obj,_=self.adder();obj.dllm_config=object();obj.rem_dllm_tokens=500
        obj._update_prefill_budget(128,25,0,False,replay_len=128)
        self.assertEqual(obj.rem_dllm_tokens,340);self.assertEqual(obj.rem_chunk_tokens,150)

    def test_old_budget_signature_rejects_auxiliary_new_native_dispatch_accepts(self):
        obj,_=self.adder();kwargs=dict(extend_input_len=25,total_tokens=41,max_new_tokens=0,input_tokens=25,swa_host_hit_length=0,chunk_limit=150,auxiliary_swa_host_hit_length=17)
        with self.assertRaisesRegex(TypeError,'auxiliary_swa_host_hit_length'):BASE[2](obj.memory_budget,**kwargs)
        for cls in [B.PrefillBudget,B.SWAPrefillBudget,B.SharedSWAPrefillBudget]:
            budget=cls(obj.token_to_kv_pool_allocator,obj.tree_cache)
            self.assertEqual(budget.check_prefill(**kwargs),(True,150))

    def test_capacity_defers_staged_only_when_no_admitted_or_running_work(self):
        obj,_=self.adder(capacity=30);seen=[];obj.tree_cache.buffer_pipeline=S(defer_staged_admission=lambda req,pool:seen.append(pool));req=self.req(prefix=0)
        with patch.object(N,'get_exec',return_value=S(features=S(enable_encoder_swa_bounded_replay=False))):self.assertIs(obj.add_one_req(req,False,None),N.AddReqResult.NO_TOKEN)
        self.assertEqual(seen,['prefill_budget']);seen.clear();obj.can_run_list=[object()];obj._defer_staged_for_capacity_if_idle(req,pool='x');self.assertEqual(seen,[])
        obj.can_run_list=[];obj.running_batch=S(reqs=[object()]);obj._defer_staged_for_capacity_if_idle(req,pool='x');self.assertEqual(seen,[])

    def test_retraction_finishes_storage_accounting_once_without_rebilling_replay(self):
        obj,events=self.adder();req=self.req();req.retracted_stain=True
        obj._account_prefill_cache_admission(req,128)
        self.assertEqual(events,[('discard',{}),('finish',{'fulfilled_tokens':0,'reason':None})]);self.assertEqual(obj.rem_input_tokens,1000);self.assertEqual(obj.log_replay_tokens,0)

    def test_req_auxiliary_property_preserves_legacy_plan_and_reads_window_contract(self):
        req=object.__new__(R.Req);self.assertIsInstance(req,BASE[1]);req.host_hit_length=req.swa_host_hit_length=req.mamba_host_hit_length=0
        req.staged_prefetch_plan=StagedPrefetchPlan(operation_id=1,key=None,device_prefix_len=0,full_tokens=0,swa_tokens=16)
        self.assertEqual(req.auxiliary_swa_host_hit_length,0);self.assertFalse(req.needs_host_load_back())
        req.staged_prefetch_plan=S(window_tokens=lambda component:21);self.assertEqual(req.auxiliary_swa_host_hit_length,21);self.assertTrue(req.needs_host_load_back())


if __name__=='__main__':unittest.main()
