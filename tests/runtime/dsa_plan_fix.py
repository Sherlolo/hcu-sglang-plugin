from __future__ import annotations

from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.mem_cache import dsa, dsa_cache_layer_split as P
from sglang.srt.plugins.hook_registry import HookRegistry
HookRegistry.reset();dsa.register_hooks();HookRegistry.apply_hooks()
from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from hcu_sglang import dsa_utils as U
from sglang.srt.model_executor import forward_context as F
U.register_hooks();HookRegistry.apply_hooks()


class Batch:
    pass


class DsaPlanTests(unittest.TestCase):
    def setUp(self):
        from sglang.srt.runtime_context import get_parallel
        scope=get_parallel().override(attn_dcp_size=1)
        scope.__enter__();self.addCleanup(scope.__exit__,None,None,None)
        self.parallel=patch.object(P,'get_parallel',return_value=S(attn_cp_group=S(world_size=2,pynccl_comm=None,ranks=[0,1],device_group=object())))
        self.parallel.start();self.addCleanup(self.parallel.stop)
        self.pool=LayerSplitDSATokenToKVPool(size=128,page_size=64,kv_lora_rank=2,dtype=torch.bfloat16,qk_rope_head_dim=2,layer_num=2,device='cpu',index_head_dim=128,enable_memory_saver=False,kv_cache_dim=4,layer_shard_rank=0,layer_shard_size=2)

    def test_public_builder_to_real_pool_compact_consumer_and_current_writes(self):
        reqs=ReqToTokenPool(2,192,'cpu',False)
        reqs.req_to_token[1,:64]=torch.arange(128,192)
        batch=Batch();batch.forward_mode=S(is_extend_without_speculative=lambda:True)
        batch.can_run_tbo=False;batch.tbo_parent_token_range=None
        batch.extend_prefix_lens_cpu=[64];batch.out_cache_loc=torch.tensor([64,65]);batch.req_pool_indices=torch.tensor([1]);batch.dsa_layer_split_main_kv_page_plan=None
        with patch.object(U.U,'get_token_to_kv_pool',return_value=self.pool),patch.object(U,'dsa_use_prefill_cp',return_value=True),patch.object(F,'get_req_to_token_pool',return_value=reqs):
            U.U.maybe_configure_main_kv_page_plan(batch)
            plan=batch.dsa_layer_split_main_kv_page_plan
            self.assertIsInstance(plan,P.MainKVPagePlan)
            self.assertEqual(plan.history_page_ids.tolist(),[2])
            self.assertEqual(plan.all_page_ids.tolist(),[1,2])
            self.assertEqual(self.pool.translate_main_kv_loc_to_compact(torch.tensor([128,129,64,65,-1])).tolist(),[64,65,128,129,-1])
            values=torch.arange(192*4,dtype=torch.float32).view(192,1,4).to(torch.bfloat16)
            self.pool.kv_buffer[0].copy_(values)
            with patch.object(torch.distributed,'broadcast') as transfer:
                self.pool.prefetch_kv_buffer(0)
            self.assertEqual(transfer.call_args.args[0].numel(),64*4)
            self.assertTrue(torch.equal(self.pool.remote_kv_buffer[64:128],values[128:192]))
            def scatter(buffer,loc,nope,rope):buffer[loc]=torch.cat((nope,rope),dim=-1)
            current=torch.full((2,1,4),9,dtype=torch.bfloat16)
            with patch.object(self.pool,'_write_mla_kv_buffer',side_effect=scatter):
                self.pool.set_mla_kv_buffer(S(layer_id=0),batch.out_cache_loc,current[...,:2],current[...,2:])
            self.assertTrue(torch.equal(self.pool.remote_kv_buffer[128:130],current))
            self.assertTrue(torch.equal(self.pool.kv_buffer[0][64:66],current))
            batch.can_run_tbo=True
            U.U.maybe_configure_main_kv_page_plan(batch)
            self.assertIsNone(self.pool._active_main_kv_page_plan)
            self.assertEqual(self.pool.translate_main_kv_loc_to_compact(torch.tensor([128])).tolist(),[128])

    def test_plan_rejects_bad_lengths_and_ignores_dummy_negative_rows(self):
        table=torch.tensor([[0,0,0,0],[64,65,66,67]])
        plan=P.build_main_kv_page_plan(table,torch.tensor([1]),[4],torch.tensor([-1,0,128]),4)
        self.assertEqual(plan.history_page_ids.tolist(),[16])
        self.assertEqual(plan.all_page_ids.tolist(),[16,32])
        with self.assertRaises(ValueError):P.build_main_kv_page_plan(table,torch.tensor([1]),[-1],torch.tensor([]),4)


if __name__=='__main__':unittest.main()
