from __future__ import annotations

from contextlib import ExitStack, nullcontext
from types import SimpleNamespace as S
import unittest
from unittest.mock import Mock, patch

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang import distributed_backend as T
from hcu_sglang.mem_cache import cache_retention as C


class TopologyTests(unittest.TestCase):
    def test_dp_tp_excludes_cp_and_pipeline_and_passes_rank_options(self):
        parallel=S(tp_size=8,pp_size=2,attn_cp_size=2,attn_dp_size=2)
        group=S()
        create=Mock(return_value=group)
        with patch.object(T,'is_hcu',return_value=True), patch.object(T.P,'get_parallel',return_value=parallel), patch.object(T.P,'get_world_group',return_value=S(local_rank=3)), patch.object(T.P,'init_model_parallel_group',create), patch.object(T.P,'_ATTN_DP_TP',None,create=True):
            T.around_initialize_model_parallel(Mock(),backend='gloo',recovered_rank=True,rank_offset=16,max_world_size=32)
            self.assertIs(T.get_attn_dp_tp_group(),group)
            self.assertEqual(create.call_args.args[0],[[0,1,4,5],[2,3,6,7],[8,9,12,13],[10,11,14,15]])
            self.assertEqual(create.call_args.kwargs['rank_offset'],16)
            self.assertTrue(create.call_args.kwargs['recovered_rank'])
            self.assertFalse(create.call_args.kwargs['use_custom_allreduce'])

    def test_alias_destroy_once_and_reset(self):
        group=S(destroy=Mock())
        names=('_SHARED_EXPERTS_TP','_TP','_PP','_DCP','_MOE_EP','_MOE_TP','_MOE_DP','_ATTN_CP_OVERLAP','_ATTN_CP','_ATTN_DP_TP','_ATTN_TP','_PDMUX_PREFILL_TP_GROUP')
        with ExitStack() as stack:
            stack.enter_context(patch.object(T,'is_hcu',return_value=True))
            stamp=stack.enter_context(patch.object(T.P,'get_parallel',return_value=S(clear_stamp=Mock())))
            stack.enter_context(patch.object(T.P,'get_global_dwdp_manager',return_value=None))
            for name in names: stack.enter_context(patch.object(T.P,name,group,create=True))
            T.around_destroy_model_parallel(Mock())
            group.destroy.assert_called_once()
            for name in names:self.assertIsNone(getattr(T.P,name))

    def test_overlap_group_created_after_native_cp_and_only_once(self):
        T.register_hooks()
        overlap=S()
        native_cp=S(ranks=[0,2])
        with patch.object(T,'is_hcu',return_value=True),patch.object(T.P,'get_parallel',return_value=S(tp_size=4,pp_size=1,attn_cp_size=2,attn_dp_size=1)),patch.object(T.P,'get_world_group',return_value=S(local_rank=0)),patch.object(T.P,'_ATTN_DP_TP',None),patch.object(T.P,'_ATTN_CP_OVERLAP',None),patch.object(T.P,'_ATTN_CP',native_cp),patch.object(T.P,'_ATTN_TP',S()),patch.object(T.P.torch.distributed,'get_rank',return_value=0),patch.object(T.P,'init_model_parallel_group',return_value=overlap) as create:
            T.around_initialize_model_parallel(Mock(),backend='gloo',duplicate_attn_cp_group=True)
            self.assertIs(T.P._ATTN_CP_OVERLAP,overlap)
            self.assertEqual(create.call_args.args[0],[[0,2],[1,3]])
            self.assertEqual(create.call_args.kwargs['group_name'],'attn_cp_overlap')

    def test_single_cp_or_dp_reuses_correct_owner(self):
        for cp,dp,name in [(1,4,'_TP'),(2,1,'_ATTN_TP')]:
            owner=S()
            with patch.object(T,'is_hcu',return_value=True),patch.object(T.P,'get_parallel',return_value=S(tp_size=8,pp_size=1,attn_cp_size=cp,attn_dp_size=dp)),patch.object(T.P,'_ATTN_DP_TP',None,create=True),patch.object(T.P,name,owner),patch.object(T.P,'init_model_parallel_group') as create:
                T.around_initialize_model_parallel(Mock(),backend='gloo')
                self.assertIs(T.get_attn_dp_tp_group(),owner)
                create.assert_not_called()


class CacheRetentionTests(unittest.TestCase):
    def test_source_builder_receives_scheduler_draft_window_and_creates_params(self):
        from hcu_sglang.mem_cache import cache_builder as B
        from sglang.srt.mem_cache import kv_cache_builder as N
        from sglang.srt.plugins.hook_registry import HookRegistry
        HookRegistry.reset()
        C.register_hooks()
        HookRegistry.apply_hooks()
        group=S(cpu_group=object())
        parallel=S(dcp_enabled=False,pp_rank=0,pp_size=1,attn_cp_rank=0,attn_cp_size=1,tp_size=1,tp_rank=0,tp_group=group,attn_cp_group=group,attn_tp_group=group,pp_group=group)
        memory=S(disable_radix_cache=False,radix_eviction_policy='lru',radix_eviction_policy_config=None,enable_session_radix_cache=False)
        schedule=S(chunked_prefill_size=32)
        exec_config=S(features=S(enable_decoder_swa_bounded_replay=False),mamba=S(enable_mamba_extra_buffer=False,enable_mamba_extra_buffer_lazy=False))
        model=S(hf_config=S(),is_multimodal=False)
        worker=S(model_runner=S(token_to_kv_pool=object(),model_config=model,mtp_draft_device_pools=(),mtp_draft_mamba_pools=()),is_hybrid_swa=True,sliding_window_size=64,get_tokens_per_layer_info=lambda:(1,1),get_memory_pool=lambda:(None,object()))
        seen=[]
        def create(context):
            seen.append(context.params)
            return 'tree'
        with ExitStack() as stack:
            for module in (B,C):stack.enter_context(patch.object(module,'is_hcu',return_value=True)) if hasattr(module,'is_hcu') else None
            for name,value in [('get_parallel',parallel),('get_memory',memory),('get_schedule',schedule),('get_exec',exec_config),('get_disagg',S(disaggregation_decode_enable_radix_cache=False))]:
                stack.enter_context(patch.object(B,name,return_value=value))
            stack.enter_context(patch.object(B,'get_resolved_model_impl',return_value=None))
            stack.enter_context(patch.object(B,'uses_ssm_state',return_value=False))
            stack.enter_context(patch.object(B,'is_deepseek_dsa',return_value=False))
            stack.enter_context(patch.object(B,'resolve_decode_retraction_backup',return_value='none'))
            stack.enter_context(patch.object(B,'is_swa_req_ring',return_value=False))
            stack.enter_context(patch.object(B,'get_dp_tp_group',return_value=group))
            stack.enter_context(patch.object(B,'auto_size_hicache',return_value=nullcontext()))
            stack.enter_context(patch.object(B,'create_tree_cache',side_effect=create))
            stack.enter_context(patch.object(B,'init_mm_embedding_cache'))
            scheduler=S(draft_worker=S(draft_swa_window=128))
            def initialize(self):
                return N.build_kv_cache(server_args=S(),model_config=model,tp_worker=worker,page_size=4,spec_algorithm=S(is_eagle=lambda:False),enable_metrics=False,enable_kv_cache_events=False,enable_hierarchical_cache=False)
            result=C.around_scheduler_init(initialize,scheduler)
        self.assertEqual(result.tree_cache,'tree')
        self.assertEqual(seen[0].draft_swa_window,128)
        self.assertEqual(seen[0].sliding_window_size,64)
        self.assertEqual(seen[0].swa_retain_window,128)
        self.assertIsNone(C._SCHEDULER.get())

    def test_constructor_state_and_exception_context_restore(self):
        params=C.CacheInitParams(False,None,None,4,sliding_window_size=64,draft_swa_window=128,swa_gates_prefix_match=False)
        cache=S()
        observed=[]
        with patch.object(C,'is_hcu',return_value=True):
            C.around_unified_init(lambda c,p:observed.append(p.sliding_window_size),cache,params)
            with self.assertRaises(ValueError): C.around_scheduler_init(Mock(side_effect=ValueError),S())
        self.assertEqual(observed,[128])
        self.assertEqual(cache.draft_swa_window,128)
        self.assertEqual(params.sliding_window_size,64)
        self.assertIsNone(C._SCHEDULER.get())

class CommunicationMethodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.plugins.hook_registry import HookRegistry
        HookRegistry.reset()
        T.register_hooks()
        HookRegistry.apply_hooks()

    def test_hcu_capture_allgather_uses_live_copy_in(self):
        communicator=S(disabled=False,_IS_CAPTURING=True,enable_register_for_capturing=True,
            should_custom_ag=lambda x:True,all_gather_unreg=Mock(),all_gather_reg=Mock())
        group=T.P.GroupCoordinator.__new__(T.P.GroupCoordinator)
        group.ca_comm=communicator
        input=torch.ones(4)
        output=torch.empty(8)
        for capturing in (True, False):
            communicator.all_gather_unreg.reset_mock()
            with patch.object(T,'is_hcu',return_value=True),patch.object(T.P,'_ATTN_TP_USE_AITER_CUSTOM_COMM',False),patch.object(torch.cuda,'is_current_stream_capturing',return_value=capturing),patch.object(T.P,'is_in_tc_piecewise_cuda_graph',return_value=False):
                T.P.GroupCoordinator._all_gather_into_tensor(group,output,input)
            communicator.all_gather_unreg.assert_called_once_with(input,out=output,dim=0)
            communicator.all_gather_reg.assert_not_called()

    def test_hcu_general_reduce_scatter_declines_aiter(self):
        group=S(ca_comm=S(disabled=False,should_custom_ar=lambda x:True,reduce_scatter=Mock()))
        with patch.object(T,'is_hcu',return_value=True):
            self.assertFalse(T.P.GroupCoordinator._maybe_aiter_reduce_scatter(group,torch.empty(4),torch.ones(8)))
        group.ca_comm.reduce_scatter.assert_not_called()

    def test_opt_in_attn_scatter_uses_registered_policy(self):
        communicator=S(disabled=False,_IS_CAPTURING=True,enable_register_for_capturing=False,
            should_custom_ar=lambda x:True,reduce_scatter=Mock())
        group=S(ca_comm=communicator)
        input=torch.ones(8);output=torch.empty(4)
        with patch.object(T,'is_hcu',return_value=True),patch.object(T.P,'_ATTN_TP_USE_AITER_CUSTOM_COMM',True),patch.object(torch.cuda,'is_current_stream_capturing',return_value=True):
            self.assertIs(T.P.GroupCoordinator._reduce_scatter_tensor(group,output,input),output)
        communicator.reduce_scatter.assert_called_once_with(input,output,registered=False)

    def test_variable_alltoall_keeps_actual_device_group(self):
        group=S(world_size=2,device_group=object())
        input=torch.ones(3);output=torch.empty(5)
        with patch.object(T,'is_hcu',return_value=True),patch.object(torch.distributed,'all_to_all_single') as transfer:
            self.assertIs(T.P.GroupCoordinator.all_to_all_single(group,output,input,[2,3],[1,2]),output)
        transfer.assert_called_once_with(output,input,output_split_sizes=[2,3],input_split_sizes=[1,2],group=group.device_group)


if __name__=='__main__':unittest.main()
