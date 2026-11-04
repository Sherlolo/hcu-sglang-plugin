from __future__ import annotations
from contextlib import ExitStack,contextmanager
from dataclasses import fields,replace,MISSING
from types import SimpleNamespace as S
import os,pickle,unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.managers.scheduler_components import dp_attn as N
from sglang.srt.managers import schedule_batch as B
BASE=[N.prepare_mlp_sync_batch_raw,N.MLPSyncBatchInfo,B.ScheduleBatch]
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class PrefillSyncCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    @contextmanager
    def context(self,mode):
        parallel=S(num_dp_ranks=1,attn_tp_size=1,attn_cp_size=1,enable_cp_tp_group_sharing=False,tp_group=S(device_group=None,cpu_group=None,device='cpu'))
        preparer=S(prepare_all_gather=lambda batch:(True,mode.value),compute_output=lambda info:(None,mode))
        with patch.object(N,'get_parallel',return_value=parallel),patch.object(N,'TboDPAttentionPreparer',return_value=preparer),patch.object(N,'world_dp_gather_enabled',return_value=False),patch.object(N,'check_cuda_graph_backend',side_effect=lambda phase,backend:backend==N.Backend.BREAKABLE),patch.object(N,'_ENABLE_METRICS_DP_ATTENTION',False),patch.dict(os.environ,{'SGLANG_SCHEDULER_SKIP_ALL_GATHER':'0','SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION':'0'}):
            yield

    def args(self,batch):
        from sglang.srt.model_executor.runner import PrefillCudaGraphRunner
        runner=object.__new__(PrefillCudaGraphRunner);runner.enable_lora=False;runner.max_context_size=None;runner.max_seq_len=None;runner._backend_vote_prefill_cuda_graph=None
        seen=[];runner.can_replay_locally=lambda **kw:seen.append(kw) or True
        mr=S(prefill_cuda_graph_runner=runner,spec_algorithm=None,model_config=None,attn_backend=S(folds_encoder_swa_replay=True))
        args=dict(local_batch=batch,model_runner=mr,get_idle_batch=lambda:None,disable_cuda_graph=False,require_mlp_tp_gather=False,disable_overlap_schedule=False,offload_tags=set())
        return args,seen

    def test_native_public_producer_reproduces_missing_argument_and_owned_idle_fixes_it(self):
        args,_=self.args(None)
        with self.context(N.ForwardMode.IDLE):
            with self.assertRaisesRegex(TypeError,'num_tokens'):BASE[0](**args)
            self.assertIsNone(N.prepare_mlp_sync_batch_raw(**args))

    def test_public_producer_passes_full_encoder_fold_count_to_vote_and_metadata(self):
        batch=S(forward_mode=N.ForwardMode.EXTEND,extend_num_tokens=3,encoder_swa_reset=[True],prefix_lens=[129],extend_lens=[3],extend_logprob_start_lens=[2],return_logprob=False,batch_size=lambda:1,input_embeds=None,replace_embeds=None,multimodal_inputs=[],seq_lens_cpu=torch.tensor([132]),reqs=[],spec_info=None)
        args,seen=self.args(batch)
        with self.context(N.ForwardMode.EXTEND):result=N.prepare_mlp_sync_batch_raw(**args)
        self.assertIs(result,batch);self.assertEqual(seen[0]['num_tokens'],131)
        self.assertEqual(batch.global_num_tokens,[131]);self.assertEqual(batch.global_cp_num_tokens,[0]);self.assertTrue(batch.can_run_dp_prefill_cuda_graph)

    def test_info_schema_wire_width_and_fallback_cp_column(self):
        obj=N.MLPSyncBatchInfo(num_dp_ranks=1,tp_size=1,cp_size=1,num_tokens=7,num_tokens_for_logprob=1,can_run_decode_cuda_graph=False,can_run_draft_cuda_graph=True,can_run_prefill_cuda_graph=True,is_extend_in_batch=True,local_can_run_tbo=True,local_forward_mode=N.ForwardMode.EXTEND.value,cp_num_tokens=4)
        self.assertIsInstance(obj,BASE[1]);self.assertEqual(obj._get_local_tensor('cpu').shape,(12,));self.assertEqual(obj._get_fallback_tensor('cpu')[-1].item(),0)
        obj.finalize_local();self.assertEqual(obj.global_cp_num_tokens,[4]);self.assertEqual(obj.tp0_info_cpu[0,11].item(),4)

    def test_schedule_record_declares_cp_counts_for_copy_and_pickle(self):
        native={f.name for f in fields(BASE[2])};self.assertTrue(native<={f.name for f in fields(B.ScheduleBatch)})
        values={f.name:None for f in fields(B.ScheduleBatch) if f.default is MISSING and f.default_factory is MISSING};values['global_cp_num_tokens']=[4,0]
        obj=B.ScheduleBatch(**values);self.assertIsInstance(obj,BASE[2]);self.assertEqual(replace(obj).global_cp_num_tokens,[4,0]);self.assertEqual(pickle.loads(pickle.dumps(obj)).global_cp_num_tokens,[4,0])

    def test_cp_geometry_and_mixed_replica_layout(self):
        from hcu_sglang import cp_token_counts as C
        self.assertEqual(C.compute_zigzag_cp_physical_token_count([3,5],1),8)
        self.assertEqual(C.compute_zigzag_cp_physical_token_count([3,5],2),4)
        self.assertEqual(C.normalize_dp_cp_token_counts([8,0],[4,0]),[4,0])
        self.assertEqual(C.normalize_dp_cp_token_counts([8,5],[4,0]),[0,0])
        with self.assertRaises(ValueError):C.normalize_dp_cp_token_counts([1],[])
        with patch.object(C,'get_parallel',return_value=S(enable_cp_tp_group_sharing=False)),patch.dict(os.environ,{'SGLANG_ENABLE_CP_V2':'0'}):self.assertEqual(C.get_cp_v2_physical_token_count(num_tokens=8,extend_seq_lens=[3,5],cp_size=2),0)

    def test_non_hcu_wrappers_preserve_effective_original(self):
        from hcu_sglang import prefill_graph_sync as O
        for name in ['_resolve_elastic_world_num_dp_ranks','_update_gather_batch','prepare_mlp_sync_batch_raw']:
            token=object();seen=[]
            with patch.object(O,'is_hcu',return_value=False):self.assertIs(getattr(O,'around_'+name)(lambda *a,**k:seen.append((a,k)) or token,token,option=3),token)
            self.assertEqual(seen,[((token,),{'option':3})])


if __name__=='__main__':unittest.main()
