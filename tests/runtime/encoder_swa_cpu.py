from __future__ import annotations
from types import SimpleNamespace as S
from unittest.mock import patch
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.model_executor import encoder_swa_replay as N
from sglang.srt.managers import tp_worker as W
BASE_RUN=[N.run_encoder_swa_replay]
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class EncoderSwaCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def fixture(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        requests=[S(full_untruncated_fill_ids=list(range(i*100,i*100+200))) for i in range(3)]
        batch=S(forward_mode=ForwardMode.EXTEND,reqs=requests,encoder_swa_reset=[True,False,True],prefix_lens=[4,0,130],extend_lens=[2,1,2],extend_num_tokens=5,input_ids=torch.tensor([900,901,902,903,904]),prefill_input_ids_cpu=torch.tensor([900,901,902,903,904]),out_cache_loc=torch.arange(500,505),req_pool_indices=torch.arange(3),req_pool_indices_cpu=torch.arange(3),seq_lens=torch.tensor([6,1,132]),seq_lens_cpu=torch.tensor([6,1,132]),seq_lens_sum=139,orig_seq_lens=torch.tensor([6,1,132]),extend_logprob_start_lens=[1,0,1],engram_history=torch.zeros(3,2,dtype=torch.int32),global_num_tokens=None,global_num_tokens_for_logprob=None,return_logprob=False,hicache_consumer_index=7)
        resets=[];table=torch.arange(600).reshape(3,200)
        runner=S(device='cpu',token_to_kv_pool=S(request_window=S(reset=lambda x:resets.append(x.tolist()))),req_to_token_pool=S(req_to_token=table),model=S(model=S(engram_hasher=S(max_ngram_size=3))))
        return S(model_runner=runner),batch,resets

    def test_fold_rebuilds_request_rows_and_preserves_original_batch(self):
        from hcu_sglang import encoder_swa_replay as O
        worker,batch,resets=self.fixture();result=O.fold_encoder_swa_replay(worker,batch)
        self.assertEqual(resets,[[0],[2]]);self.assertEqual(result.replay_lens,[4,0,128]);self.assertEqual(result.num_rows,137)
        self.assertEqual(result.keep_rows.tolist(),[4,5,6,135,136]);self.assertEqual(result.batch.input_ids[result.keep_rows].tolist(),batch.input_ids.tolist())
        self.assertEqual(result.batch.out_cache_loc[result.keep_rows].tolist(),batch.out_cache_loc.tolist())
        self.assertEqual(result.batch.prefix_lens,[0,0,2]);self.assertEqual(result.batch.extend_lens,[6,1,130]);self.assertEqual(result.batch.extend_logprob_start_lens,[5,0,129])
        self.assertEqual(result.row_floor.tolist(),[0]*7+[2]*130)
        self.assertEqual(result.scheduled_row[result.keep_rows].tolist(),list(range(5)))
        self.assertEqual(batch.prefix_lens,[4,0,130]);self.assertEqual(batch.extend_lens,[2,1,2]);self.assertEqual(batch.extend_num_tokens,5)
        self.assertEqual(result.batch.engram_history.tolist(),[[0,0],[0,0],[200,201]])

    def test_non_hits_odd_boundary_and_missing_window(self):
        from hcu_sglang import encoder_swa_replay as O
        worker,batch,resets=self.fixture();batch.prefix_lens=[0,0,0]
        self.assertIsNone(O.fold_encoder_swa_replay(worker,batch));self.assertEqual(resets,[[0],[2]])
        worker,batch,resets=self.fixture();batch.prefix_lens[0]=3
        with self.assertRaisesRegex(ValueError,'even cached-prefix'):O.fold_encoder_swa_replay(worker,batch)
        worker.model_runner.token_to_kv_pool.request_window=None;self.assertIsNone(O.fold_encoder_swa_replay(worker,batch))

    def test_batched_legacy_replay_replaces_two_native_forwards_and_updates_counts(self):
        from hcu_sglang import encoder_swa_replay as O
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        worker,batch,_=self.fixture();calls=[];worker.model_runner.forward=lambda fb:calls.append(fb)
        def init(b,runner,**kw):return S(batch=b)
        with patch.object(ForwardBatch,'init_new',side_effect=init):
            BASE_RUN[0](worker,batch);self.assertEqual(len(calls),2);calls.clear()
            batch.global_num_tokens=[5]
            N.run_encoder_swa_replay(worker,batch)
        self.assertEqual(len(calls),1);replay=calls[0].batch
        self.assertEqual(replay.extend_lens,[4,128]);self.assertEqual(replay.global_num_tokens,[132]);self.assertEqual(replay.global_num_tokens_for_logprob,[2]);self.assertEqual(replay.req_pool_indices.tolist(),[0,2]);self.assertTrue(calls[0].encoder_swa_replay)
        batch.global_num_tokens=[1,4]
        with self.assertRaisesRegex(ValueError,'one MLP-sync'):O._build_replay_batch(batch=batch,runner=worker.model_runner,rows=[(0,0,4)])

    def test_drop_dense_and_late_tail_rows_maps_back_to_scheduled_indices(self):
        from hcu_sglang import encoder_swa_replay as O
        worker,batch,_=self.fixture();folded=O.fold_encoder_swa_replay(worker,batch)
        dense=S(hidden_states=torch.arange(140)[:,None],hidden_states_token_indices=None);O.drop_folded_rows(logits_output=dense,folded=folded);self.assertEqual(dense.hidden_states.flatten().tolist(),[4,5,6,135,136])
        indices=torch.cat([torch.arange(3,6),torch.tensor([6]),torch.arange(133,137)])
        tail=S(hidden_states=indices[:,None],hidden_states_token_indices=indices);O.drop_folded_rows(logits_output=tail,folded=folded,tail_lens=[3,1,4])
        self.assertEqual(tail.hidden_states.flatten().tolist(),[4,5,6,135,136]);self.assertEqual(tail.hidden_states_token_indices.tolist(),[0,1,2,3,4])

    def test_history_clamps_multimodal_ids_and_shared_counter_owner(self):
        from hcu_sglang import encoder_swa_replay as O,cp_token_counts as C
        from sglang.srt.managers.schedule_batch import MM_PAD_SHIFT_VALUE
        self.assertEqual(O.history_row(S(max_ngram_size=4),[17,MM_PAD_SHIFT_VALUE+99]),[0,17,MM_PAD_SHIFT_VALUE]);self.assertIs(C.folded_replay_rows,O.folded_replay_rows);self.assertIs(C.replay_len,O.replay_len)

    def test_public_worker_fold_restores_hidden_rows_and_returns_original_plan(self):
        from hcu_sglang import encoder_swa_replay as O
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        worker,batch,resets=self.fixture();events=[];original_plan=object();folded_plan=object();captured=[]
        def init(b,mr,**kw):
            captured.append(kw);return S(input_ids=b.input_ids,seq_lens=b.seq_lens,is_prefill_only=False,sampling_info=S(grammars=None),kv_loc_plan=folded_plan,apply_deprecated_skip_attn_backend_init=lambda x:None)
        def forward(fb,**kw):
            self.assertEqual(fb.encoder_swa_row_floor.shape[0],137);self.assertEqual(fb.encoder_swa_compress_rows.tolist(),[4,5,6,135,136]);return S(logits_output=LogitsProcessorOutput(None,hidden_states=torch.arange(137)[:,None]),can_run_graph=False,expert_distribution_metrics=None,routed_experts_output=None,indexer_topk_output=None)
        worker.set_hicache_consumer=lambda i:events.append(i);worker.is_dllm=lambda:False;worker.pp_group=S(is_last_rank=True);worker.model_runner.attn_backend=S(folds_encoder_swa_replay=True,tail_forward_metadata=None);worker.model_runner.forward=forward;worker.model_runner.kv_index_translator=S(own_plan=lambda b:original_plan);worker.enable_overlap=False;worker.enable_spec=False
        with patch.object(W,'get_exec',return_value=S(features=S(enable_encoder_swa_bounded_replay=True))),patch.object(ForwardBatch,'init_new',side_effect=init),patch.object(W,'capture_pre_sample_logits'):
            result=W.TpModelWorker.forward_batch_generation(worker,batch,is_verify=True,return_hidden_states_before_norm=True,return_kv_loc_plan=True)
        self.assertEqual(events,[7]);self.assertEqual(captured[0]['return_hidden_states_before_norm'],True);self.assertIs(result.kv_loc_plan,original_plan);self.assertIsNot(result.kv_loc_plan,folded_plan);self.assertEqual(result.logits_output.hidden_states.flatten().tolist(),[4,5,6,135,136])

    def test_public_worker_nonfold_backend_runs_replay_before_normal_forward(self):
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        worker,batch,_=self.fixture();worker.model_runner.model.model.engram_hasher=None;events=[]
        def init(b,mr,**kw):return S(batch=b,input_ids=b.input_ids,is_prefill_only=False,kv_loc_plan=None,apply_deprecated_skip_attn_backend_init=lambda x:None)
        def forward(fb,**kw):events.append('replay' if getattr(fb,'encoder_swa_replay',False) else 'extend');return S(logits_output=LogitsProcessorOutput(None),can_run_graph=False,expert_distribution_metrics=None,routed_experts_output=None,indexer_topk_output=None)
        worker.set_hicache_consumer=lambda i:None;worker.is_dllm=lambda:False;worker.pp_group=S(is_last_rank=True);worker.model_runner.attn_backend=S();worker.model_runner.forward=forward
        with patch.object(W,'get_exec',return_value=S(features=S(enable_encoder_swa_bounded_replay=True))),patch.object(ForwardBatch,'init_new',side_effect=init),patch.object(W,'capture_pre_sample_logits'):W.TpModelWorker.forward_batch_generation(worker,batch,is_verify=True)
        self.assertEqual(events,['replay','extend'])

    def test_non_hcu_effective_fallback_preserves_signature_arguments(self):
        from hcu_sglang import encoder_swa_worker as O
        seen=[];token=object()
        with patch.object(O,'is_hcu',return_value=False):self.assertIs(O.around_generation(lambda *a,**k:seen.append((a,k)) or token,token,token,is_verify=True),token)
        self.assertEqual(seen,[((token,token),{'is_verify':True})])


if __name__=='__main__':unittest.main()
