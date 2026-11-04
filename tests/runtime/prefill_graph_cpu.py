from __future__ import annotations
from contextlib import ExitStack,nullcontext
from dataclasses import replace,fields
from types import SimpleNamespace as S
import os,pickle,unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.model_executor.runner import prefill_cuda_graph_runner as N
from sglang.srt.model_executor import runner as R
from sglang.srt.model_executor.model_runner_components import cuda_graph_setup as G
from sglang.srt.layers import logits_processor as L
NATIVE=[N.PrefillCudaGraphRunner,L.LogitsProcessorOutput]
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class PrefillGraphCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def runner(self,**kw):
        obj=object.__new__(N.PrefillCudaGraphRunner)
        defaults=dict(_is_full_backend=False,_fa4_prefill=False,_qwen_bcg_hc_sidechannel=False,_qwen_bcg_mtp_draft=False,_captured_attn_metadata_max_bs=None,has_mha_companion_layers=False,prefill_backend_name=N.Backend.BREAKABLE,max_context_size=None,max_seq_len=None,capture_hidden_mode=N.CaptureHiddenMode.NULL,max_num_tokens=16,capture_num_tokens=[4,8,16],enable_lora=False,enable_cp_bcg_capture=False,_backend_can_run_prefill_cuda_graph=None,_backend_vote_prefill_cuda_graph=None,_capture_chunked_prefix=False)
        defaults.update(kw)
        for k,v in defaults.items():setattr(obj,k,v)
        return obj

    def vote(self,obj,**kw):
        args=dict(batch_size=1,num_tokens=4,input_embeds=None,replace_embeds=None,prefix_lens=[0],is_target_verify=False,capture_hidden_mode=None,return_logprob=False,batch_max_context_len=8)
        args.update(kw);return obj.can_replay_locally(**args)

    def test_native_class_identity_and_preimported_consumers(self):
        from hcu_sglang import prefill_graph_runner as O
        self.assertIs(N.PrefillCudaGraphRunner,O.PrefillCudaGraphRunner)
        self.assertIs(R.PrefillCudaGraphRunner,O.PrefillCudaGraphRunner)
        self.assertIs(G.PrefillCudaGraphRunner,O.PrefillCudaGraphRunner)
        self.assertTrue(issubclass(O.PrefillCudaGraphRunner,NATIVE[0]))
        self.assertIsInstance(self.runner(),NATIVE[0])

    def test_output_dataclass_preserves_native_fields_copy_and_pickle(self):
        native={f.name for f in fields(NATIVE[1])}
        self.assertTrue(native<={f.name for f in fields(L.LogitsProcessorOutput)})
        out=L.LogitsProcessorOutput(None,draft_topk_index=torch.tensor([[4]]),draft_top1_probs=torch.tensor([1.]))
        copied=replace(out,hidden_states=torch.ones(1,2));self.assertIs(copied.draft_topk_index,out.draft_topk_index)
        self.assertEqual(pickle.loads(pickle.dumps(copied)).draft_topk_index.item(),4)
        self.assertIsInstance(out,NATIVE[1])

    def test_full_and_breakable_trim_preserve_draft_rows_and_hidden_row_identity(self):
        output=L.LogitsProcessorOutput(None,hidden_states=torch.arange(12).reshape(6,2),draft_topk_index=torch.arange(6).reshape(6,1),hidden_states_token_indices=torch.tensor([4,5]),mm_input_embeds=torch.ones(6,3))
        for full,expected in [(True,2),(False,3)]:
            obj=self.runner(_is_full_backend=full,raw_bs=2,raw_num_tokens=3,model_runner=S(spec_algorithm=S(is_speculative=lambda:True)))
            result=obj._trim_logits_output(output)
            self.assertEqual(result.draft_topk_index.shape[0],expected)
            self.assertEqual(result.hidden_states.shape[0],3)
            self.assertIs(result.hidden_states_token_indices,output.hidden_states_token_indices)
            self.assertEqual(result.mm_input_embeds.shape[0],3)

    def test_scope_nesting_exception_reset_and_compile_probe(self):
        from hcu_sglang import prefill_graph_policy as P
        self.assertFalse(P.is_in_prefill_graph_scope())
        with P.prefill_graph_scope(full_graph=True,raw_num_tokens=3):
            self.assertTrue(P.is_in_full_prefill_graph());self.assertEqual(P.get_prefill_raw_num_tokens(),3)
            with self.assertRaisesRegex(RuntimeError,'scope'):
                with P.prefill_graph_scope(full_graph=False,raw_num_tokens=7):
                    self.assertFalse(P.is_in_full_prefill_graph());raise RuntimeError('scope')
            self.assertEqual(P.get_prefill_raw_num_tokens(),3)
            with patch.object(torch.compiler,'is_compiling',return_value=True):self.assertFalse(P.is_in_prefill_graph_scope());self.assertFalse(P.is_in_full_prefill_graph())
        self.assertIsNone(P.get_prefill_raw_num_tokens())

    def test_real_forward_context_publishes_owned_scope_and_cleans_up(self):
        from hcu_sglang import prefill_graph_policy as P
        obj=self.runner(model_runner=S(attn_backend=object()),attention_layers=[],moe_layers=[],moe_fusions=[],dsa_indexers=[],mha_companion_layers=[],quant_config=None)
        with patch.object(N,'set_forward_batch',return_value=nullcontext()),patch.object(N,'forward_context',return_value=nullcontext()),patch.object(N,'set_tc_piecewise_forward_context',return_value=nullcontext()):
            with obj._prefill_forward_context(S(),num_tokens=8,raw_num_tokens=3):self.assertTrue(P.is_in_prefill_graph_scope());self.assertEqual(P.get_prefill_raw_num_tokens(),3)
        self.assertFalse(P.is_in_prefill_graph_scope())

    def test_both_context_limits_gate_local_vote_and_dsa_prefix_exception(self):
        obj=self.runner(max_seq_len=8)
        self.assertTrue(self.vote(obj));self.assertFalse(self.vote(obj,batch_max_context_len=9));self.assertFalse(self.vote(obj,batch_max_context_len=None))
        obj.max_context_size=6;self.assertFalse(self.vote(obj));self.assertTrue(self.vote(obj,batch_max_context_len=6))
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
        obj=self.runner(has_mha_companion_layers=True,model_runner=S(attn_backend=object.__new__(DeepseekSparseAttnBackend)))
        with patch.object(N,'is_cuda',return_value=False):self.assertTrue(self.vote(obj,prefix_lens=[2]))
        obj.model_runner.attn_backend=object()
        with patch.object(N,'is_cuda',return_value=False):self.assertFalse(self.vote(obj,prefix_lens=[2]))

    def test_schedule_vote_reads_backend_vote_and_seq_limit_from_real_runner(self):
        from sglang.srt.managers.scheduler_components import dp_attn as D
        mode=N.ForwardMode.EXTEND
        batch=S(forward_mode=mode,input_embeds=None,replace_embeds=None,prefix_lens=[0],return_logprob=False,multimodal_inputs=[],seq_lens_cpu=torch.tensor([9]),batch_size=lambda:1)
        obj=self.runner(max_seq_len=8)
        args=dict(local_batch=batch,num_tokens=4,prefill_graph_runner=obj,coordinated_prefill=True,breakable_prefill=True,spec_algorithm=None,model_config=None)
        self.assertFalse(D._local_prefill_cuda_graph_vote(**args));batch.seq_lens_cpu.fill_(8);self.assertTrue(D._local_prefill_cuda_graph_vote(**args))
        obj._backend_vote_prefill_cuda_graph=lambda batch:False;self.assertFalse(D._local_prefill_cuda_graph_vote(**args))

    def test_complete_constructor_prewarm_before_backend_and_retains_new_metadata(self):
        from hcu_sglang import prefill_graph_runner as O
        for default,explicit in [(None,None),(32,None),(32,64)]:
            events=[];attn=S(dllm_attention=None,vote_prefill_cuda_graph=lambda b:True)
            if default is not None:attn.prefill_cuda_graph_default_max_seq_len=default
            cfg=S(backend=N.Backend.BREAKABLE,bs=[4,8],max_context_size=None,max_seq_len=explicit)
            spec=S(is_eagle=lambda:False,is_dflash_family=lambda:False)
            mr=S(attn_backend=attn,model=S(),model_config=S(is_multimodal=False,hf_config=S(model_type='deepseek_v41')),lora_manager=None,is_generation=True,spec_algorithm=spec,is_draft_worker=False,req_to_token_pool=S(size=2,req_to_token=torch.empty(2,32)),dtype=torch.float32,attn_tp_sequence_sharded=lambda n:False,pp_group=S(is_first_rank=True),get_pp_proxy_topk_size=lambda:0,get_pp_proxy_residual_num_blocks=lambda:0,get_pp_proxy_dspark_hidden_size=lambda:0,attention_layers=[],mha_companion_layers=[],moe_layers=[],moe_fusions=[],server_args=S())
            def base_init(obj,mr):obj.model_runner=mr;obj.device='cpu';obj.device_module=S(synchronize=lambda:events.append('sync'));obj.pp_size=1;obj.return_hidden_states_mode=N.CaptureHiddenMode.NULL
            with ExitStack() as stack:
                patches=[(N.BaseCudaGraphRunner,'__init__',base_init),(NATIVE[0],'__init__',lambda *a:(_ for _ in ()).throw(AssertionError('old constructor ran'))),(N,'get_schedule',lambda:S(enable_mixed_chunk=False)),(N,'get_exec',lambda:S(graph=S(cuda_graph_config=S(prefill=cfg)))),(N,'get_parallel',lambda:S(attn_dp_enabled=False,tp_group=S(barrier=lambda:events.append('barrier')))),(N,'is_qwen4_exp',lambda cfg:False),(N,'get_prefill_num_tokens_to_capture',lambda x:list(x)),(O.PrefillCudaGraphRunner,'_is_mamba_track_enabled',lambda self:False),(O.PrefillCudaGraphRunner,'_input_embeds_hidden_size',lambda self:4),(O.PrefillCudaGraphRunner,'_cache_loc_dtype',lambda self:torch.int64),(N.PrefillInputBuffers,'create',lambda **k:S(share_buffers=lambda:None)),(N,'build_prefill_registry',lambda **k:object()),(N,'enable_num_token_non_padded',lambda:False),(N,'require_gathered_buffer',lambda:False),(N,'require_mlp_tp_gather',lambda:False),(N,'require_attn_tp_gather',lambda:False),(N,'is_dsa_enable_prefill_cp',lambda:False),(N,'is_mla_cp_enabled',lambda:False),(N,'maybe_pre_warm_aiter_chip_info',lambda:events.append('prewarm')),(N,'resolve_prefill_backend',lambda self:events.append('backend') or object()),(O.PrefillCudaGraphRunner,'capture',lambda self:events.append('capture'))]
                for obj,key,value in patches:stack.enter_context(patch.object(obj,key,value))
                obj=O.PrefillCudaGraphRunner(mr)
            self.assertEqual(events,['prewarm','backend','sync','barrier','capture'])
            self.assertEqual(obj.max_seq_len,explicit if explicit is not None else default)
            self.assertTrue(obj.is_dsv41_model);self.assertIs(obj._backend_vote_prefill_cuda_graph,attn.vote_prefill_cuda_graph)

    def test_public_execute_waits_for_full_kv_layers_before_replay(self):
        for full,consumer,expected_wait in [(True,0,True),(False,0,False),(True,-1,False)]:
            events=[];counter=S(consumer_index=consumer,num_layers=5,wait_until=lambda n:events.append(('wait',n)))
            obj=self.runner(_is_full_backend=full,raw_num_tokens=2,device_module=object(),backend=S(replay_session=lambda:nullcontext()),model_runner=S(token_to_kv_pool=S(layer_transfer_counter=counter)))
            obj._validate_capture_hidden_mode=lambda fb:None;obj.load_batch=lambda fb,**k:S(input_ids=torch.zeros(4,dtype=torch.long));obj._shape_key=lambda *a:S(variant_label=None);obj._uses_eager_prefill_tail=lambda:False;obj._execute_tc_piecewise=lambda *a,**k:events.append('replay') or 7;obj._finalize_execute_output=lambda out:out
            with patch.object(N,'maybe_publish_prefill_shared_read_done',side_effect=lambda *a:events.append('published')):self.assertEqual(obj.execute(S()),7)
            self.assertEqual(events,['published']+([('wait',4)] if expected_wait else [])+['replay'])

    def test_megamoe_selector_updates_canonical_buffer_only_when_enabled(self):
        from hcu_sglang import mega_moe_graph as O
        buf=torch.tensor([99],dtype=torch.int32)
        with patch.object(O,'_is_hcu',True),patch.dict(O.N._MEGA_MOE_SYMM_BUFFER,{'x':S(cuda_graph_num_tokens=buf)},clear=True):
            with patch.dict(os.environ,{'SGLANG_HCU_MEGA_MOE_RUNTIME':'deep_gemm'}):O.set_mega_moe_cuda_graph_num_tokens(3);self.assertEqual(buf.item(),99)
            with patch.dict(os.environ,{'SGLANG_HCU_MEGA_MOE_RUNTIME':'megamoe'}):O.set_mega_moe_cuda_graph_num_tokens(3);self.assertEqual(buf.item(),3)
            with patch.dict(os.environ,{'SGLANG_HCU_MEGA_MOE_RUNTIME':'invalid'}):
                with self.assertRaises(ValueError):O.set_mega_moe_cuda_graph_num_tokens(8)


    def test_capture_prepares_model_storage_before_metadata_and_recording(self):
        events=[];fb=S(lora_ids=None,global_num_tokens_gpu=None)
        obj=self.runner(model_runner=S(model=S(prepare_for_forward=lambda fb:events.append('prepare')),req_to_token_pool=S(mamba_pool=None)),backend=S(capture_one=lambda key,run,**kw:(events.append('capture'),run())),_capture_lora=False)
        obj.capture_prepare=lambda n:(fb,S());obj._init_forward_metadata_for_capture=lambda *a:events.append('metadata');obj._run_forward=lambda *a:events.append('forward')
        obj.capture_one_shape(4)
        self.assertEqual(events,['prepare','metadata','capture','forward'])

    def test_load_batch_carries_real_forward_record_encoder_fold_fields(self):
        fb=N.ForwardBatch(forward_mode=N.ForwardMode.EXTEND,batch_size=1,input_ids=torch.tensor([7,8,9]),req_pool_indices=torch.tensor([0]),seq_lens=torch.tensor([3]),out_cache_loc=torch.arange(3),seq_lens_sum=3,positions=torch.arange(3),extend_num_tokens=3,extend_seq_lens_cpu=[3],extend_prefix_lens_cpu=[0],encoder_swa_replay=True,encoder_swa_row_floor=torch.tensor([0,0,1]),encoder_swa_compress_rows=torch.tensor([0,2]))
        fb.kv_loc_plan=S(bind_replay=lambda target,reader,**kw:None)
        buffers={key:torch.zeros(4,dtype=torch.long) for key in ['input_ids','positions','out_cache_loc']}
        def fill(batch,**kw):
            for key,value in buffers.items():value[:3].copy_(getattr(batch,key))
        registry=S(fill_from=fill,has_slot=lambda key:key in buffers,get_slot=lambda key:S(slice_for=lambda bs,n:buffers[key][:n]))
        seen=[];obj=self.runner(model_runner=S(attn_tp_sequence_sharded=lambda n:False,prepare_dummy_forward_batch=lambda fb:fb,kv_index_translator=object()),backend=object(),buffer_registry=registry,_use_draft_input_embeds=False,static_draft_hidden_states=None,_prefill_static_buffers=None,capture_return_pooled_hidden_states=False)
        obj._prefill_logits_buffer_rows=lambda fb:0;obj._next_token_logits_buffer=lambda n:None;obj._shape_key=lambda *a:object();obj._prepare_forward_metadata_for_replay=lambda live,static,key:seen.append(static.encoder_swa_compress_rows)
        result=obj.load_batch(fb)
        self.assertTrue(result.encoder_swa_replay);self.assertIs(result.encoder_swa_row_floor,fb.encoder_swa_row_floor);self.assertIs(result.encoder_swa_compress_rows,fb.encoder_swa_compress_rows);self.assertEqual(seen,[fb.encoder_swa_compress_rows]);self.assertEqual(result.input_ids[:3].tolist(),[7,8,9])

    def test_body_replay_refreshes_v41_ids_and_megamoe_count_then_restores_forward(self):
        from hcu_sglang import prefill_graph_policy as P,mega_moe_graph as E
        from sglang.srt.layers.moe import utils as U
        from contextlib import contextmanager
        for failure in [False,True]:
            static=S(input_ids=torch.zeros(4,dtype=torch.long),positions=torch.arange(4));live=S(mm_input_embeds=None);original=lambda *a:None;layer=S(forward=original);events=[]
            def backend_replay(key,fb,**kw):
                self.assertEqual(fb.input_ids[:2].tolist(),[41,42]);self.assertEqual(P.get_prefill_raw_num_tokens(),2);events.append('replay');return torch.ones(4,2)
            def model_forward(*args,**kw):
                out=layer.forward(torch.tensor([41,42]))
                if failure:raise RuntimeError('eager tail')
                return out
            obj=self.runner(is_dsv41_model=True,layer_model=layer,buffer_registry=S(has_slot=lambda name:False),backend=S(replay=backend_replay),model_runner=S(pp_group=S(is_first_rank=True),model=S(forward=model_forward)),_use_draft_input_embeds=False)
            obj._restore_qwen_bcg_hc_output=lambda x:x
            obj._prefill_forward_context=lambda fb,**kw:P.prefill_graph_scope(full_graph=False,raw_num_tokens=kw['raw_num_tokens'])
            counter=torch.tensor([99],dtype=torch.int32)
            with patch.object(U,'get_moe_a2a_backend',return_value=S(is_megamoe=lambda:True)),patch.object(E,'_is_hcu',True),patch.dict(E.N._MEGA_MOE_SYMM_BUFFER,{'x':S(cuda_graph_num_tokens=counter)},clear=True),patch.dict(os.environ,{'SGLANG_HCU_MEGA_MOE_RUNTIME':'megamoe'}):
                if failure:
                    with self.assertRaisesRegex(RuntimeError,'eager tail'):obj._execute_body_capture(live,static,4,2,object())
                else:self.assertEqual(obj._execute_body_capture(live,static,4,2,object()).shape,(4,2))
            self.assertEqual(counter.item(),2);self.assertEqual(events,['replay']);self.assertIs(layer.forward,original);self.assertFalse(P.is_in_prefill_graph_scope())


if __name__=='__main__':unittest.main()
