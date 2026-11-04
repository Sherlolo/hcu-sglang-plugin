from __future__ import annotations
from contextlib import nullcontext
from types import SimpleNamespace as S
import os,unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


@unittest.skipUnless(os.environ.get('HCU_PLUGIN_RUN_GPU_TESTS')=='1','explicit GPU opt-in required')
class PrefillGraphGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def test_public_load_body_replay_and_draft_trim_with_changed_prepared_ids(self):
        from sglang.srt.model_executor.runner import prefill_cuda_graph_runner as N
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.layers.moe import utils as U
        from hcu_sglang import prefill_graph_policy as P
        obj=object.__new__(N.PrefillCudaGraphRunner)
        buffers={key:torch.zeros(4,dtype=torch.long,device='cuda') for key in ['input_ids','positions','out_cache_loc']}
        hs=torch.empty((4,2),dtype=torch.float32,device='cuda')
        def kernel():hs.copy_(buffers['input_ids'][:,None].expand(4,2)*2)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):kernel()
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):kernel()
        original=lambda *a:None;layer=S(forward=original);events=[]
        def fill(batch,**kwargs):
            for key,buffer in buffers.items():buffer[:3].copy_(getattr(batch,key))
        def replay(key,fb,**kwargs):
            self.assertEqual(P.get_prefill_raw_num_tokens(),3);self.assertTrue(P.is_in_prefill_graph_scope());graph.replay();return hs
        def model_forward(ids,positions,fb,**kwargs):
            result=layer.forward(ids+100)
            return LogitsProcessorOutput(None,hidden_states=result,draft_topk_index=buffers['input_ids'][:,None])
        mr=S(attn_backend=S(),spec_algorithm=S(is_speculative=lambda:False),pp_group=S(is_first_rank=True),model=S(forward=model_forward),attn_tp_sequence_sharded=lambda n:False,prepare_dummy_forward_batch=lambda fb:fb,kv_index_translator=object())
        attrs=dict(prefill_backend_name=N.Backend.BREAKABLE,_is_full_backend=False,_use_draft_input_embeds=False,is_dsv41_model=True,model_runner=mr,layer_model=layer,device_module=torch.cuda,capture_num_tokens=[4],max_context_size=None,enable_cp_bcg_capture=False,static_draft_hidden_states=None,_prefill_static_buffers=None,capture_return_pooled_hidden_states=False,has_mha_companion_layers=False,capture_hidden_mode=N.CaptureHiddenMode.NULL,attention_layers=[],moe_layers=[],moe_fusions=[],dsa_indexers=[],mha_companion_layers=[],quant_config=None,backend=S(replay_session=lambda:nullcontext(),replay=replay),buffer_registry=S(fill_from=fill,has_slot=lambda key:key in buffers,get_slot=lambda key:S(slice_for=lambda bs,n:buffers[key][:n])))
        for key,value in attrs.items():setattr(obj,key,value)
        obj._prefill_logits_buffer_rows=lambda fb:0;obj._next_token_logits_buffer=lambda n:None;obj._shape_key=lambda *a:S(variant_label=None);obj._prepare_forward_metadata_for_replay=lambda live,static,key:events.append(static.encoder_swa_row_floor);obj._restore_qwen_bcg_hc_output=lambda x:x
        with patch.object(N,'set_tc_piecewise_forward_context',return_value=nullcontext()),patch.object(N,'maybe_publish_prefill_shared_read_done'),patch.object(U,'get_moe_a2a_backend',return_value=S(is_megamoe=lambda:False)):
            for ids in [[4,5,6],[9,8,7]]:
                x=torch.tensor(ids,device='cuda');fb=N.ForwardBatch(forward_mode=N.ForwardMode.EXTEND,batch_size=1,input_ids=x,req_pool_indices=torch.tensor([0],device='cuda'),seq_lens=torch.tensor([3],device='cuda'),out_cache_loc=torch.arange(3,device='cuda'),seq_lens_sum=3,positions=torch.arange(3,device='cuda'),extend_num_tokens=3,capture_hidden_mode=N.CaptureHiddenMode.NULL,encoder_swa_row_floor=torch.tensor([0,1,2],device='cuda'))
                fb.kv_loc_plan=S(bind_replay=lambda target,reader,**kw:None)
                output=obj.execute(fb);torch.cuda.synchronize()
                torch.testing.assert_close(output.hidden_states,(x+100)[:,None].expand(3,2).float()*2,rtol=0,atol=0)
                torch.testing.assert_close(output.draft_topk_index,(x+100)[:,None],rtol=0,atol=0)
                self.assertIs(layer.forward,original);self.assertFalse(P.is_in_prefill_graph_scope());self.assertIs(events[-1],fb.encoder_swa_row_floor)


if __name__=='__main__':unittest.main()
