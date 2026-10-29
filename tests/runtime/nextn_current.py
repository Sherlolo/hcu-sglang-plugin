from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import inspect,unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.models import deepseek_nextn as S
from sglang.srt.plugins.hook_registry import HookRegistry
from hcu_sglang.models.deepseek.deepseek_nextn.hooks import model as hooks
from hcu_sglang.models.deepseek.deepseek_nextn.methods import model as plugin

class Norm(torch.nn.Module):
    def __init__(self,size,eps):super().__init__();self.weight=torch.ones(size);self.variance_epsilon=eps
    def forward(self,x):return x

class NextNCurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_init=staticmethod(S.DeepseekModelNextN.__init__);cls.original_forward=staticmethod(S.DeepseekModelNextN.forward)
        cls.decoder_signature=inspect.signature(S.DeepseekV2DecoderLayer)
        HookRegistry.reset();hooks.register_hooks();HookRegistry.apply_hooks()
    @classmethod
    def tearDownClass(cls):
        S.DeepseekModelNextN.__init__=cls.original_init;S.DeepseekModelNextN.forward=cls.original_forward;HookRegistry.reset()
    def make(self,*,sbo=False,nope=False):
        config=SimpleNamespace(vocab_size=16,hidden_size=8,rms_norm_eps=1e-6,qk_rope_head_dim=0 if nope else 4)
        calls=[]
        signature=self.decoder_signature
        class Decoder(torch.nn.Module):
            def __init__(self,*args,**kwargs):
                super().__init__();signature.bind(*args,**kwargs);calls.append(kwargs);self.layer_id=0
            def forward(self,positions,hidden_states,forward_batch,zero_allocator,prev_topk_indices=None):
                return hidden_states+1,None
        def embed(vocab,size,**kwargs):return torch.nn.Embedding(vocab,size)
        with ExitStack() as stack:
            for name,value in [('DeepseekV2DecoderLayer',Decoder),('VocabParallelEmbedding',embed),('RMSNorm',Norm),('get_embedding_tp_kwargs',lambda:{}),('enable_nextn_moe_bf16_cast_to_fp8',lambda quant:False),('_is_cuda',False),('_is_npu',False)]:stack.enter_context(patch.object(S,name,value))
            stack.enter_context(patch.object(plugin.M,'is_sbo_enabled',return_value=sbo))
            stack.enter_context(patch.object(S.envs.SGLANG_NPU_USE_MULTI_STREAM,'get',return_value=False))
            stream=object();stack.enter_context(patch.object(torch.cuda,'Stream',return_value=stream))
            layer=S.DeepseekModelNextN(config)
        return layer,calls,stream
    def test_current_constructor_uses_stage_decoder_contract(self):
        layer,calls,_=self.make(nope=True)
        self.assertTrue(calls[0]['skip_rope']);self.assertTrue(calls[0]['is_nextn'])
        self.assertNotIn('dsa_enable_prefill_cp',calls[0]);self.assertNotIn('mla_enable_prefill_cp',calls[0])
        self.assertFalse(hasattr(layer,'dsa_enable_prefill_cp'))
    def test_das_sbo_stream_reaches_decoder(self):
        layer,calls,stream=self.make(sbo=True)
        self.assertIs(layer.alt_stream,stream);self.assertIs(calls[0]['alt_stream'],stream)
    def test_forward_uses_two_value_stage_result_and_residual_boundaries(self):
        layer,_,_=self.make();layer.eh_proj.weight.data.fill_(1.)
        x=torch.ones(2,8);ids=torch.tensor([1,2]);positions=torch.tensor([0,1]);events=[]
        batch=SimpleNamespace(attn_cp_metadata=None,spec_info=SimpleNamespace(hidden_states=torch.zeros_like(x)),forward_mode=SimpleNamespace(is_idle=lambda:False))
        state=SimpleNamespace(topk_indices=None,should_publish=False,update=lambda value:events.append('update'),publish=lambda:events.append('publish'))
        residual=SimpleNamespace(start=lambda batch:events.append('start'),complete_output=lambda x,batch:(events.append('complete') or x),final_norm=lambda x,batch,norm:(events.append('norm') or norm(x)))
        with ExitStack() as stack:
            stack.enter_context(patch('sglang.srt.model_executor.forward_context.get_token_to_kv_pool',return_value=object()))
            stack.enter_context(patch.object(plugin,'enable_cp_v2',return_value=False))
            stack.enter_context(patch.object(S,'get_parallel',return_value=SimpleNamespace(enable_dsa_cache_layer_split=False)))
            stack.enter_context(patch.object(plugin.D,'dsa_use_prefill_cp',return_value=False))
            stack.enter_context(patch.object(plugin.C,'get_parallel',return_value=SimpleNamespace(enable_prefill_cp=False)))
            stack.enter_context(patch.object(S,'residual_batch',residual))
            stack.enter_context(patch.object(S.IndexTopKShareState,'from_mtp_carry',return_value=state))
            stack.enter_context(patch.object(S,'get_global_expert_distribution_recorder',return_value=SimpleNamespace(disable_this_region=nullcontext)))
            out=layer(ids,positions,batch,input_embeds=x)
        self.assertTrue(torch.equal(out,torch.full_like(x,9.)))
        self.assertEqual(events,['start','complete','norm','update','publish'])

if __name__=='__main__':unittest.main()
