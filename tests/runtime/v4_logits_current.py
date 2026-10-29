from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.models.deepseek.deepseek_v4.hooks import causal_lm as H
from hcu_sglang.models.deepseek.current_v4.execution import core as C
from sglang.srt.plugins.hook_registry import HookRegistry
import sglang.srt.models.deepseek_v4 as N

class V4LogitsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();H.register_hooks();HookRegistry.apply_hooks()

    def run_case(self,pd):
        ids=torch.tensor([1,2]);positions=torch.tensor([0,1]);h=torch.ones(2,4);pre=torch.full((2,8),2.);aux=torch.arange(24.).reshape(2,3,4);captured=[torch.ones(2,4)]
        fb=SimpleNamespace(pd_hidden_capture_layer_ids=[1] if pd else None)
        class Body(torch.nn.Module):
            late_layer_start=None
            dspark_layers_to_capture=[1] if pd else None
            def forward(owner,*args):return ((h,pre),captured) if pd else (h,pre)
        layer=object.__new__(N.DeepseekV4ForCausalLM);torch.nn.Module.__init__(layer);layer.dsa_enable_prefill_cp=False;layer.pp_group=SimpleNamespace(is_last_rank=True);layer.model=Body();layer.capture_aux_hidden_states=pd;layer.lm_head=torch.nn.Identity();calls=[]
        def logits(*args,**kwargs):calls.append((args,kwargs));return SimpleNamespace(hidden_states=None)
        layer.logits_processor=logits
        with patch.object(C,'get_attn_tp_context',return_value=SimpleNamespace(maybe_input_scattered=lambda batch:nullcontext())),patch.object(C,'pack_aux_hidden_states',return_value=aux):out=layer.forward(ids,positions,fb)
        self.assertEqual(len(calls),1);self.assertIs(calls[0][0][1],h)
        if pd:
            self.assertIs(calls[0][0][4],aux);self.assertIsNone(calls[0][1]['hidden_states_before_norm']);torch.testing.assert_close(out.hidden_states,aux.flatten(1))
        else:self.assertIs(calls[0][1]['hidden_states_before_norm'],pre)

    def test_public_forward_consumes_pre_hc_head(self):self.run_case(False)
    def test_public_forward_consumes_pd_aux_and_flattens_output(self):self.run_case(True)

if __name__=='__main__':unittest.main()
