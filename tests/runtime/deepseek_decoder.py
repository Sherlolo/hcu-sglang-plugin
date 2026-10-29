from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat

bootstrap_aiter_compat()

from sglang.srt.models import deepseek_v2 as S
from sglang.srt.plugins.hook_registry import HookRegistry
from hcu_sglang.models.deepseek.deepseek_v2 import hooks


class Attention(torch.nn.Module):
    def __init__(self, **options):
        super().__init__()
        self.options=options

    def prepare_qkv_latent(self, *args):
        return args

    def maybe_use_decode_attn_tp(self, batch):
        return nullcontext()

    def forward(self, *, hidden_states, **kwargs):
        return hidden_states+1


class Dense(torch.nn.Module):
    def __init__(self, **options):
        super().__init__()
        self.options=options

    def forward(self, value, batch, allocator):
        return value*2


class MoE(Dense):
    pass


class Boundary:
    input_on_attn_tp_slices=False

    def __init__(self, name, events):
        self.name=name
        self.events=events

    def prepare(self, value, batch, **kwargs):
        self.events.append(self.name+'.prepare')
        return value

    def finish(self, value, batch=None):
        self.events.append(self.name+'.finish')
        return value

    def exit(self, batch):
        return nullcontext(self)


class DecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_init=staticmethod(S.DeepseekV2DecoderLayer.__init__)
        cls.original_forward=staticmethod(S.DeepseekV2DecoderLayer.forward)
        HookRegistry.reset()
        with patch.object(hooks.attention,'register_hooks',return_value=0), \
             patch.object(hooks.mlp,'register_hooks',return_value=0), \
             patch.object(hooks.moe,'register_hooks',return_value=0):
            hooks.register_hooks()
        HookRegistry.apply_hooks()

    @classmethod
    def tearDownClass(cls):
        S.DeepseekV2DecoderLayer.__init__=cls.original_init
        S.DeepseekV2DecoderLayer.forward=cls.original_forward
        HookRegistry.reset()

    def make(self, *, skip_rope=False, build_stages=True, nextn=False, nope=False):
        config=SimpleNamespace(hidden_size=8,rope_parameters={'rope_theta':10000.,'rope_type':'default'},
                               max_position_embeddings=64,num_attention_heads=2,qk_nope_head_dim=2,
                               qk_rope_head_dim=2,v_head_dim=4,q_lora_rank=4,kv_lora_rank=4,
                               n_routed_experts=None,first_k_dense_replace=1,moe_layer_freq=1,
                               intermediate_size=16,hidden_act='silu',rms_norm_eps=1e-6,mla_nope=nope)
        events=[]
        boundaries=(Boundary('attn',events),Boundary('ffn',events))
        builds=[]
        def build(owner, **kwargs):
            builds.append(kwargs)
            return boundaries
        with ExitStack() as stack:
            stack.enter_context(patch.object(S,'DeepseekV2AttentionMLA',Attention))
            stack.enter_context(patch.object(S,'DeepseekV2MLP',Dense))
            stack.enter_context(patch.object(S,'DeepseekV2MoE',MoE))
            stack.enter_context(patch.object(S,'get_spec',return_value=SimpleNamespace(speculative_algorithm=None)))
            stack.enter_context(patch.object(S,'is_dense_ffn_fully_dp',return_value=False))
            stack.enter_context(patch.object(S,'_is_gfx95_supported',False))
            stack.enter_context(patch.object(S.DeepseekV2DecoderLayer,'_build_stages',build))
            layer=S.DeepseekV2DecoderLayer(config=config,layer_id=0,skip_rope=skip_rope,build_stages=build_stages,is_nextn=nextn)
        return layer,builds,events

    def test_current_constructor_keeps_skip_rope_and_native_stage_fields(self):
        layer,builds,_=self.make(skip_rope=True)
        self.assertTrue(layer.self_attn.options['skip_rope'])
        self.assertIs(builds[0]['input_layernorm'],layer.input_layernorm)
        self.assertIs(builds[0]['post_attention_layernorm'],layer.post_attention_layernorm)
        self.assertEqual(layer.mlp.options['parallel_group'],'tp')
        self.assertFalse(layer.mlp.options['reduce_results'])
        self.assertFalse(layer.mlp.options['allow_fused_down'])
        self.assertTrue(hasattr(layer,'attn_boundary'))
        self.assertTrue(hasattr(layer,'ffn_boundary'))
        self.assertFalse(hasattr(layer,'layer_communicator'))

    def test_build_stages_false_preserves_subclass_initialization_boundary(self):
        layer,builds,_=self.make(build_stages=False)
        self.assertFalse(builds)
        self.assertFalse(hasattr(layer,'attn_boundary'))
        self.assertFalse(layer.self_attn.options['skip_rope'])

    def test_nextn_nope_forces_skip_rope_and_preserves_moe_reduce_owner(self):
        layer,_,_=self.make(nextn=True,nope=True)
        self.assertTrue(layer.self_attn.options['skip_rope'])
        self.assertIsInstance(layer.mlp,MoE)
        self.assertFalse(layer.mlp.options['reduce_results'])

    def test_native_forward_consumes_current_boundaries(self):
        layer,_,events=self.make()
        value=torch.arange(16,dtype=torch.float32).reshape(2,8)
        with patch.object(S,'DeepseekV2MLP',Dense),patch.object(S,'DeepseekV2MoE',MoE), \
             patch.object(S,'get_attn_tp_context',return_value=SimpleNamespace(clear_attn_inputs=lambda:None)), \
             patch.object(S,'maybe_prefetch_next_full_attention_kv',return_value=None):
            output,indices=layer(torch.tensor([0,1]),value,SimpleNamespace(),zero_allocator=None)
        self.assertTrue(torch.equal(output,(value+1)*2))
        self.assertIsNone(indices)
        self.assertEqual(events,['attn.prepare','attn.finish','ffn.prepare','ffn.finish'])

    def test_upstream_owned_methods_are_not_replaced(self):
        self.assertIs(S.DeepseekV2DecoderLayer.__init__,self.original_init)
        self.assertIs(S.DeepseekV2DecoderLayer.forward,self.original_forward)
        self.assertFalse(any(name.startswith('sglang.srt.models.deepseek_v2.DeepseekV2DecoderLayer.') for name in HookRegistry._hooks))


if __name__=='__main__':
    unittest.main()
