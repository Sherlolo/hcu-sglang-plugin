from types import SimpleNamespace
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.models.deepseek.deepseek_common.methods import weight_loader as W

class MLAInt8ScaleTests(unittest.TestCase):
    def test_channel_scale_with_explicit_none_block_size(self):
        weight=torch.arange(32,dtype=torch.int8).reshape(8,4)
        scale=torch.arange(1,9,dtype=torch.float32).reshape(8,1)/128
        attn=SimpleNamespace(kv_b_proj=SimpleNamespace(weight=weight,weight_scale=scale),qk_nope_head_dim=2,v_head_dim=2,w_kc=None,w_vc=None,w_scale=None)
        owner=SimpleNamespace(config=SimpleNamespace(num_hidden_layers=1,architectures=['GlmMoeDsaForCausalLM']),quant_config=SimpleNamespace(weight_block_size=None,get_name=lambda:'slimquant_w4a8_marlin'),model=SimpleNamespace(start_layer=0,end_layer=1,layers=[SimpleNamespace(self_attn=attn)]))
        expected=weight.bfloat16()*scale.bfloat16();kc,vc=expected.unflatten(0,(-1,4)).split([2,2],dim=1)
        W.post_load_weights(owner)
        self.assertEqual(attn.w_kc.dtype,torch.bfloat16)
        torch.testing.assert_close(attn.w_kc,kc,rtol=0,atol=0)
        torch.testing.assert_close(attn.w_vc,vc.transpose(1,2),rtol=0,atol=0)

if __name__=='__main__':unittest.main()
