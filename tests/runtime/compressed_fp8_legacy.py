from __future__ import annotations
import unittest
from unittest.mock import patch
import torch
from compressed_tensors.quantization import QuantizationStrategy
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import compressed_fp8 as H
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsLinearMethod
from sglang.srt.layers.quantization.compressed_tensors.schemes import compressed_tensors_w8a8_int8 as I
from sglang.srt.plugins.hook_registry import HookRegistry


class NativeLegacySchemeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();H.register_hooks();HookRegistry.apply_hooks()

    def run_scheme(self,static):
        method=CompressedTensorsLinearMethod(None);l=torch.nn.Module()
        l.scheme=I.CompressedTensorsW8A8Int8(QuantizationStrategy.CHANNEL,static,True)
        method.create_weights(l,4,[3],4,3,torch.bfloat16,weight_loader=lambda *a,**k:None)
        w=torch.tensor([[1,2,3,4],[2,3,4,1],[4,3,2,1]],dtype=torch.int8)
        ws=torch.tensor([[.25],[.5],[1.]])
        l.weight.data.copy_(w);l.weight_scale.data.copy_(ws)
        if static:l.input_scale.data.fill_(.25)
        method.process_weights_after_loading(l)
        q=torch.tensor([[1,2,3,4],[4,3,2,1]],dtype=torch.int8);s=torch.tensor([[.25],[.5]])
        x=torch.ones(2,4,dtype=torch.bfloat16);bias=torch.ones(3,dtype=x.dtype)
        def gemm(q_in,weight,scale,weight_scale,out_dtype,bias):
            self.assertIs(q_in,q);self.assertIs(weight,l.weight);self.assertIs(scale,s);self.assertIs(weight_scale,l.weight_scale)
            out=((q_in.float()*scale)@(weight.float()*weight_scale.T))+bias
            return out.to(out_dtype)
        with patch.object(I,'per_token_quant_int8',return_value=(q,s)) as quant,patch.object(I,'int8_scaled_mm',side_effect=gemm,create=True):
            out=method.apply(l,x,bias,input_quant_args=[torch.ones(1)],silu_quant_args=[torch.zeros(1)])
        quant.assert_called_once_with(x)
        torch.testing.assert_close(out,(((q.float()*s)@(w.float()*ws).T)+bias).to(x.dtype))
        self.assertEqual(l.weight.shape,(4,3));self.assertNotIn('input_quant_args',l.scheme._apply_weights_supported_kwargs)
        self.assertNotIn('silu_quant_args',l.scheme._apply_weights_supported_kwargs)
        return l

    def test_real_native_dynamic_int8_scheme_through_source_wrapper(self):
        l=self.run_scheme(False);self.assertIsNone(l.input_scale)

    def test_real_native_static_int8_scheme_retains_legacy_contract(self):
        l=self.run_scheme(True);self.assertEqual(l.input_scale.item(),.25)


if __name__=='__main__':unittest.main()
