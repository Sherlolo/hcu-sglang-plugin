from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers import row_fusion as R

class RowFusionTests(unittest.TestCase):
    def test_das_capability_gates_require_backend_and_consumer(self):
        with patch.object(R,'is_dcu',return_value=True):R.register_hooks()
        layer=object.__new__(R.N.RowParallelLinear);torch.nn.Module.__init__(layer)
        layer.quant_method=SimpleNamespace(supports_prequantized_input=True);layer.scheme=SimpleNamespace(supports_fp8_prequantized_input=True)
        with patch.object(R,'_use_fused_silu_mul_quant',True),patch.object(R,'_use_fused_silu_mul_fp8_quant',True),patch.object(R,'_lightop_fuse_silu_mul_clamp_quant',object()),patch.object(R,'_lightop_fuse_silu_mul_fp8_quant',object()):
            self.assertTrue(layer.supports_fused_silu_mul_quant_input());self.assertTrue(layer.supports_fused_silu_mul_fp8_quant_input())
            layer.scheme.supports_fp8_prequantized_input=False;self.assertFalse(R.supports_fused_silu_mul_fp8_quant_input(layer))
        with patch.object(R,'_use_fused_silu_mul_fp8_quant',False):self.assertFalse(R.supports_fused_silu_mul_fp8_quant_input(layer))

    def test_fused_fp8_carrier_reaches_quantizer_with_output_dtype(self):
        x=torch.tensor([[1.,2.,3.,4.]],dtype=torch.bfloat16);calls=[]
        def quant(inp,*,fp8type,output,scales,limit):
            self.assertIs(inp,x);self.assertEqual(limit,3.);self.assertEqual(fp8type,0);output.fill_(2);scales.fill_(.25)
        def apply(layer,carrier,*,bias):
            q,scale,dtype=carrier;self.assertEqual(q.dtype,torch.float8_e4m3fn);self.assertEqual(dtype,x.dtype);calls.append(carrier)
            return (q.float()*scale).to(dtype)
        layer=SimpleNamespace(tp_group=None,input_is_parallel=True,skip_bias_add=False,bias=None,use_dp_attention_reduce=False,reduce_results=False,use_decode_attn_tp=False,quant_method=SimpleNamespace(apply=apply),supports_fused_silu_mul_fp8_quant_input=lambda:True)
        with patch.object(R.N,'get_parallel',return_value=SimpleNamespace(tp_group=None)),patch.object(R.N,'get_group_rank_size',return_value=(0,1)),patch.object(R.N,'get_forward',return_value=SimpleNamespace(sp_active=False)),patch.object(R.N,'use_symmetric_memory',return_value=nullcontext()),patch.object(R.N,'is_allocation_symmetric',return_value=False),patch.object(R,'_lightop_fuse_silu_mul_fp8_quant',side_effect=quant):
            y,bias=R.forward(layer,x,use_fused_silu_mul_fp8_quant=True,swiglu_limit=3.)
        self.assertEqual(len(calls),1);self.assertIsNone(bias);torch.testing.assert_close(y,torch.full((1,2),.5,dtype=x.dtype))

    def test_default_and_non_hcu_delegate_existing_forward(self):
        marker=object();layer=object();x=object()
        def original(owner,value,**kwargs):self.assertIs(owner,layer);self.assertIs(value,x);return marker
        with patch.object(R,'is_dcu',return_value=True):self.assertIs(R.around_row_forward(original,layer,x),marker)
        with patch.object(R,'is_dcu',return_value=False):self.assertIs(R.around_row_forward(original,layer,x,use_fused_silu_mul_fp8_quant=True),marker)

if __name__=='__main__':unittest.main()
