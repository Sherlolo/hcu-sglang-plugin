from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch,Mock
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import fp8_linear as C,w8a8_fp8 as H,fp8_quant as F
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.srt.layers.quantization import fp8 as P
from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear as early_alias


class Fp8LinearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset();H.register_hooks();HookRegistry.apply_hooks()

    def setUp(self):
        self.q=torch.tensor([[1.,2.,3.,4.],[2.,3.,1.,0.]]).to(torch.float8_e4m3fn)
        self.s=torch.tensor([[.25],[.5]])
        self.w=torch.arange(12).reshape(4,3).float().to(torch.float8_e4m3fn)
        self.ws=torch.tensor([[.5],[.125],[.25]])
        self.bias=torch.tensor([.5,1.,1.5])
        self.calls=[]

    def gemm(self,a,b,out):
        q,s=a;w,ws=b;self.calls.append((a,b,out))
        result=(q.float()*s)@(w.float()*ws.T)
        self.assertEqual(tuple(out.shape),tuple(result.shape))
        out.copy_(result)

    def method(self):
        method=object.__new__(Fp8LinearMethod)
        method.use_marlin=False;method.block_fp8_as_mxfp8=False;method.use_mxfp8=False;method.block_quant=False;method.cutlass_fp8_supported=False;method.use_per_token_if_dynamic=False
        return method

    def test_native_public_consumer_reuses_fp8_scale_and_output_dtype(self):
        layer=SimpleNamespace(weight=self.w,weight_scale=self.ws,input_scale=None)
        self.assertIs(P.apply_fp8_linear,early_alias)
        with patch.object(C.deepgemm,'fp8_gemm',side_effect=self.gemm),patch.object(C.N,'scaled_fp8_quant',side_effect=AssertionError('requantized')),patch.object(C.N,'per_token_group_quant_fp8',side_effect=AssertionError('requantized')):
            output=self.method().apply(layer,(self.q,self.s,torch.float16),self.bias)
        self.assertEqual(output.dtype,torch.float16);self.assertEqual(self.calls[0][0][0].data_ptr(),self.q.data_ptr());self.assertIs(self.calls[0][0][1],self.s)
        expected=((self.q.float()*self.s)@(self.w.float()*self.ws.T)).to(torch.float16)+self.bias
        torch.testing.assert_close(output,expected.to(torch.float16))

    def test_direct_carrier_calls_source_deepgemm_without_quantization(self):
        with patch.object(C.deepgemm,'fp8_gemm',side_effect=self.gemm):
            output=C.N.apply_fp8_linear((self.q,self.s),self.w,self.ws,bias=self.bias,cutlass_fp8_supported=False)
        self.assertEqual(output.dtype,torch.bfloat16);self.assertEqual(self.calls[0][0][0].data_ptr(),self.q.data_ptr())
        expected=((self.q.float()*self.s)@(self.w.float()*self.ws.T)).to(torch.bfloat16)
        torch.testing.assert_close(output,(expected+self.bias).to(torch.bfloat16))

    def test_dynamic_tensor_uses_source_per_token_group_quant(self):
        x=torch.zeros(2,4,dtype=torch.bfloat16)
        with patch.object(C.N,'per_token_group_quant_fp8',return_value=(self.q,self.s)) as quant,patch.object(C.deepgemm,'fp8_gemm',side_effect=self.gemm):
            output=C.N.apply_fp8_linear(x,self.w,self.ws,bias=self.bias,cutlass_fp8_supported=False)
        self.assertEqual(quant.call_count,1);self.assertEqual(quant.call_args.args[0].data_ptr(),x.data_ptr());self.assertEqual(quant.call_args.kwargs,{'group_size':4});self.assertEqual(output.shape,(2,3))

    def test_static_tensor_scale_uses_source_static_quant(self):
        x=torch.zeros(2,4,dtype=torch.bfloat16);scale=torch.tensor([.25])
        with patch.object(C.N,'static_quant_fp8',return_value=(self.q,scale)) as quant,patch.object(C.deepgemm,'fp8_gemm',side_effect=self.gemm):
            C.N.apply_fp8_linear(x,self.w,self.ws,input_scale=scale,cutlass_fp8_supported=False)
        self.assertEqual(quant.call_count,1);self.assertEqual(quant.call_args.args[0].data_ptr(),x.data_ptr());self.assertIs(quant.call_args.args[1],scale);self.assertEqual(quant.call_args.kwargs,{'repeat_scale':False})

    def test_prequantized_scale_is_required_and_rejected_before_backend(self):
        with patch.object(C.deepgemm,'fp8_gemm') as gemm:
            with self.assertRaises(AssertionError):C.N.apply_fp8_linear(self.q,self.w,self.ws,cutlass_fp8_supported=False)
        gemm.assert_not_called()

    def test_compressed_path_preserves_quant_padding_arguments(self):
        x=torch.zeros(2,4,dtype=torch.bfloat16)
        with patch.object(C.N,'scaled_fp8_quant',return_value=(self.q,self.s)) as quant,patch.object(C.deepgemm,'fp8_gemm',side_effect=self.gemm):
            C.N.apply_fp8_linear(x,self.w,self.ws,compressed_tensor_quant=True,use_per_token_if_dynamic=True,cutlass_fp8_supported=False)
        self.assertEqual(quant.call_args.kwargs['num_token_padding'],17);self.assertTrue(quant.call_args.kwargs['use_per_token_if_dynamic'])

    def test_padded_quantizer_output_is_allocated_then_trimmed_for_backend(self):
        x=torch.zeros(2,4,dtype=torch.bfloat16);padded_q=torch.cat([self.q,torch.zeros(15,4).to(self.q.dtype)]);padded_scale=torch.cat([self.s,torch.ones(15,1)])
        with patch.object(C.N,'scaled_fp8_quant',return_value=(padded_q,padded_scale)),patch.object(C.deepgemm,'fp8_gemm',side_effect=self.gemm):
            output=C.N.apply_fp8_linear(x,self.w,self.ws,bias=self.bias,compressed_tensor_quant=True,cutlass_fp8_supported=False)
        self.assertEqual(self.calls[0][2].shape,(17,3));self.assertEqual(output.shape,(2,3))
        expected=((self.q.float()*self.s)@(self.w.float()*self.ws.T)).to(torch.bfloat16)
        torch.testing.assert_close(output,(expected+self.bias).to(torch.bfloat16))

    def test_multidimensional_carrier_is_flattened_for_real_backend_contract(self):
        q=self.q.view(1,2,4);scale=self.s.view(1,2,1)
        with patch.object(C.deepgemm,'fp8_gemm',side_effect=self.gemm):
            output=C.N.apply_fp8_linear((q,scale),self.w,self.ws,cutlass_fp8_supported=False)
        self.assertEqual(self.calls[0][0][0].shape,(2,4));self.assertEqual(self.calls[0][2].shape,(2,3));self.assertEqual(output.shape,(1,2,3))
        self.assertEqual(self.calls[0][0][0].data_ptr(),q.data_ptr());self.assertEqual(self.calls[0][0][1].data_ptr(),scale.data_ptr())

    def test_hcu_quantizer_returns_matched_unpadded_q_scale_and_contiguous_input(self):
        x=torch.arange(16,dtype=torch.bfloat16).reshape(2,8)[:,::2]
        def quant(value,*,dtype,out_q,out_scale):
            self.assertTrue(value.is_contiguous());self.assertEqual(out_q.shape,value.shape);self.assertEqual(out_scale.shape,(2,1))
            scale=value.float().abs().amax(1,keepdim=True).clamp_min(1e-12)/448
            out_scale.copy_(scale);out_q.copy_((value.float()/scale).clamp(-448,448).to(dtype));return out_q,out_scale
        with patch.object(F,'lightop_per_token_quant_fp8',side_effect=quant) as producer:
            q,scale=C.N.scaled_fp8_quant(x,None,num_token_padding=17,use_per_token_if_dynamic=True)
        self.assertEqual(q.shape,(2,4));self.assertEqual(scale.shape,(2,1));self.assertEqual(producer.call_count,1)

    def test_source_tensor_static_producers_write_only_logical_padded_rows(self):
        x=torch.arange(8,dtype=torch.float32).reshape(2,4)
        for supplied in (None,torch.tensor([.015625])):
            q,scale=C.N.scaled_fp8_quant(x,supplied,num_token_padding=17,use_per_token_if_dynamic=False)
            self.assertEqual(q.shape,(17,4));self.assertEqual(scale.numel(),1)
            torch.testing.assert_close(q[:2].float()*scale,x)
            if supplied is not None:self.assertIs(scale,supplied)

    def test_source_quantizer_rejects_bad_dims_scale_and_preserves_other_platform(self):
        with self.assertRaises(AssertionError):C.N.scaled_fp8_quant(torch.ones(1,2,4))
        with self.assertRaises(AssertionError):C.N.scaled_fp8_quant(torch.ones(2,4),torch.ones(2))
        marker=object();original=Mock(return_value=marker)
        with patch.object(F,'is_hcu',return_value=False):self.assertIs(F.around_scaled_fp8_quant(original,'input',setting=True),marker)
        original.assert_called_once_with('input',setting=True)

    def test_backend_failure_propagates_and_non_hcu_retains_original(self):
        error=RuntimeError('dense gemm failure')
        with patch.object(C.deepgemm,'fp8_gemm',side_effect=error):
            with self.assertRaises(RuntimeError) as caught:C.N.apply_fp8_linear(self.q,self.w,self.ws,input_scale=self.s,cutlass_fp8_supported=False)
        self.assertIs(caught.exception,error)
        marker=object();original=Mock(return_value=marker)
        with patch.object(C,'is_hcu',return_value=False):self.assertIs(C.around_apply_fp8_linear(original,'input',setting=True),marker)
        original.assert_called_once_with('input',setting=True)


if __name__=='__main__':unittest.main()
