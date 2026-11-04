from __future__ import annotations

from unittest.mock import Mock,patch
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import fp8 as Q,fp8_fallback as C
from sglang.srt.plugins.hook_registry import HookRegistry


class Fp8FallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset();Q.register_hooks();HookRegistry.apply_hooks()

    def setUp(self):
        C.TORCH_DEVICE_IDENTITY=None;C._HIPBLASLT_UNSUPPORTED_FP8_SHAPES.clear()
        self.q=torch.tensor([[1.,2.,3.,4.],[2.,1.,0.,3.],[1.,1.,1.,1.],[2.,2.,2.,2.]]).to(torch.float8_e4m3fn)
        self.w=torch.arange(12).reshape(4,3).float().to(torch.float8_e4m3fn)
        self.a=torch.tensor([[.25],[.5],[.75],[1.]])
        self.b=torch.tensor([[.5],[.125],[.25]])
        self.bias=torch.tensor([1.,2.,3.])

    def call(self,q=None,dtype=torch.bfloat16):
        value=self.q if q is None else q
        return C.N._apply_fallback_scaled_mm(value,self.w,self.a,self.b,(2,4),(1,2,3),self.bias,dtype)

    def raw(self,q,w,**kwargs):
        self.assertIs(kwargs['scale_a'],C.TORCH_DEVICE_IDENTITY);self.assertIs(kwargs['scale_b'],C.TORCH_DEVICE_IDENTITY)
        self.assertEqual(kwargs['out_dtype'],torch.float32)
        return q.float()@w.float()

    def triton(self,q,w,a,b,dtype,bias):
        return ((q.float()*a)@(w.float()*b.T)+bias).to(dtype)

    def test_public_success_scales_once_trims_padding_and_preserves_bias(self):
        with patch.object(C.torch,'_scaled_mm',side_effect=self.raw) as probe,patch.object(C,'triton_scaled_mm',side_effect=AssertionError('unexpected fallback')):
            result=self.call()
        expected=(((self.q.float()@self.w.float())[:2]*self.a[:2])*self.b.T+self.bias).to(torch.bfloat16).reshape(1,2,3)
        torch.testing.assert_close(result,expected);self.assertEqual(probe.call_count,1)
        self.assertFalse(C._HIPBLASLT_UNSUPPORTED_FP8_SHAPES)

    def test_blaslt_error_is_probed_once_then_reuses_real_scale_contract(self):
        with patch.object(C.torch,'_scaled_mm',side_effect=RuntimeError('hipBLASLt unsupported shape')) as probe,patch.object(C,'triton_scaled_mm',side_effect=self.triton) as fallback:
            first=self.call();second=self.call()
        self.assertEqual(probe.call_count,1);self.assertEqual(fallback.call_count,2);self.assertEqual(len(C._HIPBLASLT_UNSUPPORTED_FP8_SHAPES),1)
        expected=self.triton(self.q,self.w,self.a,self.b,torch.bfloat16,self.bias)[:2].reshape(1,2,3)
        torch.testing.assert_close(first,expected);torch.testing.assert_close(second,expected)
        self.assertIs(fallback.call_args.args[2],self.a);self.assertIs(fallback.call_args.args[3],self.b)

    def test_cache_distinguishes_input_and_output_dtype(self):
        with patch.object(C.torch,'_scaled_mm',side_effect=RuntimeError('BLASLT no solution')) as probe,patch.object(C,'triton_scaled_mm',side_effect=self.triton):
            self.call();self.call(dtype=torch.float16)
            q=self.q.to(torch.float8_e4m3fnuz);self.call(q=q)
            self.call()
        self.assertEqual(probe.call_count,3);self.assertEqual(len(C._HIPBLASLT_UNSUPPORTED_FP8_SHAPES),3)

    def test_non_blaslt_runtime_error_propagates_without_poisoning_cache(self):
        error=RuntimeError('invalid memory access')
        with patch.object(C.torch,'_scaled_mm',side_effect=error),patch.object(C,'triton_scaled_mm',side_effect=AssertionError('error was swallowed')):
            with self.assertRaises(RuntimeError) as caught:self.call()
        self.assertIs(caught.exception,error);self.assertFalse(C._HIPBLASLT_UNSUPPORTED_FP8_SHAPES)

    def test_non_runtime_backend_exception_is_not_reclassified(self):
        error=TypeError('bad dtype ABI')
        with patch.object(C.torch,'_scaled_mm',side_effect=error):
            with self.assertRaises(TypeError) as caught:self.call()
        self.assertIs(caught.exception,error);self.assertFalse(C._HIPBLASLT_UNSUPPORTED_FP8_SHAPES)

    def test_multidimensional_output_scales_in_flattened_rows_before_reshape(self):
        with patch.object(C.torch,'_scaled_mm',side_effect=self.raw):
            result=C.N._apply_fallback_scaled_mm(self.q,self.w,self.a,self.b,(4,4),(2,2,3),self.bias,torch.bfloat16)
        expected=(((self.q.float()@self.w.float())*self.a)*self.b.T+self.bias).to(torch.bfloat16).reshape(2,2,3)
        self.assertEqual(result.shape,(2,2,3));torch.testing.assert_close(result,expected)

    def test_legacy_tuple_output_and_effective_non_hcu_original(self):
        with patch.object(C.torch,'_scaled_mm',side_effect=lambda q,w,**kwargs:(q.float()@w.float(),torch.tensor(0.))):
            result=self.call()
        self.assertEqual(result.shape,(1,2,3))
        marker=object();original=Mock(return_value=marker)
        with patch('hcu_sglang.utils.detection.is_hcu',return_value=False):
            self.assertIs(Q.around_fallback_scaled_mm(original,self.q,self.w,self.a,self.b,(2,4),(1,2,3),self.bias,torch.bfloat16),marker)
        original.assert_called_once()


if __name__=='__main__':unittest.main()
