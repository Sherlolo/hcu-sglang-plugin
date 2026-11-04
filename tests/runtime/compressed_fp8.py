from __future__ import annotations
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import compressed_fp8 as C, fp8_linear as L, w8a8_fp8 as H, fp8_block as B
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_fp8 import CompressedTensorsW8A8Fp8
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsLinearMethod


class CompressedFp8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset();H.register_hooks();HookRegistry.apply_hooks()

    def scheme(self,static=False,strategy='channel'):
        return CompressedTensorsW8A8Fp8(QuantizationArgs(num_bits=8,type='float',strategy=strategy),static)

    def layer(self,scheme):
        l=torch.nn.Module()
        scheme.create_weights(l,4,[3],4,3,torch.bfloat16,lambda *a,**k:None)
        l.weight.data.copy_(torch.tensor([[1.,2.,3.,4.],[2.,1.,0.,3.],[3.,2.,1.,0.]]).to(torch.float8_e4m3fn))
        l.weight_scale.data.copy_(torch.tensor([[.5],[.25],[.125]]) if scheme.strategy==QuantizationStrategy.CHANNEL else torch.ones_like(l.weight_scale))
        if scheme.is_static_input_scheme:l.input_scale.data.fill_(.25)
        return l

    def setUp(self):
        self.q=torch.tensor([[1.,2.,3.,4.],[2.,1.,0.,3.]]).to(torch.float8_e4m3fn)
        self.scale=torch.tensor([[.25],[.5]])
        self.bias=torch.tensor([.5,1.,1.5])

    def vendor(self,q,w,s,ws,m,n,k,layout,dtype,bias):
        self.assertEqual((m,n,k,layout),(2,3,4,'NT'))
        out=(q.float()*s)@(w.float()*ws).T
        if bias is not None:out+=bias
        return 0,out.to(dtype)

    def test_native_creation_and_optin_postload_preserve_ocp_nk(self):
        m=self.scheme();l=self.layer(m);old=l.weight.float().clone()
        with patch.object(C.Q,'is_fp8_fnuz',return_value=True),patch.object(C.Q,'normalize_e4m3fn_to_e4m3fnuz',side_effect=AssertionError('normalize')):
            m.process_weights_after_loading(l)
        self.assertEqual(l.weight.shape,(3,4));self.assertEqual(l.weight.dtype,torch.float8_e4m3fn);self.assertTrue(l.weight.is_contiguous())
        self.assertIsInstance(l.weight,torch.nn.Parameter);self.assertFalse(l.weight.requires_grad)
        torch.testing.assert_close(l.weight.float(),old);self.assertIsNone(l.input_scale)

    def test_capability_only_advertises_dynamic_channel_hcu(self):
        self.assertTrue(self.scheme().supports_fp8_prequantized_input)
        self.assertFalse(self.scheme(static=True).supports_fp8_prequantized_input)
        self.assertFalse(self.scheme(strategy='tensor').supports_fp8_prequantized_input)
        m=self.scheme();m.weight_block_size=[128,128];self.assertFalse(m.supports_fp8_prequantized_input)
        with patch.object(C.Q,'_use_aiter',True):self.assertFalse(self.scheme().supports_fp8_prequantized_input)
        with patch.object(C,'_is_hcu',False):self.assertFalse(self.scheme().supports_fp8_prequantized_input)

    def test_native_apply_accepts_compatibility_keywords_and_forwards_bias_once(self):
        m=self.scheme();l=self.layer(m);m.process_weights_after_loading(l)
        with patch.object(C,'hipblaslt_w8a8_channelwise_gemm',side_effect=self.vendor) as gemm,patch.object(C,'lightop_per_token_quant_fp8',side_effect=AssertionError('requant')):
            out=m.apply_weights(l,(self.q,self.scale,torch.float16),self.bias,input_quant_args=[],silu_quant_args=[])
        self.assertEqual(out.dtype,torch.float16);self.assertIs(gemm.call_args.args[-1],self.bias)
        torch.testing.assert_close(out,self.vendor(self.q,l.weight,self.scale,l.weight_scale,2,3,4,'NT',torch.float16,self.bias)[1])

    def test_dynamic_tensor_uses_ocp_quantizer_and_multidimensional_output(self):
        m=self.scheme();l=self.layer(m);m.process_weights_after_loading(l);x=torch.ones(1,2,4,dtype=torch.bfloat16)
        with patch.object(C,'lightop_per_token_quant_fp8',return_value=(self.q,self.scale)) as quant,patch.object(C,'hipblaslt_w8a8_channelwise_gemm',side_effect=self.vendor):out=m.apply_weights(l,x,self.bias)
        self.assertEqual(quant.call_args.args[0].shape,(2,4));self.assertEqual(quant.call_args.kwargs['dtype'],torch.float8_e4m3fn)
        self.assertEqual(out.shape,(1,2,3));self.assertEqual(out.dtype,x.dtype)

    def test_fnuz_pair_reencodes_values_without_requantizing(self):
        m=self.scheme();l=self.layer(m);m.process_weights_after_loading(l);q=self.q.float().to(torch.float8_e4m3fnuz)
        with patch.object(C,'hipblaslt_w8a8_channelwise_gemm',side_effect=self.vendor) as gemm,patch.object(C,'lightop_per_token_quant_fp8',side_effect=AssertionError('requant')):
            m.apply_weights(l,(q,self.scale))
        received=gemm.call_args.args[0];self.assertEqual(received.dtype,torch.float8_e4m3fn)
        torch.testing.assert_close(received.float(),q.float());self.assertIs(gemm.call_args.args[2],self.scale)

    def test_kme_retry_uses_identical_quantized_operand_and_contract(self):
        m=self.scheme();l=self.layer(m);m.process_weights_after_loading(l)
        with patch.object(C,'hipblaslt_w8a8_channelwise_gemm',side_effect=RuntimeError('no algorithm')) as primary,patch.object(C,'hipblaslt_w8a8_channelwise_gemm_kme',side_effect=self.vendor) as retry:
            out=m.apply_weights(l,(self.q,self.scale),self.bias)
        self.assertEqual(primary.call_count,1);self.assertEqual(retry.call_count,1)
        for a,b in zip(primary.call_args.args,retry.call_args.args):self.assertIs(a,b)
        self.assertEqual(out.shape,(2,3))

    def test_retry_error_and_non_runtime_errors_propagate(self):
        m=self.scheme();l=self.layer(m);m.process_weights_after_loading(l)
        with patch.object(C,'hipblaslt_w8a8_channelwise_gemm',side_effect=RuntimeError('primary')),patch.object(C,'hipblaslt_w8a8_channelwise_gemm_kme',side_effect=RuntimeError('retry failure')):
            with self.assertRaisesRegex(RuntimeError,'retry failure'):m.apply_weights(l,(self.q,self.scale))
        with patch.object(C,'hipblaslt_w8a8_channelwise_gemm',side_effect=ValueError('invalid input')),patch.object(C,'hipblaslt_w8a8_channelwise_gemm_kme') as retry:
            with self.assertRaisesRegex(ValueError,'invalid input'):m.apply_weights(l,(self.q,self.scale))
            retry.assert_not_called()

    def test_invalid_weight_orientation_is_rejected_before_launch(self):
        with patch.object(C,'hipblaslt_w8a8_channelwise_gemm') as gemm:
            with self.assertRaisesRegex(ValueError,'weight'):
                C.apply_fp8_lightop_channelwise_linear((self.q,self.scale),torch.ones(4,3),torch.ones(4,1))
            gemm.assert_not_called()

    def test_optout_preserves_transposed_weight_and_full_hcu_per_token_tuple(self):
        m=self.scheme();l=self.layer(m)
        def deepgemm(a,b,out):
            q,s=a;w,ws=b;self.assertEqual(w.shape,(4,3));self.assertIs(s,self.scale)
            out.copy_((q.float()*s)@(w.float()*ws.T))
        with patch.object(C,'_use_lightop_channel_fp8',False),patch.object(C.Q,'is_fp8_fnuz',return_value=False),patch.object(C.Q,'_use_aiter',False),patch.object(L.deepgemm,'fp8_gemm',side_effect=deepgemm),patch.object(L.N,'scaled_fp8_quant',side_effect=AssertionError('requant')):
            m.process_weights_after_loading(l);out=m.apply_weights(l,(self.q,self.scale,torch.float16),self.bias)
        self.assertEqual(l.weight.shape,(4,3));self.assertEqual(out.dtype,torch.float16)

    def test_static_input_scale_is_reduced_after_postload(self):
        m=self.scheme(static=True);l=self.layer(m);m.process_weights_after_loading(l)
        self.assertEqual(l.input_scale.numel(),1);self.assertEqual(l.input_scale.item(),.25)
        self.assertFalse(m.supports_fp8_prequantized_input)

    def test_native_linear_method_forwards_extra_args_through_actual_scheme(self):
        m=CompressedTensorsLinearMethod(None);l=torch.nn.Module();l.scheme=self.scheme()
        m.create_weights(l,4,[3],4,3,torch.bfloat16,weight_loader=lambda *a,**k:None)
        l.weight.data.fill_(1.);l.weight_scale.data.fill_(.25);m.process_weights_after_loading(l)
        iq=[torch.ones(1)];sq=[torch.zeros(1)]
        with patch.object(C,'hipblaslt_w8a8_channelwise_gemm',side_effect=self.vendor),patch.object(C,'apply_weights',wraps=C.apply_weights) as owned:
            out=m.apply(l,(self.q,self.scale,torch.float16),self.bias,input_quant_args=iq,silu_quant_args=sq)
        self.assertEqual(out.dtype,torch.float16);self.assertIs(owned.call_args.args[-2],iq);self.assertIs(owned.call_args.args[-1],sq)

    def test_native_wrapper_respects_other_scheme_signatures(self):
        m=CompressedTensorsLinearMethod(None)
        class LegacyScheme:
            def apply_weights(self,layer,x,bias=None):return x
        l=SimpleNamespace(scheme=LegacyScheme())
        out=m.apply(l,self.q,input_quant_args=[torch.ones(1)],silu_quant_args=[torch.ones(1)])
        self.assertIs(out,self.q)
        with self.assertRaisesRegex(ValueError,'scheme'):m.apply(SimpleNamespace(scheme=None),self.q)

    def test_block_hipblaslt_native_postload_reorients_weight_and_scale(self):
        with patch.object(C.Q,'dispatch_w8a8_block_fp8_linear',return_value=B.hipblaslt_w8a8_block_fp8_linear):
            scheme=CompressedTensorsW8A8Fp8(QuantizationArgs(num_bits=8,type='float',strategy='block',block_structure=[128,128]),False)
        m=CompressedTensorsLinearMethod(None);l=torch.nn.Module();l.scheme=scheme;l.tp_group=None
        m.create_weights(l,512,[256],512,256,torch.bfloat16,weight_loader=lambda *a,**k:None)
        l.weight.data.fill_(1.);l.weight_scale.data.copy_(torch.tensor([[.25,.5,1.,2.],[2.,1.,.5,.25]]))
        old_w=l.weight;old_s=l.weight_scale;scales=old_s.data.clone()
        with patch.object(C.Q,'is_fp8_fnuz',return_value=False):m.process_weights_after_loading(l)
        self.assertIs(l.weight,old_w);self.assertIs(l.weight_scale,old_s)
        self.assertEqual(l.weight.shape,(512,256));self.assertTrue(l.weight.is_contiguous())
        torch.testing.assert_close(l.weight_scale,scales.T.contiguous())
        q=torch.ones(2,512).to(torch.float8_e4m3fn);s=torch.ones(2,4)
        with patch.object(B,'per_token_group_quant_fp8_hcu',return_value=(q,s)),patch.object(B.quant_ops,'hipblaslt_w8a8_blockwise_gemm',return_value=(0,torch.ones(2,256,dtype=torch.bfloat16))) as gemm:
            out=m.apply(l,torch.ones(2,512,dtype=torch.bfloat16))
        self.assertEqual(out.shape,(2,256));self.assertIs(gemm.call_args.args[1],l.weight);self.assertIs(gemm.call_args.args[3],l.weight_scale)

    def test_disabled_direct_helper_rejects_and_non_hcu_delegates(self):
        with patch.object(C,'_use_lightop_channel_fp8',False):
            with self.assertRaisesRegex(RuntimeError,'SGLANG_USE_LIGHTOP_CHANNEL_FP8'):C.apply_fp8_lightop_channelwise_linear(None,None,None)
        original=Mock(return_value='baseline');m=self.scheme();l=self.layer(m)
        with patch.object(C,'is_hcu',return_value=False):
            self.assertEqual(C.around_process_weights(original,m,l),'baseline')
            self.assertEqual(C.around_apply_weights(original,m,l,self.q,self.bias),'baseline')
        self.assertEqual(l.weight.shape,(3,4))


if __name__=='__main__':unittest.main()
