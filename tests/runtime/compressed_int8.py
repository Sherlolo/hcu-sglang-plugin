from __future__ import annotations
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import torch
from compressed_tensors.quantization import QuantizationStrategy as S
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import compressed_int8 as C, w8a8_fp8 as H
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsLinearMethod
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import CompressedTensorsW8A8Int8
from sglang.srt.plugins.hook_registry import HookRegistry


class CompressedInt8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();H.register_hooks();HookRegistry.apply_hooks()

    def setUp(self):
        self.cache=SimpleNamespace(weight_shapes=[],triton_json_dict={},get_w8a8json_name=Mock(return_value='task-cache.json'),get_triton_cache=Mock(return_value=None),gen_model_json=Mock())
        self.cache_patch=patch.object(C,'W8A8_TRITONJSON',self.cache);self.cache_patch.start();self.addCleanup(self.cache_patch.stop)
        self.q=torch.tensor([[1,2,3,4],[4,3,2,1]],dtype=torch.int8);self.s=torch.tensor([[.25],[.5]])
        self.w=torch.tensor([[1,2,3,4],[2,3,4,1],[4,3,2,1]],dtype=torch.int8)
        self.ws=torch.tensor([[.25],[.5],[1.]])

    def create(self,method=3,static=False,symmetric=True,strategy=S.CHANNEL):
        with patch.dict(os.environ,{'W8A8_SUPPORT_METHODS':str(method)}):scheme=CompressedTensorsW8A8Int8(strategy,static,symmetric)
        m=CompressedTensorsLinearMethod(None);l=torch.nn.Module();l.scheme=scheme
        m.create_weights(l,4,[3],4,3,torch.bfloat16,weight_loader=lambda *a,**k:None)
        l.weight.data.copy_(self.w);l.weight_scale.data.copy_(self.ws if strategy==S.CHANNEL else torch.ones_like(l.weight_scale)*.5)
        if static:
            l.input_scale.data.fill_(.25)
            if not symmetric:l.input_zero_point.data.fill_(2)
        return m,l

    def test_ctor_records_explicit_strategy_and_preserves_parameter_loader(self):
        m,l=self.create(3);self.assertEqual(l.scheme.w8a8_strategy,3)
        self.assertEqual(l.weight.shape,(3,4));self.assertTrue(hasattr(l.weight,'weight_loader'))

    def test_bf16_checkpoint_scales_normalize_before_legacy_nt_loading(self):
        m,l=self.create();l.weight_scale=torch.nn.Parameter(self.ws.to(torch.bfloat16),requires_grad=False)
        m.process_weights_after_loading(l)
        self.assertEqual(l.weight_scale.dtype,torch.float32);self.assertEqual(l.weight.shape,(3,4));self.assertTrue(l.weight.is_contiguous())
        torch.testing.assert_close(l.weight_scale,self.ws)

    def test_gfx928_packing_and_dynamic_azp_keep_original_output_channels(self):
        m,l=self.create(symmetric=False)
        with patch.object(C,'_use_kme_hipblaslt',return_value=True):m.process_weights_after_loading(l)
        self.assertEqual(l.weight.shape,(4,3));self.assertEqual(l.weight.stride(),(1,4))
        torch.testing.assert_close(l.weight.T,self.w);torch.testing.assert_close(l.azp_adj,self.w.sum(1,dtype=torch.int32).reshape(1,3))

    def test_static_asymmetric_metadata_remains_well_formed(self):
        m,l=self.create(static=True,symmetric=False);m.process_weights_after_loading(l)
        self.assertEqual(l.input_scale.dtype,torch.float32);self.assertEqual(l.input_zero_point.dtype,torch.int32)
        torch.testing.assert_close(l.azp_adj,l.input_zero_point*self.w.sum(1,dtype=torch.int32).reshape(1,3))

    def test_tensor_scale_strategy_requantizes_before_nt_layout(self):
        m,l=self.create(strategy=S.TENSOR);m.process_weights_after_loading(l)
        self.assertEqual(l.weight.shape,(3,4));self.assertEqual(l.weight_scale.numel(),1);self.assertIsNone(l.input_scale)

    def test_triton_loading_warms_only_provided_cache_rows(self):
        m,l=self.create(1);self.cache.get_triton_cache.return_value={'1_3_4':[1,2,3]}
        with patch.object(C.ops,'triton_int8_gemm_helper') as warm:m.process_weights_after_loading(l)
        self.assertEqual(l.weight.shape,(4,3));warm.assert_called_once()
        self.assertEqual(warm.call_args.kwargs['best_config'],[1,2,3]);self.assertEqual(self.cache.weight_shapes,[[3,4]])

    def test_fused_input_priority_through_actual_native_wrapper(self):
        m,l=self.create();m.process_weights_after_loading(l);sq=[self.q,self.s*2];iq=[self.q,self.s]
        with patch.object(C,'_use_fused_rms_quant',True),patch.object(C,'_use_fused_silu_mul_quant',True),patch.object(C,'per_token_quant_int8',side_effect=AssertionError('requant')),patch.object(C.ops,'blaslt_scaled_mm',return_value=torch.ones(2,3,dtype=torch.bfloat16)) as gemm:
            m.apply(l,torch.ones(2,4,dtype=torch.bfloat16),None,input_quant_args=iq,silu_quant_args=sq)
        self.assertIs(gemm.call_args.kwargs['scale_a'],self.s)
        self.assertIn('input_quant_args',l.scheme._apply_weights_supported_kwargs)

    def test_silu_pair_skips_dynamic_quantization(self):
        m,l=self.create();m.process_weights_after_loading(l)
        with patch.object(C,'_use_fused_rms_quant',False),patch.object(C,'_use_fused_silu_mul_quant',True),patch.object(C,'per_token_quant_int8',side_effect=AssertionError('requant')),patch.object(C.ops,'blaslt_scaled_mm',return_value=torch.ones(2,3)) as gemm:
            m.apply(l,torch.ones(2,4),None,silu_quant_args=[self.q,self.s])
        self.assertIs(gemm.call_args.args[0],self.q);self.assertIs(gemm.call_args.kwargs['scale_a'],self.s)

    def test_triton_m_buckets_select_source_configuration(self):
        m,l=self.create(1);m.process_weights_after_loading(l)
        for rows,bucket in [(17,20),(65,72),(180,160),(480,512)]:
            self.cache.triton_json_dict={'1_3_4':'base',f'{bucket}_3_4':'chosen'}
            q=torch.zeros(rows,4,dtype=torch.int8);s=torch.ones(rows,1)
            with patch.object(C,'per_token_quant_int8',return_value=(q,s)),patch.object(C.ops,'triton_scaled_mm',return_value=torch.zeros(rows,3)) as gemm:
                m.apply(l,torch.ones(rows,4),None)
            self.assertEqual(gemm.call_args.kwargs['best_config'],'chosen')

    def test_all_strategy_backends_reachable_and_errors_propagate(self):
        for strategy,name in [(1,'triton_scaled_mm'),(2,'cutlass_scaled_mm'),(3,'blaslt_scaled_mm'),(4,'rocblas_scaled_mm')]:
            m,l=self.create(strategy);m.process_weights_after_loading(l)
            with patch.object(C,'per_token_quant_int8',return_value=(self.q,self.s)),patch.object(C.ops,name,side_effect=RuntimeError(name)):
                with self.assertRaisesRegex(RuntimeError,name):m.apply(l,torch.ones(2,4),None)

    def test_legacy_blaslt_bias_is_not_lost(self):
        m,l=self.create();m.process_weights_after_loading(l);bias=torch.tensor([1.,2.,3.],dtype=torch.bfloat16)
        with patch.object(C,'per_token_quant_int8',return_value=(self.q,self.s)),patch.object(C.ops.quant_ops,'hipblaslt_w8a8_gemm',return_value=(0,torch.zeros(2,3,dtype=torch.bfloat16))):
            out=m.apply(l,torch.ones(2,4,dtype=torch.bfloat16),bias)
        torch.testing.assert_close(out,bias.expand(2,3))

    def test_kme_arch_detector_has_precise_device_boundary(self):
        self.assertFalse(C._use_kme_hipblaslt(torch.device('cpu')))
        for arch,result in [('gfx928:sramecc+',True),('gfx938',False),('gfx936',False)]:
            with patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(gcnArchName=arch)):
                self.assertEqual(C._use_kme_hipblaslt(torch.device('cuda')),result)

    def test_non_hcu_delegates_effective_native_methods(self):
        original=Mock(return_value='baseline');m,l=self.create()
        with patch.object(C,'is_hcu',return_value=False):
            self.assertEqual(C.around_init(original,l.scheme,S.CHANNEL,False,True),'baseline')
            self.assertEqual(C.around_process(original,l.scheme,l),'baseline')
            self.assertEqual(C.around_apply(original,l.scheme,l,self.q,None),'baseline')
        self.assertEqual(original.call_count,3)


if __name__=='__main__':unittest.main()
