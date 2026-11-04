from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock,patch
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers import current_linear as C,linear_fusion as H
from hcu_sglang.layers.quantization import slimquant_w4a8 as Q
from sglang.srt.plugins.hook_registry import HookRegistry


class CurrentLinearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset();H.register_hooks();HookRegistry.apply_hooks()

    def setUp(self):
        self.scope=ExitStack();self.addCleanup(self.scope.close)
        self.scope.enter_context(patch.object(C.N,'get_parallel',return_value=SimpleNamespace(tp_group=None,tp_size=1,tp_rank=0)))
        self.scope.enter_context(patch.object(C.N,'get_forward',return_value=SimpleNamespace(sp_active=False)))
        self.scope.enter_context(patch.object(C,'_use_fused_rms_quant',False))

    def layer(self,kind='column',**kwargs):
        if kind=='column':return C.N.ColumnParallelLinear(8,4,bias=False,params_dtype=torch.float32,**kwargs)
        if kind=='merged':return C.N.MergedColumnParallelLinear(8,[2,2],bias=False,params_dtype=torch.float32,**kwargs)
        return C.N.ReplicatedLinear(8,4,bias=False,params_dtype=torch.float32,**kwargs)

    def consumer(self,layer):
        method=object.__new__(Q.SlimQuantW4A8Int8LinearMethod);method.w8a8_strategy=3
        layer.register_parameter('weight',torch.nn.Parameter(torch.ones(4,8,dtype=torch.int8),requires_grad=False))
        layer.weight_scale=torch.full((4,1),.5);layer.quant_method=method
        method.process_weights_after_loading(layer)
        return method

    def test_explicit_carrier_reaches_real_int8_consumer_with_flag_off(self):
        for kind in ('column','merged','replicated'):
            layer=self.layer(kind);self.consumer(layer);x=torch.zeros(2,8);q=torch.ones(2,8,dtype=torch.int8);scale=torch.tensor([[.25],[.5]])
            def gemm(actual_q,w,*,scale_a,scale_b,out_dtype,bias):
                self.assertIs(actual_q,q);self.assertIs(scale_a,scale)
                return ((actual_q.float()*scale_a)@(w.float()*scale_b).T).to(out_dtype)
            with patch.object(Q,'per_token_quant_int8',side_effect=AssertionError('explicit carrier discarded')),patch.object(Q.quant_ops,'blaslt_scaled_mm',side_effect=gemm):
                output,bias=layer(x,input_quant_args=(q,scale))
            torch.testing.assert_close(output,torch.tensor([[1.]*4,[2.]*4]));self.assertIsNone(bias)

    def test_constructor_epsilon_partition_and_inherited_qkv_state(self):
        layer=self.layer(eps=3e-4);self.assertEqual(layer.eps,3e-4);self.assertEqual(layer.output_partition_sizes,[4])
        merged=self.layer('merged');self.assertEqual(merged.eps,1e-6);self.assertEqual(merged.output_partition_sizes,[2,2])
        qkv=C.N.QKVParallelLinear(8,2,4,2,bias=False,params_dtype=torch.float32)
        self.assertEqual(qkv.eps,1e-6);self.assertEqual(qkv.input_size,8)

    def test_fused_column_preserves_input_residual_update_and_epsilon(self):
        layer=self.layer(eps=2e-4);self.consumer(layer);x=torch.ones(2,8);residual=torch.full_like(x,2.);weight=torch.ones(8);q=torch.ones_like(x,dtype=torch.int8);scale=torch.full((2,1),.5)
        def fused(**kwargs):
            self.assertIs(kwargs['input'],x);self.assertIs(kwargs['rms_weight'],weight);self.assertEqual(kwargs['epsilon'],2e-4);self.assertFalse(kwargs['update_input'])
            kwargs['residual'].add_(x)
            return q,scale
        def gemm(actual_q,w,*,scale_a,scale_b,out_dtype,bias):return ((actual_q.float()*scale_a)@(w.float()*scale_b).T).to(out_dtype)
        with patch.object(C,'_use_fused_rms_quant',True),patch.object(C,'lm_faster_rmsquant',side_effect=fused) as producer,patch.object(Q,'per_token_quant_int8',side_effect=AssertionError('requantized')),patch.object(Q.quant_ops,'blaslt_scaled_mm',side_effect=gemm):
            output,_=layer(x,rms_weight=weight,residual=residual,update_hd=False)
        producer.assert_called_once();torch.testing.assert_close(x,torch.ones_like(x));torch.testing.assert_close(residual,torch.full_like(x,3.));torch.testing.assert_close(output,torch.full((2,4),2.))

    def test_merged_fused_return_retains_all_five_consumer_values(self):
        layer=self.layer('merged');x=torch.ones(2,8);residual=torch.ones_like(x);weight=torch.ones(8);q=torch.ones_like(x,dtype=torch.int8);scale=torch.full((2,1),.5);marker=object()
        method=SimpleNamespace(supports_prequantized_input=True,apply=Mock(return_value=marker));layer.quant_method=method
        with patch.object(C,'_use_fused_rms_quant',True),patch.object(C,'lm_faster_rmsquant',return_value=(q,scale)):
            result=layer(x,rms_weight=weight,residual=residual)
        self.assertEqual(len(result),5);self.assertIs(result[0],marker);self.assertIs(result[1],residual);self.assertIs(result[2],q);self.assertIs(result[3],scale);self.assertIsNone(result[4])
        self.assertIs(method.apply.call_args.kwargs['input_quant_args'][0],q)

    def test_replicated_alias_conflict_and_explicit_fusion_epsilon(self):
        layer=self.layer('replicated');x=torch.ones(2,8);carrier=(torch.ones_like(x,dtype=torch.int8),torch.ones(2,1));layer.quant_method=SimpleNamespace(supports_prequantized_input=True,apply=Mock(return_value=x))
        with self.assertRaisesRegex(ValueError,'only one'):layer(x,quant_args=carrier,input_quant_args=carrier)
        layer(x,quant_args=carrier);self.assertIs(layer.quant_method.apply.call_args.kwargs['input_quant_args'],carrier)
        with patch.object(C,'_use_fused_rms_quant',True),self.assertRaisesRegex(ValueError,'rms_norm_eps'):layer(x,rms_weight=torch.ones(8))

    def test_unsupported_consumer_is_rejected_without_dropping_carrier(self):
        for kind in ('column','merged','replicated'):
            layer=self.layer(kind);layer.quant_method=SimpleNamespace(supports_prequantized_input=False,apply=Mock());x=torch.ones(2,8)
            with self.assertRaisesRegex(TypeError,'prequantized'):layer(x,input_quant_args=(x,x))
            layer.quant_method.apply.assert_not_called()

    def test_attention_gather_and_cp_resolution_use_selected_groups(self):
        marker=object();group=SimpleNamespace(all_gather=Mock(return_value=marker));layer=self.layer();layer.quant_method=SimpleNamespace(apply=Mock(return_value=torch.ones(2,4)));layer.gather_output=True;layer.parallel_group='attn_tp'
        with patch.object(C.N,'get_parallel',return_value=SimpleNamespace(attn_tp_group=group,attn_cp_group=group)):
            output,_=layer(torch.ones(2,8));self.assertIs(output,marker);group.all_gather.assert_called_once()
            self.assertIs(C.N._resolve_linear_group('attn_cp'),group)

    def test_public_row_accepts_source_positional_carrier(self):
        layer=C.N.RowParallelLinear(8,4,bias=False,input_is_parallel=True,reduce_results=False,params_dtype=torch.float32)
        self.consumer(layer);x=torch.zeros(2,8);q=torch.ones_like(x,dtype=torch.int8);scale=torch.full((2,1),.5)
        def gemm(actual_q,w,*,scale_a,scale_b,out_dtype,bias):return ((actual_q.float()*scale_a)@(w.float()*scale_b).T).to(out_dtype)
        with patch.object(Q,'per_token_quant_int8',side_effect=AssertionError('carrier discarded')),patch.object(Q.quant_ops,'blaslt_scaled_mm',side_effect=gemm):
            output,bias=layer(x,False,None,False,False,None,None,(q,scale))
        torch.testing.assert_close(output,torch.full((2,4),2.));self.assertIsNone(bias)

    def test_public_row_accepts_unused_source_limit_and_old_output_position(self):
        layer=C.N.RowParallelLinear(8,4,bias=False,input_is_parallel=True,reduce_results=False,params_dtype=torch.float32)
        layer.weight.data.fill_(.5);x=torch.ones(2,8)
        output,_=layer(x,swiglu_limit=3.);torch.testing.assert_close(output,torch.full((2,4),4.))
        from hcu_sglang.layers import row_fusion
        marker=object();original=Mock(return_value=marker);buffer=torch.empty(2,4)
        self.assertIs(row_fusion.around_row_forward(original,layer,x,False,None,buffer),marker)
        original.assert_called_once_with(layer,x,False,None,buffer)

    def test_default_and_non_hcu_call_contracts_remain_normal_pairs(self):
        layer=self.layer();layer.weight.data.fill_(.5);x=torch.ones(2,8)
        output,bias=layer(x,rms_weight=torch.full((8,),2.));torch.testing.assert_close(output,torch.full((2,4),4.));self.assertIsNone(bias)
        marker=object();original=Mock(return_value=marker)
        with patch.object(C,'is_hcu',return_value=False):self.assertIs(C._around(C.column_forward)(original,layer,x),marker)
        original.assert_called_once_with(layer,x)


if __name__=='__main__':unittest.main()
