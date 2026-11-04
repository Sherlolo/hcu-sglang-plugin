from __future__ import annotations
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import fp8_mxscale as C, fp8_block as B, w8a8_fp8 as H
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.fp8_utils import aiter_w8a8_block_fp8_linear as early_aiter


class DenseMxscaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset();H.register_hooks();HookRegistry.apply_hooks()

    def setUp(self):
        self.old=C._DENSE_MXSCALE;C._DENSE_MXSCALE=False
        C._dense_mxscale_shapes.cache_clear();C._DENSE_MXSCALE_LOGGED.clear()

    def tearDown(self):
        C._DENSE_MXSCALE=self.old;C._dense_mxscale_shapes.cache_clear()

    def test_public_symbols_and_idempotent_registration(self):
        for name in ['set_dense_mxscale','dense_mxscale_ready','quant_fp8_e8m0_128']:
            self.assertIs(getattr(C.N,name),getattr(C,name))
        C.register_hooks()
        self.assertIs(Fp8LinearMethod._maybe_prepare_dense_mxscale,C._maybe_prepare_dense_mxscale)

    def test_capability_rejection_keeps_global_disabled(self):
        with patch.object(C,'aiter_has_dense_mxscale',return_value=False):
            C.N.set_dense_mxscale(True)
        self.assertFalse(C._DENSE_MXSCALE)
        with patch.object(C,'aiter_has_dense_mxscale',return_value=True):
            C.N.set_dense_mxscale(True)
        self.assertTrue(C._DENSE_MXSCALE);C.N.set_dense_mxscale(False)
        self.assertFalse(C._DENSE_MXSCALE)

    def test_shape_policy_and_actual_architecture_gate(self):
        C._DENSE_MXSCALE=True
        with patch.object(C.N,'_use_aiter_bpreshuffle_gfx95',False):
            self.assertFalse(C.dense_mxscale_wanted(2048,7168))
        with patch.object(C.N,'_use_aiter_bpreshuffle_gfx95',True),patch.dict(os.environ,{'SGLANG_DSV4_DENSE_MXSCALE_SHAPES':''}):
            self.assertTrue(C.dense_mxscale_wanted(2048,7168));self.assertFalse(C.dense_mxscale_wanted(512,7168))
        C._dense_mxscale_shapes.cache_clear()
        with patch.object(C.N,'_use_aiter_bpreshuffle_gfx95',True),patch.dict(os.environ,{'SGLANG_DSV4_DENSE_MXSCALE_SHAPES':'256:512,128:256'}):
            self.assertTrue(C.dense_mxscale_wanted(256,512));self.assertFalse(C.dense_mxscale_wanted(2048,7168))
        C._dense_mxscale_shapes.cache_clear()
        with patch.dict(os.environ,{'SGLANG_DSV4_DENSE_MXSCALE_SHAPES':'all'}):self.assertIsNone(C._dense_mxscale_shapes())
        C._dense_mxscale_shapes.cache_clear()
        with patch.dict(os.environ,{'SGLANG_DSV4_DENSE_MXSCALE_SHAPES':'invalid'}):
            with self.assertRaises(ValueError):C._dense_mxscale_shapes()

    def test_lossless_scale_bytes_and_rejection(self):
        scales=torch.tensor([[.25,.5],[1.,2.]])
        out=C.block_fp8_scale_to_e8m0_128(scales)
        self.assertEqual(out.dtype,torch.float8_e8m0fnu)
        torch.testing.assert_close(out.view(torch.uint8),torch.tensor([[125,126],[127,128]],dtype=torch.uint8))
        for value in [0.,-1.,.3,float('nan'),float('inf'),2.**-149]:
            self.assertIsNone(C.block_fp8_scale_to_e8m0_128(torch.tensor([[value]])))

    def test_shared_producer_requires_all_consumers(self):
        a=SimpleNamespace(weight=torch.zeros(1));b=SimpleNamespace(weight=torch.zeros(1))
        a.weight._mxscale_w_scale=torch.zeros(1)
        C._DENSE_MXSCALE=True
        self.assertFalse(C.N.dense_mxscale_ready(a,b));self.assertTrue(C.N.dense_mxscale_ready(a,None))
        b.weight._mxscale_w_scale=torch.zeros(1)
        self.assertTrue(C.N.dense_mxscale_ready(a,b))
        C._DENSE_MXSCALE=False;self.assertFalse(C.N.dense_mxscale_ready(a,b))

    def method(self):
        m=object.__new__(Fp8LinearMethod)
        m.use_marlin=False;m.block_fp8_as_mxfp8=False;m.use_mxfp8=False;m.convert_mxfp8_to_block=False
        m.weight_block_size=[128,128];m.block_quant=True;m.quant_config=SimpleNamespace(weight_block_size=[128,128])
        m.w8a8_block_fp8_linear=C.N.aiter_w8a8_block_fp8_linear
        return m

    def test_native_postload_prepares_scale_only_after_weight_shuffle(self):
        m=self.method();l=torch.nn.Module()
        l.weight=torch.nn.Parameter(torch.arange(256*512).remainder(16).reshape(256,512).float().to(torch.float8_e4m3fn),requires_grad=False)
        l.weight_scale_inv=torch.nn.Parameter(torch.tensor([[.25,.5,1.,2.],[2.,1.,.5,.25]]),requires_grad=False)
        old=l.weight;original=old.view(torch.uint8).clone();C._DENSE_MXSCALE=True
        with patch.object(B.P,'aiter_w8a8_block_fp8_linear',C.N.aiter_w8a8_block_fp8_linear,create=True),patch.object(B.P,'_use_aiter_bpreshuffle_gfx95',True),patch.object(C.N,'_use_aiter_bpreshuffle_gfx95',True),patch.object(B.P,'use_aiter_triton_gemm_w8a8_tuned_gfx950',return_value=False,create=True),patch.object(B.P,'shuffle_weight',side_effect=lambda w,t:w.view(torch.uint8).flip(1).view(w.dtype),create=True),patch.dict(os.environ,{'SGLANG_DSV4_DENSE_MXSCALE_SHAPES':'all'}):
            m.process_weights_after_loading_block_quant(l)
        self.assertIs(l.weight,old);self.assertTrue(l.aiter_bpreshuffled);self.assertTrue(l.weight.is_shuffled)
        torch.testing.assert_close(l.weight.view(torch.uint8),original.flip(1))
        torch.testing.assert_close(l.weight._mxscale_w_scale.view(torch.uint8),torch.tensor([[125,126,127,128],[128,127,126,125]],dtype=torch.uint8))
        self.assertTrue(C.N.dense_mxscale_ready(l))

    def test_non_power_two_and_plain_layout_do_not_become_ready(self):
        m=self.method();l=SimpleNamespace(weight=torch.zeros(256,512).to(torch.float8_e4m3fn),weight_scale_inv=torch.ones(2,4)*.3)
        with patch.object(C,'dense_mxscale_wanted',return_value=True):m._maybe_prepare_dense_mxscale(l)
        self.assertFalse(hasattr(l.weight,'_mxscale_w_scale'))
        m.weight_block_size=[64,64]
        with patch.object(C,'dense_mxscale_wanted',side_effect=AssertionError('unexpected')):m._maybe_prepare_dense_mxscale(l)

    def test_public_early_alias_consumes_e8m0_without_requantization(self):
        q=torch.ones(2,512).to(torch.float8_e4m3fn);w=torch.ones(256,512).to(q.dtype)
        w._mxscale_w_scale=C.block_fp8_scale_to_e8m0_128(torch.ones(2,4))
        scale=C.block_fp8_scale_to_e8m0_128(torch.ones(2,4));bias=torch.arange(256).to(torch.bfloat16)
        with patch.object(C.N,'gemm_a8w8_blockscale_bpreshuffle',return_value=torch.ones(2,256,dtype=torch.bfloat16),create=True) as gemm,patch.object(C,'quant_fp8_e8m0_128',side_effect=AssertionError('requant')):
            out=early_aiter(q,w,[128,128],torch.ones(2,4),scale,bias)
        self.assertEqual(gemm.call_args.args[0].data_ptr(),q.data_ptr());self.assertIs(gemm.call_args.args[2],scale);self.assertIs(gemm.call_args.args[3],w._mxscale_w_scale)
        torch.testing.assert_close(out,torch.ones(2,256,dtype=torch.bfloat16)+bias)

    def test_mixed_scale_input_keeps_original_fp32_backend(self):
        q=torch.ones(2,512).to(torch.float8_e4m3fn);w=torch.ones(256,512).to(q.dtype)
        w._mxscale_w_scale=C.block_fp8_scale_to_e8m0_128(torch.ones(2,4));scale=torch.ones(2,4)
        with patch.object(C.N,'_use_aiter_bpreshuffle_gfx95',False),patch.object(C.N,'_use_aiter_gfx95',False),patch.object(C.N,'_use_aiter_gfx942',False),patch.object(C.N,'triton_gemm_a8w8_blockscale',return_value=torch.zeros(2,256,dtype=torch.bfloat16),create=True) as gemm:
            early_aiter(q,w,[128,128],torch.ones(2,4),scale)
        self.assertIs(gemm.call_args.args[2],scale)

    def test_missing_scale_fails_and_empty_ranks_never_launch(self):
        q=torch.ones(2,512).to(torch.float8_e4m3fn);w=torch.ones(256,512).to(q.dtype)
        scale=C.block_fp8_scale_to_e8m0_128(torch.ones(2,4))
        with self.assertRaisesRegex(RuntimeError,'without an e8m0'):early_aiter(q,w,[128,128],None,scale)
        with patch.object(C,'quant_fp8_e8m0_128',side_effect=AssertionError('launch')):
            out=early_aiter(q[:0],w,[128,128],None,scale[:0])
        self.assertEqual(out.shape,(0,256));self.assertEqual(out.dtype,torch.bfloat16)

    def test_dynamic_producer_uses_transposed_e8m0_contract(self):
        x=torch.ones(2,512,dtype=torch.bfloat16)
        with patch.object(C,'per_group_quant_hip',return_value=('q','scale'),create=True) as quant,patch.object(C.N,'aiter',SimpleNamespace(dtypes=SimpleNamespace(fp8=torch.float8_e4m3fn)),create=True):
            self.assertEqual(C.N.quant_fp8_e8m0_128(x),('q','scale'))
        self.assertEqual(quant.call_args.kwargs['group_size'],128);self.assertTrue(quant.call_args.kwargs['transpose_scale'])
        self.assertEqual(quant.call_args.kwargs['scale_type'],torch.float8_e8m0fnu)


if __name__=='__main__':unittest.main()
