from __future__ import annotations
from types import SimpleNamespace
from unittest.mock import patch, Mock
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import fp8_block as C
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.srt.layers.quantization.fp8_utils import _dispatch_auto_backend as early_auto


class BlockFp8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset(); C.register_hooks(); HookRegistry.apply_hooks()

    def method(self):
        m=object.__new__(Fp8LinearMethod)
        m.use_marlin=False; m.block_fp8_as_mxfp8=False; m.use_mxfp8=False
        m.convert_mxfp8_to_block=False; m.weight_block_size=[128,128]
        m.block_quant=True; m.quant_config=SimpleNamespace(weight_block_size=[128,128])
        m.w8a8_block_fp8_linear=C.hipblaslt_w8a8_block_fp8_linear
        return m

    def layer(self):
        l=torch.nn.Module()
        l.weight=torch.nn.Parameter(torch.arange(512*256).reshape(256,512).remainder(8).float().to(torch.float8_e4m3fn),requires_grad=False)
        l.weight_scale_inv=torch.nn.Parameter(torch.tensor([[.25,.5,.75,1.],[1.,.75,.5,.25]]),requires_grad=False)
        l._block_fp8_bf16_weight=torch.ones(1); l._derived_weight_cache_error='stale'
        return l

    def test_auto_priority_and_early_alias(self):
        with patch.object(C.N.deep_gemm_wrapper,'ENABLE_JIT_DEEPGEMM',False),patch.object(C.N,'_use_aiter',False),patch.object(C.N,'get_platform',return_value=SimpleNamespace(is_blackwell=False,is_sm120=False)):
            self.assertIs(early_auto(),C.hipblaslt_w8a8_block_fp8_linear)
            with patch.object(C.N,'_use_aiter',True):
                self.assertIs(early_auto(),C.N.aiter_w8a8_block_fp8_linear)
            with patch.object(C.N.deep_gemm_wrapper,'ENABLE_JIT_DEEPGEMM',True):
                self.assertIs(early_auto(),C.N.deepgemm_w8a8_block_fp8_linear_with_fallback)

    def test_native_postload_preserves_parameter_identity_and_reorients_both_buffers(self):
        m=self.method();l=self.layer();w=l.weight;s=l.weight_scale_inv
        before_w=w.float().clone();before_s=s.clone()
        m.process_weights_after_loading_block_quant(l)
        self.assertIs(l.weight,w);self.assertIs(l.weight_scale_inv,s)
        self.assertEqual(l.weight.shape,(512,256));self.assertTrue(l.weight.is_contiguous());self.assertTrue(s.is_contiguous())
        torch.testing.assert_close(w.float(),before_w.T);torch.testing.assert_close(s,before_s.T)
        self.assertIsNone(l._block_fp8_bf16_weight);self.assertIsNone(l._derived_weight_cache_error)

    def test_public_consumer_uses_loaded_orientation_and_exact_vendor_contract(self):
        m=self.method();l=self.layer();m.process_weights_after_loading_block_quant(l)
        x=torch.randn(2,3,512,dtype=torch.bfloat16);bias=torch.randn(256,dtype=torch.bfloat16)
        q=torch.zeros(6,512).to(torch.float8_e4m3fn);scale=torch.ones(6,4)
        result=torch.ones(6,256,dtype=x.dtype)
        def vendor(A,B,As,Bs,M,N,K,layout,dtype,block,extra):
            self.assertIs(A,q);self.assertIs(B,l.weight);self.assertIs(As,scale);self.assertIs(Bs,l.weight_scale_inv)
            self.assertEqual((M,N,K,layout,dtype,block,extra),(6,256,512,'NN',x.dtype,C.BlockSize.block_128x128,None))
            return 0,result.clone()
        with patch.object(C,'per_token_group_quant_fp8_hcu',return_value=(q,scale)) as quant,patch.object(C.quant_ops,'hipblaslt_w8a8_blockwise_gemm',side_effect=vendor):
            out=m.apply(l,x,bias)
        self.assertEqual(quant.call_args.args[0].shape,(6,512));self.assertEqual(quant.call_args.args[1],128)
        self.assertFalse(quant.call_args.kwargs['column_major_scales'])
        self.assertEqual(out.shape,(2,3,256));torch.testing.assert_close(out,(result+bias).reshape(2,3,256))

    def test_other_backend_postload_delegates_without_reorienting(self):
        m=self.method();m.w8a8_block_fp8_linear=C.N.triton_w8a8_block_fp8_linear;l=self.layer();original=Mock(return_value='native')
        self.assertEqual(C.around_process_block_weights(original,m,l),'native');original.assert_called_once_with(m,l)
        self.assertEqual(l.weight.shape,(256,512))

    def test_non_hcu_retains_effective_original(self):
        original=Mock(return_value='baseline')
        with patch.object(C,'is_hcu',return_value=False):
            self.assertEqual(C.around_auto_backend(original),'baseline')
            m=self.method();l=self.layer();self.assertEqual(C.around_process_block_weights(original,m,l),'baseline')
        self.assertEqual(l.weight.shape,(256,512))

    def test_incompatible_reduction_dimension_fails_before_vendor(self):
        with patch.object(C.quant_ops,'hipblaslt_w8a8_blockwise_gemm') as gemm:
            with self.assertRaises(AssertionError):
                C.hipblaslt_w8a8_block_fp8_matmul(torch.empty(2,128),torch.empty(256,128),None,None,C.BlockSize.block_128x128)
            gemm.assert_not_called()

    def test_backend_exception_is_not_swallowed(self):
        with patch.object(C.quant_ops,'hipblaslt_w8a8_blockwise_gemm',side_effect=RuntimeError('backend failure')):
            with self.assertRaisesRegex(RuntimeError,'backend failure'):
                C.hipblaslt_w8a8_block_fp8_matmul(torch.empty(2,128),torch.empty(128,256),None,None,C.BlockSize.block_128x128)


if __name__=='__main__': unittest.main()
