from __future__ import annotations
import unittest
from unittest.mock import Mock,patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import fp8_block as C
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.srt.layers.quantization.fp8_utils import _dispatch_explicit_backend as early_explicit


class Fp8UtilsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();C.register_hooks();HookRegistry.apply_hooks()

    def test_native_explicit_aiter_error_identifies_hcu_and_required_setting(self):
        with patch.object(C.N,'_use_aiter',False):
            with self.assertRaisesRegex(RuntimeError,'HCU devices.*SGLANG_USE_AITER=1'):
                early_explicit(C.N.Fp8GemmRunnerBackend.AITER)

    def test_public_explicit_backends_preserve_consumer_identity(self):
        self.assertIs(early_explicit(C.N.Fp8GemmRunnerBackend.TRITON),C.N.triton_w8a8_block_fp8_linear)
        with patch.object(C.N,'_use_aiter',True):
            self.assertIs(early_explicit(C.N.Fp8GemmRunnerBackend.AITER),C.N.aiter_w8a8_block_fp8_linear)
        with patch.object(C.N.deep_gemm_wrapper,'ENABLE_JIT_DEEPGEMM',True):
            self.assertIs(early_explicit(C.N.Fp8GemmRunnerBackend.DEEP_GEMM),C.N.deepgemm_w8a8_block_fp8_linear_with_fallback)
        with patch.object(C.N.deep_gemm_wrapper,'ENABLE_JIT_DEEPGEMM',False):
            with self.assertRaisesRegex(RuntimeError,'DeepGEMM is not available'):early_explicit(C.N.Fp8GemmRunnerBackend.DEEP_GEMM)

    def test_non_hcu_retains_effective_dispatcher(self):
        original=Mock(return_value='baseline')
        with patch.object(C,'is_hcu',return_value=False):self.assertEqual(C.around_explicit_backend(original,'backend'),'baseline')
        original.assert_called_once_with('backend')

    def test_reference_preserves_ragged_block_boundaries_and_inputs(self):
        a=torch.tensor([[1.,2.,3.,4.,5.],[2.,0.,1.,3.,4.]]).to(torch.float8_e4m3fn)
        b=torch.arange(15).reshape(5,3).float().to(a.dtype)
        sa=torch.tensor([[.5,1.,2.],[2.,1.,.25]])
        sb=torch.tensor([[.25,.5],[1.,2.],[.5,4.]])
        old_a=a.float().clone();old_b=b.float().clone();old_sa=sa.clone();old_sb=sb.clone()
        expected=torch.zeros(2,3,dtype=torch.float64)
        for i in range(2):
            for j in range(3):
                for k in range(5):expected[i,j]+=float(a[i,k])*float(sa[i,k//2])*float(b[k,j])*float(sb[k//2,j//2])
        for dtype in [torch.float32,torch.bfloat16,torch.float16]:
            out=C.N.torch_fp8_block_gemm(a,b,sa,sb,2,dtype)
            torch.testing.assert_close(out,expected.to(dtype),rtol=0,atol=0)
        torch.testing.assert_close(a.float(),old_a);torch.testing.assert_close(b.float(),old_b)
        torch.testing.assert_close(sa,old_sa);torch.testing.assert_close(sb,old_sb)

    def test_reference_empty_token_rows_and_default_dtype(self):
        out=C.N.torch_fp8_block_gemm(torch.empty(0,4),torch.ones(4,3),torch.empty(0,2),torch.ones(2,2),2)
        self.assertEqual(out.shape,(0,3));self.assertEqual(out.dtype,torch.bfloat16)

    def test_native_ck_guards_retain_source_rows_and_upstream_nan_protections(self):
        expected={(2560,4096):2048,(4096,1024):4096,(4096,2048):2048,(4608,4096):512}
        for shape,limit in expected.items():self.assertEqual(C.N._AITER_GFX95_CK_W8A8_MAX_SAFE_M[shape],limit)


if __name__=='__main__':unittest.main()
