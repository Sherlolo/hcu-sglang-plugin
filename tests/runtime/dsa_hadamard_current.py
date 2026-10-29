from unittest.mock import patch
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.layers.attention.dsa import dsa_indexer as S
from sglang.srt.plugins.hook_registry import HookRegistry,HookType
from hcu_sglang.layers.attention.current_dsa import rotation as R

class HadamardCurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=staticmethod(S.rotate_activation);HookRegistry.reset()
        HookRegistry.register('sglang.srt.layers.attention.dsa.dsa_indexer.rotate_activation',R.rotate_activation,HookType.REPLACE);HookRegistry.apply_hooks()
    @classmethod
    def tearDownClass(cls):S.rotate_activation=cls.original;HookRegistry.reset()
    def test_cpu_exact_transform_scale_and_shape(self):
        x=torch.arange(8,dtype=torch.float32).reshape(1,2,4)
        h=torch.tensor([[1.,1.,1.,1.],[1.,-1.,1.,-1.],[1.,1.,-1.,-1.],[1.,-1.,-1.,1.]])
        with patch.object(R,'_is_hcu',True),patch.object(R,'_use_fast_hadamard_transform',False):
            self.assertTrue(torch.equal(S.rotate_activation(x),x@h/2))
            self.assertTrue(torch.equal(S.rotate_activation(x,apply_scale=False),x@h))
            self.assertEqual(S.rotate_activation(x).shape,x.shape)
            with self.assertRaises(AssertionError):S.rotate_activation(torch.ones(2,3))
    def test_cache_preserves_device_dtype_and_storage(self):
        a=R._get_hadamard_matrix(128,torch.device('cpu'),torch.float32)
        b=R._get_hadamard_matrix(128,torch.device('cpu'),torch.float32)
        c=R._get_hadamard_matrix(128,torch.device('cpu'),torch.bfloat16)
        self.assertEqual(a.data_ptr(),b.data_ptr());self.assertNotEqual(a.data_ptr(),c.data_ptr())
        self.assertTrue(torch.equal(a@a.T,torch.eye(128)*128))
    @unittest.skipUnless(torch.cuda.is_available(),'GPU unavailable')
    def test_hcu_real_gemm_matches_cpu_hadamard(self):
        generator=torch.Generator().manual_seed(41)
        x=torch.randn(4,8,128,generator=generator).bfloat16()
        h=R._get_hadamard_matrix(128,torch.device('cpu'),torch.float32)
        expected=(x.float()@h.T)*(128**-.5)
        with patch.object(R,'_is_hcu',True),patch.object(R,'_use_fast_hadamard_transform',False):out=S.rotate_activation(x.cuda())
        self.assertEqual(out.device.type,'cuda');self.assertEqual(out.dtype,torch.bfloat16)
        torch.testing.assert_close(out.cpu().float(),expected,atol=.04,rtol=.015)
        print('Hadamard BF16 GPU max abs error:',(out.cpu().float()-expected).abs().max().item())

if __name__=='__main__':unittest.main()
