from __future__ import annotations
import unittest
from types import SimpleNamespace
from unittest.mock import Mock,patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.quantization import fp8_moe as C
from hcu_sglang.layers.moe import mega_moe as M
from sglang.srt.layers.quantization.fp8 import Fp8MoEMethod,Fp8Config
from sglang.srt.plugins.hook_registry import HookRegistry
import sglang.srt.layers.moe.mega_moe as G


class Fp8MoeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset();M.register_hooks();C.register_hooks();HookRegistry.apply_hooks()

    def method(self):
        cfg=Fp8Config(is_checkpoint_fp8_serialized=True,activation_scheme='dynamic',weight_block_size=[128,128])
        return Fp8MoEMethod(cfg)

    def create(self,m,fp4=False,**extra):
        l=torch.nn.Module();l.moe_tp_size=1;l.moe_runner_config=SimpleNamespace(is_gated=True)
        m.is_fp4_expert=fp4
        m.create_weights(l,2,128,128,torch.bfloat16,**extra)
        return l

    def test_public_constructor_create_and_static_descriptor_preserved(self):
        m=self.method();l=self.create(m)
        self.assertIsInstance(Fp8MoEMethod.__dict__['create_fp8_moe_weight_'],staticmethod)
        self.assertEqual(l.w13_weight.shape,(2,256,128));self.assertEqual(l.w2_weight.shape,(2,128,128))
        self.assertEqual(l.w13_weight.dtype,torch.float8_e4m3fn)
        self.assertEqual(l.w13_weight_scale_inv.shape,(2,2,1));self.assertEqual(l.w2_weight_scale_inv.shape,(2,1,1))

    def test_fp4_scale_buffers_are_fp32_even_when_aiter_or_scale_override_requests_e8m0(self):
        m=self.method()
        with patch.object(C.P,'_use_aiter',True):
            l=self.create(m,fp4=True,fp4_scale_dtype=torch.float8_e8m0fnu)
        self.assertEqual(l.w13_weight.dtype,torch.int8);self.assertEqual(l.w2_weight.dtype,torch.int8)
        self.assertEqual(l.w13_weight_scale_inv.dtype,torch.float32);self.assertEqual(l.w2_weight_scale_inv.dtype,torch.float32)
        self.assertEqual(l.w13_weight_scale_inv.shape,(2,256,4));self.assertEqual(l.w2_weight_scale_inv.shape,(2,128,4))

    def test_hcu_quant_info_stops_before_generic_runner_core_or_no_combine_assertion(self):
        m=self.method()
        with patch.object(C.P,'_use_aiter',True),patch.object(C.P,'_use_hip_int4',True):
            self.assertIsNone(m.maybe_get_hip_aiter_quant_info(None,no_combine=True))

    def test_native_public_megamoe_handoff_does_not_enter_generic_processing(self):
        m=self.method();m.block_quant=False;l=torch.nn.Module()
        with patch.object(C.P,'get_moe_a2a_backend',return_value=SimpleNamespace(is_megamoe=lambda:True)),patch.object(G,'get_hcu_mega_moe_runtime',return_value='megamoe'),patch.object(G,'build_hcu_w8a8_mega_moe_experts_weights') as build:
            m.process_weights_after_loading(l)
        build.assert_called_once_with(l)

    def test_standalone_megamoe_rejects_unsupported_formats_before_mutation(self):
        m=self.method();l=torch.nn.Module()
        with patch.object(C.P,'get_moe_a2a_backend',return_value=SimpleNamespace(is_megamoe=lambda:True)),patch.object(G,'get_hcu_mega_moe_runtime',return_value='megamoe'),patch.object(G,'build_hcu_w8a8_mega_moe_experts_weights') as build:
            for block,fp4,serialized in [(True,False,True),(False,True,True),(False,False,False)]:
                m.block_quant=block;m.is_fp4_expert=fp4;m.quant_config.is_checkpoint_fp8_serialized=serialized
                with self.assertRaisesRegex(ValueError,'standalone megamoe'):m.process_weights_after_loading(l)
            build.assert_not_called()
        self.assertEqual(len(list(l.parameters())),0)

    def test_grouped_megamoe_block_dispatch_reaches_block_processor(self):
        m=self.method();l=torch.nn.Module();m.process_weights_after_loading_block_quant=Mock()
        with patch.object(C.P,'get_moe_a2a_backend',return_value=SimpleNamespace(is_megamoe=lambda:True)),patch.object(G,'get_hcu_mega_moe_runtime',return_value='deep_gemm'):
            m.process_weights_after_loading(l)
        m.process_weights_after_loading_block_quant.assert_called_once_with(l)

    def test_hcu_asm_shuffles_gate_up_then_down_once_without_generic_16x16_shuffle(self):
        m=self.method();l=self.create(m);old13=l.w13_weight;old2=l.w2_weight
        l.w13_weight.data.copy_(torch.arange(old13.numel()).reshape(old13.shape).remainder(16).float().to(old13.dtype))
        l.w2_weight.data.copy_(torch.arange(old2.numel()).reshape(old2.shape).remainder(16).float().to(old2.dtype))
        originals=[old13.view(torch.uint8).clone(),old2.view(torch.uint8).clone()];calls=[]
        def asm(w,kind):calls.append((w,kind));return w.view(torch.uint8).flip(-1).view(w.dtype)
        with patch.object(C.P,'_use_aiter',True),patch.object(C,'asm_shuffle_weight_b8',side_effect=asm),patch.object(C.P,'shuffle_weight',side_effect=AssertionError('generic shuffle'),create=True):
            m.process_weights_after_loading_block_quant(l)
        self.assertEqual([x[1] for x in calls],[1,2]);self.assertIs(calls[0][0],old13);self.assertIs(calls[1][0],old2)
        self.assertIs(l.w13_weight,old13);self.assertIs(l.w2_weight,old2)
        torch.testing.assert_close(old13.view(torch.uint8),originals[0].flip(-1));torch.testing.assert_close(old2.view(torch.uint8),originals[1].flip(-1))

    def test_hcu_packed_fp4_stays_int8_on_generic_path(self):
        m=self.method();l=self.create(m,fp4=True);m.is_deepgemm_moe_runner_backend_enabled=Mock(return_value=False)
        m.process_weights_after_loading_block_quant(l)
        self.assertEqual(l.w13_weight.dtype,torch.int8);self.assertEqual(l.w2_weight.dtype,torch.int8)
        self.assertEqual(l.w13_weight_scale_inv.dtype,torch.float32)

    def test_source_dequant_branch_uses_original_runner_aware_early_return(self):
        m=self.method();l=self.create(m,fp4=True);m.dequant_fp4_to_fp8=True
        m._dequantize_aiter_fp4_experts=Mock()
        with patch.object(C.P,'_use_aiter',True),patch.object(C.P,'get_moe_a2a_backend',return_value=SimpleNamespace(is_megamoe=lambda:False)),patch.object(C,'asm_shuffle_weight_b8',side_effect=AssertionError('late ASM shuffle')):
            m.process_weights_after_loading_block_quant(l)
        m._dequantize_aiter_fp4_experts.assert_called_once_with(l);self.assertEqual(m.weight_block_size,[128,128])

    def test_non_hcu_all_entry_points_retain_effective_original(self):
        original=Mock(return_value='baseline');m=self.method();l=torch.nn.Module()
        with patch.object(C,'is_hcu',return_value=False):
            self.assertEqual(C.around_create_weights(original,l,2),'baseline')
            self.assertEqual(C.around_quant_info(original,m,l,True),'baseline')
            self.assertEqual(C.around_process_weights(original,m,l),'baseline')
            self.assertEqual(C.around_process_block_weights(original,m,l),'baseline')
        self.assertEqual(original.call_count,4)


if __name__=='__main__':unittest.main()
