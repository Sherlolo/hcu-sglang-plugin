from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers import norm_current as C
from hcu_sglang.layers.quantization import fp8 as Q
from sglang.srt.plugins.hook_registry import HookRegistry


def rms(x, weight, eps):
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)


class NormCurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset()
        Q.register_hooks()
        HookRegistry.apply_hooks()

    def make_rms(self):
        with patch.object(C.N, '_use_aiter', False):
            layer = C.N.RMSNorm(4, weight_dtype=torch.float32)
        layer.weight.data.copy_(torch.tensor([.7, 1.1, .9, 1.3]))
        layer._forward_method = layer.forward_hip
        return layer

    def make_gemma(self):
        layer = C.N.GemmaRMSNorm(4)
        layer.weight.weight_loader(layer.weight, torch.tensor([.2, -.3, .7, -.1]))
        layer._forward_method = layer.forward_hip
        return layer

    def test_public_rms_fused_residual_post_add_and_noncontiguous(self):
        layer = self.make_rms()
        original = torch.arange(24, dtype=torch.float32).reshape(3, 8)[:, ::2]
        self.assertFalse(original.is_contiguous())
        for post in (None, torch.full((3, 4), .25)):
            x = original.clone(memory_format=torch.preserve_format)
            if x.is_contiguous():
                x = torch.arange(24, dtype=torch.float32).reshape(3, 8)[:, ::2]
            residual = torch.full_like(x, .5)
            expected_residual = x + residual + (0 if post is None else post)
            calls = []
            def fused(value, residual_arg, weight, eps):
                self.assertTrue(value.is_contiguous())
                self.assertEqual(weight.data_ptr(), layer.weight.data_ptr())
                residual_arg.add_(value)
                value.copy_(rms(residual_arg, weight, eps))
                calls.append((value, residual_arg))
            with patch.object(C.N, 'is_batch_invariant_mode_enabled', return_value=False), patch.object(C.op, 'fused_add_rms_norm_opt', side_effect=fused), patch.object(layer, 'forward_native', side_effect=AssertionError('DAS HCU fused path was bypassed')):
                output, updated = layer(x, residual, post)
            self.assertEqual(len(calls), 1)
            self.assertIs(output, calls[0][0])
            self.assertIs(updated, calls[0][1])
            torch.testing.assert_close(updated, expected_residual)
            torch.testing.assert_close(output, rms(expected_residual, layer.weight, layer.variance_epsilon))
            if post is None:
                self.assertIs(updated, residual)
            else:
                torch.testing.assert_close(residual, torch.full_like(residual, .5))

    def test_public_rms_no_residual_and_batch_invariant(self):
        layer = self.make_rms()
        x = torch.tensor([[1., 2., 3., 4.]])
        calls = []
        def plain(out, value, weight, eps):
            out.copy_(rms(value, weight, eps)); calls.append(out)
        with patch.object(C.N, 'is_batch_invariant_mode_enabled', return_value=False), patch.object(C.op, 'rms_norm_opt', side_effect=plain):
            result = layer(x)
        self.assertIs(result, calls[0])
        torch.testing.assert_close(result, rms(x, layer.weight, layer.variance_epsilon))
        marker = object()
        with patch.object(C.N, 'is_batch_invariant_mode_enabled', return_value=True), patch.object(C.N, 'get_exec', return_value=SimpleNamespace(deterministic=SimpleNamespace(rl_on_policy_target='none'))), patch.object(C.N, 'rms_norm_batch_invariant', return_value=marker) as invariant, patch.object(C.op, 'rms_norm_opt', side_effect=AssertionError('invariant guard lost')):
            self.assertIs(layer(x), marker)
            invariant.assert_called_once()
        with patch.object(C.N, 'is_batch_invariant_mode_enabled', return_value=True), patch.object(layer, 'forward_native', return_value=marker):
            self.assertIs(layer(x, x), marker)

    def test_public_gemma_uses_weight_plus_one_and_residual(self):
        layer = self.make_gemma()
        x = torch.tensor([[1., 2., 3., 4.]])
        def plain(out, value, weight, eps):
            self.assertIs(weight, layer.gemma_weight)
            out.copy_(rms(value, weight, eps))
        def fused(value, residual, weight, eps):
            self.assertEqual(weight.data_ptr(), layer.weight.data_ptr())
            updated = value + residual
            return rms(updated, weight + 1, eps), updated
        with patch.object(C.N, '_use_aiter', False), patch.object(C.op, 'rms_norm_opt', side_effect=plain), patch.object(C, 'gemma_fused_add_rmsnorm_hcu', side_effect=fused):
            torch.testing.assert_close(layer(x), rms(x, layer.weight + 1, layer.variance_epsilon))
            output, residual = layer(x, torch.ones_like(x), torch.ones_like(x) * .5)
            torch.testing.assert_close(residual, x + 1.5)
            torch.testing.assert_close(output, rms(x + 1.5, layer.weight + 1, layer.variance_epsilon))

    def test_gemma_preserves_aiter_precedence(self):
        layer = self.make_gemma(); marker = object(); x = torch.ones(1, 4)
        with patch.object(C.N, '_use_aiter', True), patch.object(C.N, '_has_rocm_triton_gemma_rms_norm', True), patch.object(C.N, 'rocm_triton_gemma_rmsnorm', return_value=marker, create=True), patch.object(C.op, 'rms_norm_opt', side_effect=AssertionError('AITER priority changed')):
            self.assertIs(layer(x), marker)

    def test_gemma_fp8_explicit_gate_carrier_and_post_residual(self):
        layer = self.make_gemma(); x = torch.ones(1, 4)
        with patch.object(C, '_use_hcu_lightop_gemma_rmsnorm', False):
            with self.assertRaisesRegex(RuntimeError, 'SGLANG_USE_LIGHTOP_GEMMA_RMSNORM'):
                layer.forward_with_lightop_fp8_quant(x)
        q = torch.ones(1, 4).to(torch.float8_e4m3fn); scale = torch.tensor([[.25]]); carrier = (q, scale)
        def quant(value, weight, eps, *, fp8type, residual, update_input):
            self.assertIs(value, x); self.assertEqual(weight.data_ptr(), layer.weight.data_ptr())
            self.assertEqual(fp8type, 0); self.assertFalse(update_input)
            if residual is not None: residual.add_(value)
            return carrier
        with patch.object(C, '_use_hcu_lightop_gemma_rmsnorm', True), patch.object(C, 'gemma_rms_norm_fp8_quant_hcu', side_effect=quant):
            self.assertIs(layer.forward_with_lightop_fp8_quant(x), carrier)
            post = torch.full_like(x, .5)
            returned, updated = layer.forward_with_lightop_fp8_quant(x, None, post)
            self.assertIs(returned, carrier); self.assertIs(updated, post)
            torch.testing.assert_close(updated, torch.full_like(x, 1.5))
            residual = torch.ones_like(x); post = torch.full_like(x, .5)
            returned, updated = layer.forward_with_lightop_fp8_quant(x, residual, post)
            self.assertIs(returned, carrier); self.assertIsNot(updated, residual)
            torch.testing.assert_close(updated, torch.full_like(x, 2.5))
            torch.testing.assert_close(residual, torch.ones_like(x))

    def test_non_hcu_preserves_effective_wrapped_callable(self):
        marker = object(); layer = object(); x = object()
        def original(owner, value, *args):
            self.assertIs(owner, layer); self.assertIs(value, x); return marker
        with patch('hcu_sglang.utils.detection.is_hcu', return_value=False):
            self.assertIs(Q.around_rms_norm_forward_hip(original, layer, x), marker)
        with patch.object(C, 'is_hcu', return_value=False):
            self.assertIs(C.around_gemma_forward_hip(original, layer, x), marker)


if __name__ == '__main__':
    unittest.main()
