from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers import norm_collectives as C
from sglang.srt import distributed as D
from sglang.srt.layers.quantization import fp8_utils as F
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.srt.distributed import tensor_model_parallel_fused_allreduce_rmsnorm as preimported_norm


class NormCollectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset(); C.register_hooks(); HookRegistry.apply_hooks()

    def layer(self, gemma=False):
        layer = C.N.GemmaRMSNorm(4) if gemma else C.N.RMSNorm(4)
        layer.forward = Mock(return_value=object())
        return layer

    def groups(self, plain=None, quant=None):
        scope = ExitStack(); self.addCleanup(scope.close)
        groups = {}
        for name in ('tp', 'attn_tp', 'moe_tp'):
            coordinator = SimpleNamespace(fused_allreduce_rmsnorm=Mock(return_value=plain), fused_allreduce_rmsnorm_quant_per_group=Mock(return_value=quant))
            groups[name] = coordinator
            scope.enter_context(patch.object(C.P, 'get_' + name + '_group', return_value=coordinator))
        return groups

    def config(self, *, moe_tp=2, attn_tp=4, enabled=True, aiter=False):
        scope = ExitStack(); self.addCleanup(scope.close)
        scope.enter_context(patch.object(C.N, 'get_parallel', return_value=SimpleNamespace(moe_ep_size=8, moe_tp_size=moe_tp, attn_tp_size=attn_tp)))
        scope.enter_context(patch.object(C.N, 'get_exec', return_value=SimpleNamespace(comm=SimpleNamespace(enable_aiter_allreduce_fusion=enabled))))
        scope.enter_context(patch.object(C.N, '_is_hip', True))
        scope.enter_context(patch.object(C.N, '_use_aiter', aiter))

    def test_public_and_preimported_alias_select_exact_coordinator(self):
        marker = object(); groups = self.groups(plain=marker, quant=marker)
        self.assertIs(preimported_norm, D.tensor_model_parallel_fused_allreduce_rmsnorm)
        x = torch.ones(2, 4); weight = torch.ones(4)
        for name in groups:
            self.assertIs(D.tensor_model_parallel_fused_allreduce_rmsnorm(x, x, weight, 1e-6, group=name), marker)
            groups[name].fused_allreduce_rmsnorm.assert_called_once_with(x, x, weight, 1e-6)
            self.assertIs(D.tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group(x, x, weight, 1e-6, 128, emit_bf16=True, transpose_scale=True, group=name), marker)
            groups[name].fused_allreduce_rmsnorm_quant_per_group.assert_called_once_with(x, x, weight, 1e-6, 128, emit_bf16=True, transpose_scale=True)
        for method in (D.tensor_model_parallel_fused_allreduce_rmsnorm, D.tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group):
            with self.assertRaisesRegex(ValueError, 'Unsupported fused all-reduce group'):
                method(x, x, weight, 1e-6, group='moe_ep')

    def test_default_group_is_tp(self):
        groups = self.groups(); x = torch.ones(1, 4)
        D.tensor_model_parallel_fused_allreduce_rmsnorm(x, x, x[0], 1e-6)
        groups['tp'].fused_allreduce_rmsnorm.assert_called_once()
        groups['attn_tp'].fused_allreduce_rmsnorm.assert_not_called()
        groups['moe_tp'].fused_allreduce_rmsnorm.assert_not_called()

    def test_public_norm_fusion_uses_comm_policy_independent_of_compute_aiter(self):
        marker = object(); groups = self.groups(plain=marker); self.config(aiter=False)
        layer = self.layer(); x = torch.ones(2, 4); residual = torch.ones_like(x)
        result = layer.forward_with_allreduce_fusion(x, residual, use_attn_tp_group=False)
        self.assertIs(result, marker)
        groups['moe_tp'].fused_allreduce_rmsnorm.assert_called_once()
        groups['tp'].fused_allreduce_rmsnorm.assert_not_called()
        layer.forward.assert_not_called()

    def test_declined_fusion_reduces_correct_group_once_before_norm(self):
        groups = self.groups(); self.config(); layer = self.layer()
        x = torch.ones(2, 4); residual = torch.full_like(x, 2.); post = torch.full_like(x, .5)
        with patch.object(D, 'moe_tensor_model_parallel_all_reduce', side_effect=lambda value: value * 2) as reduce, patch.object(D, 'attention_tensor_model_parallel_all_reduce', side_effect=AssertionError('wrong communication group')):
            result = layer.forward_with_allreduce_fusion(x, residual, post, use_attn_tp_group=False)
        self.assertIs(result, layer.forward.return_value); reduce.assert_called_once_with(x)
        args = layer.forward.call_args.args
        torch.testing.assert_close(args[0], x * 2)
        torch.testing.assert_close(args[1], residual + post)
        self.assertIsNone(args[2])
        torch.testing.assert_close(residual, torch.full_like(residual, 2.))

    def test_moe_ep_size_does_not_trigger_moe_tp_reduction(self):
        groups = self.groups(); self.config(moe_tp=1); layer = self.layer(); x = torch.ones(2, 4)
        layer.forward_with_allreduce_fusion(x, x, use_attn_tp_group=False)
        groups['moe_tp'].fused_allreduce_rmsnorm.assert_not_called()
        layer.forward.assert_called_once_with(x, x, None)

    def test_gemma_passes_weight_plus_one_to_attention_group(self):
        marker = object(); groups = self.groups(plain=marker); self.config()
        layer = self.layer(gemma=True); x = torch.ones(2, 4)
        self.assertIs(layer.forward_with_allreduce_fusion(x, x), marker)
        args = groups['attn_tp'].fused_allreduce_rmsnorm.call_args.args
        self.assertIs(args[2], layer.gemma_weight)

    def test_group_quant_returns_original_carriers_and_bf16_side_output(self):
        self.config(aiter=True); layer = self.layer(); x = torch.ones(2, 4)
        q = object(); scale = object(); updated = object(); bf16 = object()
        for keep in (False, True):
            carrier = (q, updated, scale, bf16) if keep else (q, updated, scale)
            groups = self.groups(quant=carrier)
            with patch.object(F, '_use_aiter_bpreshuffle_gfx95', False):
                result, actual_residual = layer.forward_with_allreduce_fusion_quant_per_group(x, x, keep_bf16=keep, use_attn_tp_group=False)
            self.assertIs(actual_residual, updated)
            self.assertEqual(result, (bf16, q, scale) if keep else (q, scale))
            kwargs = groups['moe_tp'].fused_allreduce_rmsnorm_quant_per_group.call_args.kwargs
            self.assertEqual(kwargs.get('emit_bf16', False), keep)
            self.assertIs(kwargs['transpose_scale'], False)

    def test_quant_decline_preserves_group_and_materializes_fallback_scale(self):
        self.config(aiter=True); layer = self.layer(); x = torch.ones(2, 4)
        bf16 = torch.full_like(x, 2.); residual_out = torch.full_like(x, 3.)
        groups = self.groups(plain=(bf16, residual_out), quant=None)
        q = object(); scale = object(); packed_scale = object(); quant = Mock(return_value=(q, scale))
        with patch.object(F, '_use_aiter_bpreshuffle_gfx95', True), patch.object(F, 'materialize_bpreshuffle_fp8_scale', return_value=packed_scale) as materialize, patch.object(C.N, '_get_aiter_per_group_quant', return_value=(quant, torch.float8_e4m3fn)):
            result, updated = layer.forward_with_allreduce_fusion_quant_per_group(x, x, use_attn_tp_group=False, keep_bf16=True)
        self.assertEqual(result, (bf16, q, packed_scale)); self.assertIs(updated, residual_out)
        groups['moe_tp'].fused_allreduce_rmsnorm.assert_called_once()
        groups['tp'].fused_allreduce_rmsnorm.assert_not_called()
        quant.assert_called_once_with(bf16, quant_dtype=torch.float8_e4m3fn, transpose_scale=False)
        materialize.assert_called_once_with(scale)

    def test_non_hcu_wrappers_preserve_effective_original(self):
        marker = object(); original = Mock(return_value=marker)
        with patch.object(C, 'is_hcu', return_value=False):
            for method in [C.around_fused_norm, C.around_fused_norm_quant, C.around_norm_forward, C.around_norm_quant_forward]:
                self.assertIs(method(original, 'input', setting='value'), marker)
                original.assert_called_with('input', setting='value')


if __name__ == '__main__':
    unittest.main()
