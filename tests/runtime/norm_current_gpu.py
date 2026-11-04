from __future__ import annotations

import json
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers import norm_current as C


def reference(value, weight, eps):
    return (value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + eps) * weight.float()).to(value.dtype)


class NormGpuTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(C.is_hcu())
        self.assertFalse(C.N.is_batch_invariant_mode_enabled())
        torch.manual_seed(738)

    def make_layer(self, gemma=False, dtype=torch.bfloat16):
        layer = (C.N.GemmaRMSNorm(256) if gemma else C.N.RMSNorm(256, weight_dtype=dtype)).to(device='cuda', dtype=dtype)
        values = torch.randn(256, device='cuda', dtype=dtype) * .2
        if gemma: layer.weight.weight_loader(layer.weight, values)
        else: layer.weight.data.copy_(values + 1)
        return layer

    def test_rms_plain_and_fused_noncontiguous_against_fp32_reference(self):
        for dtype in (torch.bfloat16, torch.float16):
            layer = self.make_layer(dtype=dtype)
            tolerance = .02 if dtype == torch.bfloat16 else .002
            for mode in ('plain', 'residual', 'post_residual'):
                x = torch.randn(8, 512, device='cuda', dtype=dtype)[:, ::2]
                residual = None if mode == 'plain' else torch.randn_like(x)
                post = torch.randn_like(x) if mode == 'post_residual' else None
                effective = None if residual is None else residual if post is None else residual + post
                value = x.float() if effective is None else x.float() + effective.float()
                expected = reference(value, layer.weight, layer.variance_epsilon).to(dtype)
                result = C.rms_forward_hip(layer, x, residual, post)
                if residual is None: output = result
                else:
                    output, updated = result
                    torch.testing.assert_close(updated, value.to(dtype), rtol=tolerance, atol=tolerance)
                    if post is None: self.assertIs(updated, residual)
                torch.testing.assert_close(output, expected, rtol=tolerance, atol=tolerance)

    def test_gemma_plain_and_fused_against_fp32_reference(self):
        self.assertFalse(C.N._use_aiter)
        layer = self.make_layer(gemma=True)
        for with_residual in (False, True):
            x = torch.randn(8, 512, device='cuda', dtype=torch.bfloat16)[:, ::2]
            residual = torch.randn_like(x) if with_residual else None
            value = x.float() if residual is None else x.float() + residual.float()
            expected = reference(value, layer.gemma_weight, layer.variance_epsilon).to(x.dtype)
            result = C.gemma_forward_hip(layer, x, residual)
            if residual is not None:
                output, updated = result
                torch.testing.assert_close(updated, value.to(x.dtype), rtol=.02, atol=.02)
            else: output = result
            torch.testing.assert_close(output, expected, rtol=.02, atol=.02)

    def test_residual_gpu_graph_capture_and_replay(self):
        layer = self.make_layer()
        original_x = torch.randn(8, 256, device='cuda', dtype=torch.bfloat16)
        original_residual = torch.randn_like(original_x)
        x = original_x.clone(); residual = original_residual.clone()
        for _ in range(3):
            x.copy_(original_x); residual.copy_(original_residual)
            C.rms_forward_hip(layer, x, residual)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): output, updated = C.rms_forward_hip(layer, x, residual)
        for shift in (0., .25):
            x.copy_(original_x + shift); residual.copy_(original_residual)
            value = (original_x + shift).float() + original_residual.float()
            expected = reference(value, layer.weight, layer.variance_epsilon).to(x.dtype)
            graph.replay(); torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, rtol=.02, atol=.02)
            torch.testing.assert_close(updated, value.to(x.dtype), rtol=.02, atol=.02)


if __name__ == '__main__':
    unittest.main()
