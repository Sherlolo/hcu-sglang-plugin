from __future__ import annotations

import dataclasses
import os
import pickle
from types import SimpleNamespace as S
from unittest.mock import Mock, patch
import unittest
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from sglang.srt.configs import mamba_utils as N, glm5_next as G
from hcu_sglang import configs_mamba as C, configs_mamba_glm as H
from sglang.srt.plugins.hook_registry import HookRegistry

_BASELINE_RING = [N.BaseLinearStateParams.replayssm_ring_bytes_per_req]
_NATIVE_TYPES = {name: getattr(N, name) for name in ["Mamba2StateShape", "Mamba2CacheParams", "KimiLinearStateShape", "KimiLinearCacheParams", "Mamba2StateDType"]}
_NATIVE_GLM_TEXT = G.Glm5NextTextConfig
_NATIVE_GLM_COMPOSITE = G.Glm5NextConfig


class KDAConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset()
        C.register_hooks()
        HookRegistry.apply_hooks()

    def params(self, dtype=torch.bfloat16):
        shape = N.KimiLinearStateShape.create(tp_world_size=2, num_heads=8, head_dim=16, num_k_heads=4, head_k_dim=8, conv_kernel_size=4)
        return N.KimiLinearCacheParams(shape=shape, layers=[2, 3], dtype=N.Mamba2StateDType(conv=dtype, temporal=torch.float32))

    def test_native_typed_factories_keep_identity_and_kda_layout_contract(self):
        for name, native in _NATIVE_TYPES.items():
            self.assertIs(getattr(N, name), native)
            self.assertIs(getattr(C, name), native)
        params = self.params()
        self.assertTrue(params.use_hcu_kda)
        self.assertEqual(params.shape.conv, [(3, 96)])
        self.assertEqual(params.shape.temporal, (4, 16, 16))
        self.assertEqual(params.shape.num_k_heads_per_tp, 2)
        self.assertEqual(params.shape.conv_slice_axis, 1)
        self.assertTrue(params.shape.disable_conv_window_dedup)
        self.assertEqual(params.shape.conv_shard_groups, [128, 32, 32])
        self.assertEqual(params.mamba_cache_per_req, 2 * (3 * 96 * 2 + 4 * 16 * 16 * 4))

    def test_real_hcu_ring_counterexample_omits_extra_d_k_scratch(self):
        params = self.params()
        expected = 2 * (4 * 3 * 16 * 2 + 2 * 3 * 16 * 2 + 4 * 3 * 4 + 4 * 3 * 16 * 4)
        old = _BASELINE_RING[0](params, 3)
        self.assertEqual(params.replayssm_ring_bytes_per_req(3), expected)
        self.assertEqual(old - expected, 2 * (4 * 3 * 16 * 2 + 2 * 3 * 16 * 2))
        self.assertEqual(params.replayssm_ring_bytes_per_req(0), 0)
        print("KDA ring baseline/new bytes:", old, expected)

    def test_non_hcu_ring_keeps_effective_baseline_and_property_false(self):
        marker = object()
        original = Mock(return_value=marker)
        params = self.params()
        with patch.object(C, "is_hcu", return_value=False), patch("hcu_sglang.utils.detection.is_hcu", return_value=False):
            self.assertFalse(params.use_hcu_kda)
            self.assertIs(C.around_replayssm_ring_bytes_per_req(original, params, 5), marker)
        original.assert_called_once_with(params, 5)

    def test_frozen_params_copy_and_pickle_keep_canonical_type(self):
        params = self.params()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            params.layers = [8]
        copied = dataclasses.replace(params, layers=[8])
        restored = pickle.loads(pickle.dumps(copied))
        self.assertIs(type(restored), _NATIVE_TYPES["KimiLinearCacheParams"])
        self.assertEqual(restored.layers, [8])
        self.assertTrue(restored.use_hcu_kda)
        self.assertEqual(restored.shape, params.shape)

    def text(self, **kwargs):
        return G.Glm5NextTextConfig(num_hidden_layers=4, hidden_size=64, num_attention_heads=4, linear_attn_config={"kda_layers":[0,2], "num_heads":8, "head_dim":16, "num_k_heads":4, "head_k_dim":8, "short_conv_kernel_size":4}, **kwargs)

    def test_glm_real_constructor_produces_source_cp_tp_heads_and_dtype(self):
        self.assertIs(G.Glm5NextTextConfig, _NATIVE_GLM_TEXT)
        self.assertIs(G.Glm5NextConfig, _NATIVE_GLM_COMPOSITE)
        config = self.text(mamba_ssm_dtype="float32")
        self.assertEqual(config.num_experts_per_tok, 8)
        self.assertEqual(config.hc_post_mult_value, 2.)
        self.assertEqual(config.linear_layer_ids, [0,2])
        parallel = S(attn_tp_size=4, attn_cp_size=2)
        with patch.object(H, "get_parallel", return_value=parallel), patch("sglang.srt.layers.attention.dsa.utils.is_dsa_enable_prefill_cp", return_value=True):
            cp = config.mamba2_cache_params
        with patch.object(H, "get_parallel", return_value=parallel), patch("sglang.srt.layers.attention.dsa.utils.is_dsa_enable_prefill_cp", return_value=False):
            tp = config.mamba2_cache_params
        self.assertIs(type(cp), N.KimiLinearCacheParams)
        self.assertEqual(cp.shape.temporal, (4,16,16))
        self.assertEqual(tp.shape.temporal, (2,16,16))
        self.assertEqual(cp.shape.conv, [(3,96)])
        self.assertEqual(tp.shape.conv, [(3,48)])
        self.assertEqual(cp.shape.num_k_heads_per_tp, 2)
        self.assertEqual(tp.shape.num_k_heads_per_tp, 1)
        self.assertEqual(cp.dtype.temporal, torch.float32)
        self.assertTrue(cp.use_hcu_kda)

    def test_glm_layer_partition_rejects_overlap_before_state_sizing(self):
        with self.assertRaisesRegex(ValueError, "cover the model exactly"):
            G.Glm5NextTextConfig(num_hidden_layers=4, linear_attn_config={"kda_layers":[0,1], "full_attn_layers":[1,2,3], "num_heads":8, "head_dim":16})

    def test_glm_composite_alias_registry_and_actual_vision_base(self):
        from transformers import AutoConfig
        from transformers.models.glm4v.configuration_glm4v import Glm4vVisionConfig
        from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_spec_by_arch
        text = {"num_hidden_layers":4, "hidden_size":64, "num_attention_heads":4, "linear_attn_config":{"kda_layers":[0,2], "num_heads":8, "head_dim":16, "num_k_heads":4, "head_k_dim":8, "short_conv_kernel_size":4}}
        config = AutoConfig.for_model("glm5v_next", text_config=text, vision_config={}, architectures=["Glm5vNextForConditionalGeneration"])
        self.assertIs(type(config.text_config), G.Glm5NextTextConfig)
        self.assertEqual(config.architectures, ["Glm5NextForConditionalGeneration"])
        self.assertIsInstance(config.vision_config, Glm4vVisionConfig)
        self.assertEqual(config.linear_layer_ids, [0,2])
        spec = get_linear_attn_spec_by_arch(config.architectures[0])
        self.assertIs(spec.config_class, G.Glm5NextConfig)
        self.assertTrue(spec.support_mamba_cache_extra_buffer)
        self.assertIsInstance(AutoConfig.for_model("glm5next_text", num_hidden_layers=4), G.Glm5NextTextConfig)


if __name__ == "__main__":
    unittest.main()
