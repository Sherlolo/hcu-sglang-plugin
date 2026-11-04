from __future__ import annotations
import unittest
from unittest.mock import Mock,patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.layers.quantization import compressed_targets as C
from sglang.srt.layers.quantization.compressed_tensors.utils import find_matched_target as early
from sglang.srt.plugins.hook_registry import HookRegistry


class LinearStem(torch.nn.Module):pass
class HiddenProjection(LinearStem):pass
class MatchingTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();C.register_hooks();HookRegistry.apply_hooks()

    def test_inherited_target_matches_real_module_and_early_alias(self):
        self.assertEqual(early('decoder.proj',HiddenProjection(),['LinearStem']),'LinearStem')
        self.assertEqual(early(None,HiddenProjection(),['LinearStem']),'LinearStem')

    def test_layer_path_regex_and_leaf_class_take_priority(self):
        obj=HiddenProjection()
        self.assertEqual(early('decoder.proj',obj,['LinearStem','decoder.proj']),'decoder.proj')
        self.assertEqual(early('decoder.proj',obj,['re:decoder.*','LinearStem']),'re:decoder.*')
        self.assertEqual(early(None,obj,['LinearStem','HiddenProjection']),'HiddenProjection')

    def test_python_mro_order_precedes_target_list_order(self):
        class Other(torch.nn.Module):pass
        class Combined(Other,LinearStem):pass
        self.assertEqual(early(None,Combined(),['LinearStem','Other']),'Other')

    def test_fused_mapping_runs_only_after_class_chain_and_requires_all_components(self):
        obj=HiddenProjection();mapping={'qkv_proj':['q_proj','k_proj','v_proj']}
        targets=['decoder.q_proj','decoder.k_proj','decoder.v_proj']
        self.assertEqual(early('decoder.qkv_proj',obj,targets,mapping),'decoder.q_proj')
        self.assertEqual(early('decoder.qkv_proj',obj,targets+['LinearStem'],mapping),'LinearStem')
        with self.assertRaises(ValueError):early('decoder.qkv_proj',obj,targets[:2],mapping)

    def test_object_is_not_a_quantizable_target_and_empty_path_match_is_preserved(self):
        with self.assertRaises(ValueError):early(None,HiddenProjection(),['object'])
        self.assertEqual(early(None,HiddenProjection(),['','LinearStem']),'')

    def test_real_native_config_get_scheme_uses_inherited_match(self):
        from compressed_tensors.quantization import QuantizationArgs
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
        payload={'weights':QuantizationArgs(num_bits=8,type='int',strategy='channel'),'input_activations':None,'format':'int-quantized'}
        cfg=CompressedTensorsConfig({'LinearStem':payload},[],'int-quantized',{},[])
        self.assertIs(cfg.get_scheme_dict(HiddenProjection(),'decoder.proj'),payload)

    def test_non_hcu_preserves_effective_original(self):
        original=Mock(return_value='native');obj=HiddenProjection()
        with patch.object(C,'is_hcu',return_value=False):self.assertEqual(C.around_find_target(original,None,obj,[]),'native')
        original.assert_called_once()


if __name__=='__main__':unittest.main()
