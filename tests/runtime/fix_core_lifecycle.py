from __future__ import annotations

import importlib.util
from array import array
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.environ import envs


def load_module(name):
    baseline = os.environ.get('HCU_PLUGIN_BASELINE_CORE')
    if baseline:
        spec = importlib.util.spec_from_file_location('baseline_core_' + name, Path(baseline) / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return __import__('hcu_sglang.' + name, fromlist=[name])


R = load_module('prefill_request')
B = load_module('prefill_graph_sync')
E = load_module('environ')


class RequestAndBatchLifecycleTests(unittest.TestCase):
    def req(self):
        return R.Req('request', '', array('q', [1, 2, 3, 4, 5]), SamplingParams())

    def test_request_constructor_installs_source_state_defaults(self):
        req = self.req()
        self.assertIsNone(req.pd_hidden_max_prefix_len)
        self.assertIsNone(req.disagg_decode_prefix_len)
        self.assertIsNone(req.external_cache_hit_length)
        self.assertEqual(req.external_kv_cleanup_steps, {})
        self.assertFalse(req.external_kv_cleanup_done)
        self.assertFalse(req.mm_token_counts_ready)

    def test_multimodal_counts_scan_once_then_accumulate_new_items(self):
        req = self.req()
        merge = Mock()
        media = SimpleNamespace(compute_mm_token_counts=Mock(return_value=(3, 2, 1)), merge=merge)
        req.multimodal_inputs = media
        self.assertEqual(req.get_mm_token_counts(), (3, 2, 1))
        self.assertEqual(req.get_mm_token_counts(), (3, 2, 1))
        media.compute_mm_token_counts.assert_called_once_with()
        added = SimpleNamespace(compute_mm_token_counts=Mock(return_value=(4, 0, 2)))
        req.extend_image_inputs(added)
        merge.assert_called_once_with(added)
        self.assertEqual(req.get_mm_token_counts(), (7, 2, 3))
        other = self.req()
        other.inherit_mm_token_counts(req)
        self.assertEqual(other.get_mm_token_counts(), (7, 2, 3))

    def test_pd_hidden_prefix_cap_is_part_of_actual_request_matching_bound(self):
        req = self.req()
        req.pd_hidden_max_prefix_len = 2
        self.assertEqual(req._compute_max_prefix_len(5), 2)
        req.pd_hidden_max_prefix_len = None
        self.assertEqual(req._compute_max_prefix_len(5), 4)

    def test_batch_copy_preserves_the_cp_collective_layout(self):
        batch = B.ScheduleBatch(reqs=[], device='cpu')
        batch.global_cp_num_tokens = [0, 64]
        with patch.object(B.N if not hasattr(B, 'SB') else B.SB, 'ScheduleBatch', B.ScheduleBatch, create=True):
            copied = batch.copy()
        self.assertEqual(copied.global_cp_num_tokens, [0, 64])
        self.assertIs(copied.global_cp_num_tokens, batch.global_cp_num_tokens)

    def test_idle_transition_clears_host_sequence_metadata(self):
        batch = B.ScheduleBatch(reqs=[], device='cpu', model_config=SimpleNamespace(vocab_size=32))
        batch.orig_seq_lens_cpu = [11]
        native = B.SB if hasattr(B, 'SB') else __import__('sglang.srt.managers.schedule_batch', fromlist=['ScheduleBatch'])
        with patch.object(native.SamplingBatchInfo, 'from_schedule_batch', return_value=object()):
            batch.prepare_for_idle()
        self.assertIsNone(batch.orig_seq_lens_cpu)
        self.assertTrue(batch.forward_mode.is_idle())
        self.assertEqual(batch.seq_lens_cpu.numel(), 0)

    def test_decode_conversion_resets_no_encoder_windows(self):
        req = self.req()
        batch = B.ScheduleBatch(reqs=[req], device='cpu')
        batch.forward_mode = ForwardMode.DECODE
        batch.seq_lens_cpu = torch.tensor([6])
        batch.encoder_swa_reset = [True]
        batch.convert_decode_to_extend()
        self.assertEqual(batch.encoder_swa_reset, [False])
        self.assertEqual(batch.prefix_lens, [5])
        self.assertEqual(batch.extend_lens, [1])
        self.assertEqual(req.extend_end, 6)
        self.assertIs(batch.decoding_reqs, batch.reqs)

    def test_source_environment_alias_and_allocator_switch(self):
        E.register_envs()
        with patch.dict(os.environ, {'SGLANG_NSA_HCU_INT8_INDEX_K_CACHE': 'true',
                                   'SGLANG_LIGHTOP_KVALLOC_KERNEL': 'true'}, clear=False):
            os.environ.pop('SGLANG_DSA_HCU_INT8_INDEX_K_CACHE', None)
            self.assertTrue(envs.SGLANG_DSA_HCU_INT8_INDEX_K_CACHE.get())
            self.assertTrue(envs.SGLANG_LIGHTOP_KVALLOC_KERNEL.get())
        with patch.dict(os.environ, {'SGLANG_NSA_HCU_INT8_INDEX_K_CACHE': 'true',
                                   'SGLANG_DSA_HCU_INT8_INDEX_K_CACHE': 'false'}):
            self.assertFalse(envs.SGLANG_DSA_HCU_INT8_INDEX_K_CACHE.get())


if __name__ == '__main__':
    unittest.main()
