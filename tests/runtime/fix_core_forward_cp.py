from __future__ import annotations

from array import array
import os
from types import SimpleNamespace as S
import unittest
from unittest.mock import Mock, patch

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang import cp_runtime as C, forward_batch as F
from hcu_sglang.prefill_graph_sync import ScheduleBatch
from hcu_sglang.prefill_request import Req
from hcu_sglang.disagg.hidden_state import get_pd_hidden_req_state
from hcu_sglang.models.deepseek.current_v4.hcu_backend.forward_batch import ForwardBatch
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.layers import dp_attention as D
from sglang.srt.plugins.hook_registry import HookRegistry


class PublicCpAndForwardMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset()
        C.register_hooks()
        from hcu_sglang import dsa_utils
        dsa_utils.register_hooks()
        F.register_hooks()
        HookRegistry.apply_hooks()

    def forward(self):
        return ForwardBatch(forward_mode=ForwardMode.EXTEND, batch_size=1,
            input_ids=torch.tensor([1, 2]), req_pool_indices=torch.tensor([0]),
            seq_lens=torch.tensor([2]), out_cache_loc=torch.tensor([0, 1]),
            seq_lens_sum=2, orig_seq_lens=torch.tensor([2]))

    def test_hcu_public_cp_predicate_respects_opt_in(self):
        batch = S(forward_mode=ForwardMode.EXTEND, input_ids=torch.arange(8),
                  global_cp_num_tokens_cpu=[4, 4])
        strategy = Mock(spec=C.C.ZigzagCPStrategy)
        strategy.can_apply.return_value = True
        with patch.dict(os.environ, {'SGLANG_ENABLE_CP_V2': 'false'}), \
             patch.object(C.C, 'get_cp_strategy', return_value=strategy):
            self.assertFalse(C.C.is_cp_active(batch))
        strategy.can_apply.assert_not_called()
        with patch.dict(os.environ, {'SGLANG_ENABLE_CP_V2': 'true'}), \
             patch.object(C.C, 'get_cp_strategy', return_value=strategy):
            self.assertTrue(C.C.is_cp_active(batch))

    def test_non_cp_peer_veto_is_shared_before_model_partition(self):
        batch = S(forward_mode=ForwardMode.EXTEND, input_ids=torch.arange(8),
                  global_cp_num_tokens_cpu=[0, 0])
        strategy = Mock(spec=C.C.ZigzagCPStrategy)
        strategy.can_apply.return_value = True
        with patch.dict(os.environ, {'SGLANG_ENABLE_CP_V2': 'true'}), \
             patch.object(C.C, 'get_cp_strategy', return_value=strategy):
            self.assertFalse(C.C.is_cp_active(batch))
        strategy.can_apply.assert_not_called()

    def test_cp_scope_changes_real_token_buffers_and_restores_on_failure(self):
        old_gpu = torch.tensor([8, 12])
        batch = S(global_cp_num_tokens_cpu=[4, 6], global_num_tokens_cpu=[8, 12],
            global_num_tokens_gpu=old_gpu, global_dp_buffer_len=24,
            dp_padding_mode=D.DpPaddingMode.MAX_LEN, dp_local_start_pos=2,
            dp_local_num_tokens=3, dp_local_token_index=None, cp_local_dp_layout=False)
        writes = []
        with patch.object(C.C, 'get_parallel', return_value=S(attn_dp_rank=1)), \
             patch.object(D, 'set_dp_buffer_len', side_effect=lambda *a: writes.append(a)):
            with self.assertRaisesRegex(RuntimeError, 'body failed'):
                with C.cp_local_dp_state(batch):
                    self.assertEqual(batch.global_num_tokens_cpu, [4, 6])
                    self.assertEqual(batch.global_num_tokens_gpu.tolist(), [4, 6])
                    self.assertEqual(batch.global_dp_buffer_len, 10)
                    self.assertEqual(batch.dp_local_token_index, 1)
                    self.assertEqual(batch.dp_padding_mode, D.DpPaddingMode.SUM_LEN)
                    self.assertTrue(batch.cp_local_dp_layout)
                    raise RuntimeError('body failed')
        self.assertEqual(batch.global_num_tokens_cpu, [8, 12])
        self.assertIs(batch.global_num_tokens_gpu, old_gpu)
        self.assertEqual(batch.global_dp_buffer_len, 24)
        self.assertFalse(batch.cp_local_dp_layout)
        self.assertEqual(writes[0][0:4], (10, 6, False, [4, 6]))
        self.assertEqual(writes[-1][0:4], (24, 12, True, [8, 12]))

    def test_deferred_metadata_preserves_cpu_counts_then_materializes(self):
        fb = self.forward()
        batch = ScheduleBatch(reqs=[], device='cpu', global_num_tokens=[8, 0],
            global_num_tokens_for_logprob=[2, 0], global_cp_num_tokens=[4, 0])
        with patch.object(C.C, 'get_cp_strategy', return_value=Mock(spec=C.C.ZigzagCPStrategy)):
            fb.init_mlp_sync_metadata(batch, 'cpu', materialize_device_tensors=False)
        self.assertEqual(fb.global_num_tokens_cpu, [8, 0])
        self.assertEqual(fb.global_cp_num_tokens_cpu, [4, 0])
        self.assertIsNone(fb.global_num_tokens_gpu)
        fb.materialize_deferred_device_metadata('cpu')
        self.assertEqual(fb.global_num_tokens_gpu.tolist(), [8, 0])
        self.assertEqual(fb.global_num_tokens_for_logprob_gpu.tolist(), [2, 0])

    def test_logical_mode_retains_decode_semantics_after_extend_conversion(self):
        fb = self.forward()
        fb._original_forward_mode = ForwardMode.DECODE
        self.assertEqual(fb.logical_forward_mode, ForwardMode.DECODE)
        fb._original_forward_mode = None
        self.assertEqual(fb.logical_forward_mode, ForwardMode.EXTEND)

    def test_pd_capture_layer_disagreement_uses_the_actual_cp_group(self):
        req = Req('pd', '', array('q', [1, 2]), SamplingParams())
        get_pd_hidden_req_state(req).capture_layer_ids = [2, 4]
        batch = ScheduleBatch(reqs=[req], device='cpu')
        group = object()
        def gather(output, local, *, group):
            self.assertEqual(local, [2, 4])
            output[:] = [[2, 4], [2, 5]]
        with patch.object(F.N, 'get_parallel', return_value=S(
                attn_cp_size=2, attn_cp_group=S(cpu_group=group))), \
             patch.object(torch.distributed, 'all_gather_object', side_effect=gather) as call:
            with self.assertRaisesRegex(RuntimeError, 'capture layers disagree'):
                ForwardBatch.init_new(batch, S(server_args=S(disaggregation_mode='prefill')),
                    return_hidden_states_before_norm=False)
        self.assertIs(call.call_args.kwargs['group'], group)


    def test_public_dsa_consumer_keeps_decode_semantics_after_relabel(self):
        from hcu_sglang import dsa_utils as A
        batch = S(forward_mode=ForwardMode.EXTEND,
                  _original_forward_mode=ForwardMode.DECODE,
                  attn_cp_metadata=object())
        with patch.object(A.U, 'is_dsa_enable_prefill_cp', return_value=True):
            self.assertFalse(A.U.dsa_use_prefill_cp(batch))
        batch._original_forward_mode = None
        with patch.object(A.U, 'is_dsa_enable_prefill_cp', return_value=True):
            self.assertTrue(A.U.dsa_use_prefill_cp(batch))

    def test_round_robin_sequence_partition_matches_global_token_ownership(self):
        from hcu_sglang import dsa_utils as A
        for lengths in ([0, 0], [1], [5, 2, 1], [0, 9, 2, 8]):
            for size in (2, 3, 4):
                for rank in range(size):
                    offset = 0
                    counts, indices = [], []
                    for i, length in enumerate(lengths):
                        count = sum(token % size == rank for token in range(offset, offset + length))
                        if count:
                            counts.append(count)
                            indices.append(i)
                        offset += length
                    with patch.object(A.U, 'get_parallel', return_value=S(attn_cp_size=size, attn_cp_rank=rank)):
                        self.assertEqual(A.U.dsa_cp_round_robin_split_q_seqs_cpu(lengths), (counts, indices))

    def test_public_round_robin_accepts_python_sequences_and_keeps_their_types(self):
        from hcu_sglang import dsa_utils as A
        with patch.object(A.U, 'get_parallel', return_value=S(attn_cp_size=3, attn_cp_rank=1)):
            self.assertEqual(A.U.dsa_cp_round_robin_split_data([0, 1, 2, 3, 4, 5, 6]), [1, 4])
            self.assertEqual(A.U.dsa_cp_round_robin_split_data((0, 1, 2, 3, 4, 5, 6)), (1, 4))
            self.assertEqual(A.U.dsa_cp_round_robin_split_data([]), [])

    def test_glm5_pd_seed_policy_uses_real_kpool_and_decode_ownership(self):
        from hcu_sglang import glm5_next_policy as P
        config = S(hf_config=S(model_type='glm5_next', index_kpool=1))
        args = S(disaggregation_mode='decode', enable_hisparse=False, attn_cp_size=1)
        with patch.dict(os.environ, {'SGLANG_DSA_FUSE_TOPK': 'true'}):
            self.assertFalse(P.should_remap_pd_dsa_seed_to_local_slots(args, config))
            config.hf_config.index_kpool = 8
            self.assertTrue(P.should_remap_pd_dsa_seed_to_local_slots(args, config))
            args.attn_cp_size = 2
            self.assertFalse(P.should_remap_pd_dsa_seed_to_local_slots(args, config))
            args.attn_cp_size = 1
            args.enable_hisparse = True
            self.assertFalse(P.should_remap_pd_dsa_seed_to_local_slots(args, config))
            args.enable_hisparse = False
            args.disaggregation_mode = 'prefill'
            self.assertFalse(P.should_remap_pd_dsa_seed_to_local_slots(args, config))

    @unittest.skipUnless(os.environ.get('HCU_PLUGIN_RUN_GPU_TESTS') == '1', 'explicit GPU lane')
    def test_round_robin_public_kernel_uses_the_same_sequence_ownership(self):
        from hcu_sglang import dsa_utils as A
        for lengths in ([0, 0], [1], [5, 2, 1], [0, 9, 2, 8]):
            tensor = torch.tensor(lengths, dtype=torch.int32, device='cuda')
            for size in (2, 3, 4):
                for rank in range(size):
                    with patch.object(A.U, 'get_parallel', return_value=S(attn_cp_size=size, attn_cp_rank=rank)):
                        counts, gpu_counts, indices, gpu_indices = A.U.dsa_cp_round_robin_split_q_seqs(lengths, tensor)
                    self.assertEqual(gpu_counts.cpu().tolist(), counts)
                    self.assertEqual(gpu_indices.cpu().tolist(), indices)


class ParallelOptionPublicationTests(unittest.TestCase):
    def test_declared_derived_options_publish_and_allow_scoped_overrides(self):
        from hcu_sglang.config import register_parallel_options
        from sglang.srt import runtime_context as R
        from sglang.srt.arg_groups.fields.parallel import Parallel
        register_parallel_options()
        class PluginParallel:
            _NS_PATH = "parallel"
            minimax_opt = Parallel.minimax_opt
            hy3_sp = Parallel.hy3_sp
        class Input:
            _NAMESPACES = (PluginParallel,)
        seed = Input()
        bag = R._ConfigBag('parallel')
        with patch.dict(os.environ, {'HCU_PLUGIN_MINIMAX_OPT': '1', 'HCU_PLUGIN_HY3_SP': '0'}):
            R._install_derived_leaves({'parallel': bag}, seed)
        self.assertTrue(bag.minimax_opt)
        self.assertFalse(bag.hy3_sp)
        self.assertFalse(hasattr(seed, 'minimax_opt'))
        self.assertFalse(hasattr(seed, 'hy3_sp'))
        parallel = R.ParallelContext()
        parallel._config = bag
        with parallel.override(minimax_opt=False, hy3_sp=True):
            self.assertFalse(parallel.minimax_opt)
            self.assertTrue(parallel.hy3_sp)
        self.assertTrue(parallel.minimax_opt)
        self.assertFalse(parallel.hy3_sp)


    def test_legacy_cp_aliases_follow_the_resolved_modern_layout(self):
        from hcu_sglang.config import register_parallel_options
        from sglang.srt import runtime_context as R
        from sglang.srt.arg_groups.fields.parallel import Parallel
        register_parallel_options()
        class LegacyParallel:
            _NS_PATH = 'parallel'
            enable_dsa_prefill_context_parallel = Parallel.enable_dsa_prefill_context_parallel
            enable_prefill_context_parallel = Parallel.enable_prefill_context_parallel
            dsa_prefill_cp_mode = Parallel.dsa_prefill_cp_mode
            prefill_cp_mode = Parallel.prefill_cp_mode
        class Input:
            _NAMESPACES = (LegacyParallel,)
        for enabled, strategy, backend, expected in [
            (False, None, 'dsa', (False, False, 'round-robin-split', 'in-seq-split')),
            (True, 'interleave', 'dsa', (True, False, 'round-robin-split', 'round-robin-split')),
            (True, 'zigzag', 'triton', (False, True, 'in-seq-split', 'in-seq-split')),
        ]:
            seed = Input()
            seed.enable_prefill_cp, seed.cp_strategy, seed.attention_backend = enabled, strategy, backend
            bag = R._ConfigBag('parallel')
            R._install_derived_leaves({'parallel': bag}, seed)
            self.assertEqual((bag.enable_dsa_prefill_context_parallel,
                bag.enable_prefill_context_parallel, bag.dsa_prefill_cp_mode,
                bag.prefill_cp_mode), expected)


if __name__ == '__main__':
    unittest.main()
