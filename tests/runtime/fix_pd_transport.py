from __future__ import annotations

from contextlib import ExitStack
import ctypes
import asyncio
import json
import os
import socket
import threading
import time
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

import numpy as np
import requests
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.disagg import integration, common as C, transport_common as TC, transport_mooncake as T, mooncake as M
from hcu_sglang.disagg.common_utils import PDHiddenChunk, PDHiddenRequestState, TransferKVChunk
from hcu_sglang.disagg.utils import PDHiddenRowPool
from hcu_sglang.disagg.protocol import StateType
from sglang.srt.plugins.hook_registry import HookRegistry
import sglang.srt.disaggregation.common.conn as NC
import sglang.srt.disaggregation.mooncake.conn as NM
from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.utils import DisaggregationMode


class BoundedCPUTransport:
    regions = []
    lock = threading.Lock()
    def __init__(self, session):
        self.session = session
        self.copies = []
    def get_session_id(self):
        return self.session
    def batch_register(self, ptrs, sizes):
        with self.lock:
            self.regions.extend(zip(ptrs, sizes))
        return 0
    def batch_deregister(self, ptrs):
        with self.lock:
            self.regions[:] = [(p, size) for p, size in self.regions if p not in ptrs]
        return 0
    def batch_transfer_sync(self, session, src, dst, sizes):
        with self.lock:
            for s, d, n in zip(src, dst, sizes):
                if not all(any(p <= x and x + n <= p + size for p, size in self.regions) for x in [s, d]):
                    raise RuntimeError("outside test-owned registered buffers")
            for s, d, n in zip(src, dst, sizes):
                ctypes.memmove(d, s, n)
                self.copies.append((s, d, n))
        return 0


def eventually(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.01)
    raise AssertionError("timed out waiting for owned transport state")


class PDTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset()
        from hcu_sglang import configs_mamba
        from hcu_sglang.mem_cache import mamba_pool
        configs_mamba.register_hooks()
        mamba_pool.register_hooks()
        integration.register_hooks()
        HookRegistry.apply_hooks()

    def test_request_state_rejects_future_stale_and_invalid_final_ranges(self):
        state = PDHiddenRequestState.streaming_state(4, 8)
        make = lambda start, rows, last: PDHiddenChunk(1, 0, start, rows, last, list(range(rows)))
        self.assertEqual(state.accept_chunk(make(6, 2, True)), "future")
        self.assertEqual(state.accept_chunk(make(3, 1, False)), "stale")
        with self.assertRaisesRegex(RuntimeError, "unexpected offset"):
            state.accept_chunk(make(4, 2, True))
        self.assertEqual(state.next_start, 4)
        self.assertEqual(state.accept_chunk(make(4, 2, False)), "accepted")
        self.assertEqual(state.accept_chunk(make(6, 2, True), defer_hidden_done=True), "accepted")
        state.mark_kv_done()
        self.assertFalse(state.request_done())
        state.mark_hidden_done()
        self.assertTrue(state.request_done())

    def args(self, pool):
        args = KVArgs()
        args._kv_fixture = [torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int32)]
        args.__dict__.update(engine_rank=0, kv_data_ptrs=[x.data_ptr() for x in args._kv_fixture], kv_data_lens=[8, 8], kv_item_lens=[4, 4], kv_layer_ids=[], kv_cache_dtype_str="bf16", kv_layout=None, aux_data_ptrs=[], aux_data_lens=[], aux_item_lens=[], state_types=[StateType.PD_HIDDEN], state_data_ptrs=[[pool.buffer.data_ptr()]], state_data_lens=[[pool.buffer.numel() * pool.buffer.element_size()]], state_item_lens=[[pool.hidden_size * pool.buffer.element_size()]], state_data_formats=[""], state_layer_ids=[[]], state_dim_per_tensor=[[]], state_slice_outer_counts=[[]], state_conv_shard_groups=[[]], is_hybrid_mla_backend=False, gpu_id=0, page_size=1, system_dp_rank=0, pp_rank=0, prefill_start_layer=0, prefill_end_layer=1, num_draft_entries=0, rust_http_port=None, kv_cache_layout=None, cp_cache_layer_split=False, kv_head_num=1, total_kv_head_num=1, swa_kv_cache_layout=None, ib_device=None, mla_compression_ratios=None)
        return args

    def test_actual_http_bootstrap_zmq_hidden_bytes_and_ack_source_release(self):
        self.run_transport(False)

    def test_actual_transport_preserves_reordered_state_component_indices(self):
        self.run_transport(True)

    def test_streaming_two_chunks_reuse_decode_window_after_ack(self):
        self.run_transport(False, chunks_count=2)

    def test_dynamic_pp_hidden_packets_copy_all_fragments_before_kv_completion(self):
        self.run_transport(False, dynamic_packets=True)

    def run_transport(self, reorder, chunks_count=1, dynamic_packets=False):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        bootstrap = NC.CommonKVBootstrapServer("127.0.0.1", port)
        def close_bootstrap():
            async def cancel_tasks():
                tasks = list(bootstrap._background_tasks)
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            if hasattr(bootstrap, "_loop") and bootstrap._loop.is_running():
                asyncio.run_coroutine_threadsafe(cancel_tasks(), bootstrap._loop).result(timeout=2)
            bootstrap.close()
        self.addCleanup(close_bootstrap)
        eventually(lambda: self.http_ready(port))
        parallel = S(attn_tp_size=1, attn_tp_rank=0, attn_cp_size=1, attn_cp_rank=0, attn_dcp_size=1, attn_dcp_rank=0, attn_dp_size=1, attn_dp_rank=0, dp_size=1, pp_size=1, pp_rank=0, nnodes=1, num_dp_ranks=1, enable_cp_tp_group_sharing=False, enable_dsa_cache_layer_split=False, dist_init_addr=None, load_balance_method="follow_bootstrap_room", pp_group=None)
        scope = ExitStack()
        self.addCleanup(scope.close)
        scope.enter_context(patch.dict(os.environ, {"SGLANG_DISAGGREGATION_THREAD_POOL_SIZE":"1", "SGLANG_DISAGGREGATION_QUEUE_SIZE":"1", "SGLANG_ENABLE_FAILED_SESSION_PROBE":"0", "SGLANG_DISAGG_STAGING_BUFFER":"0", "SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE":"0"}))
        for mod in [C, NC, TC]:
            scope.enter_context(patch.object(mod, "get_parallel", return_value=parallel))
        scope.enter_context(patch.object(C, "get_serving", return_value=S(host="127.0.0.1", port=port)))
        disagg = S(disaggregation_bootstrap_port=port, disaggregation_decode_allocation_policy="early")
        for mod in [C, NC]:
            scope.enter_context(patch.object(mod, "get_disagg", return_value=disagg))
        scope.enter_context(patch.object(C, "linear_attn_tp_size", return_value=1))
        scope.enter_context(patch.object(C, "get_local_ip_auto", return_value="127.0.0.1"))
        scope.enter_context(patch.object(C.ModelConfig, "from_server_args", return_value=S(hf_config=S(num_hidden_layers=1, model_type="fixture"))))
        scope.enter_context(patch.object(T, "get_observability", return_value=S(enable_trace=False)))
        for mod in [M, T, NM]:
            scope.enter_context(patch.object(mod, "get_memory", return_value=S(enable_unified_memory=False)))
        src = PDHiddenRowPool(8, 4, torch.bfloat16)
        dst = PDHiddenRowPool(8, 4, torch.bfloat16)
        source_engine = BoundedCPUTransport("source:1")
        decode_engine = BoundedCPUTransport("decode:2")
        scope.enter_context(patch.object(NM, "get_mooncake_transfer_engine", side_effect=[source_engine, decode_engine]))
        server = S(enable_pd_role_switch=True)
        src_args = self.args(src)
        src_args._kv_fixture[0].copy_(torch.tensor([21, 22], dtype=torch.int32))
        src_args._kv_fixture[1].copy_(torch.tensor([31, 32], dtype=torch.int32))
        prefill = NM.MooncakeKVManager(src_args, DisaggregationMode.PREFILL, server)
        self.addCleanup(prefill.teardown)
        prefill.pd_hidden_pool = src
        dst_args = self.args(dst)
        if reorder:
            dst_args.state_types.insert(0, StateType.REQUEST_ROW)
            dst_args.state_data_formats.insert(0, "")
            for name in ["state_data_ptrs", "state_data_lens", "state_item_lens", "state_layer_ids", "state_dim_per_tensor", "state_slice_outer_counts", "state_conv_shard_groups"]:
                getattr(dst_args, name).insert(0, [])
        decode = NM.MooncakeKVManager(dst_args, DisaggregationMode.DECODE, server)
        self.addCleanup(decode.teardown)
        addr = f"127.0.0.1:{port}"
        self.assertTrue(decode.try_ensure_parallel_info(addr))
        room = 91031
        sender = NM.MooncakeKVSender(prefill, addr, room)
        sender.init(chunks_count, aux_index=0)
        receiver = NM.MooncakeKVReceiver(decode, addr, room)
        receiver.init(prefill_dp_rank=0)
        source_rows = src.alloc(2 * chunks_count)
        destination_rows = dst.alloc(2)
        values = torch.arange(8 * chunks_count, dtype=torch.bfloat16).reshape(2 * chunks_count, 4)
        src.buffer[source_rows] = values
        metadata = {"pd_hidden": True, "streaming_hidden": not dynamic_packets}
        if dynamic_packets:
            row_bytes = dst.hidden_size * dst.buffer.element_size()
            metadata["pp_slices"] = {"0": {"dst_indices": destination_rows, "dynamic_dst": {"ptr": dst.buffer.data_ptr(), "item_len": row_bytes, "row_chunks": [{"row_start": 0, "row_len": 1, "ptr": dst.buffer.data_ptr()}, {"row_start": 1, "row_len": 1, "ptr": dst.buffer.data_ptr() + row_bytes}]}}}
        receiver.send_metadata(np.arange(chunks_count, dtype=np.int32), aux_index=0, state_indices=([[], destination_rows] if reorder else [destination_rows]), spec_metadata=metadata)
        eventually(lambda: prefill.request_status.get(room) == KVPoll.WaitingForInput)
        if dynamic_packets:
            sender.send(np.array([0], dtype=np.int32), state_indices=[source_rows])
            eventually(lambda: sender.is_transfer_drained())
            eventually(lambda: decode.check_status(room) == KVPoll.Success)
            torch.testing.assert_close(dst.buffer[destination_rows], values, rtol=0, atol=0)
            self.assertEqual(decode.pop_pd_hidden_ready_chunks(room), [])
            self.assertEqual(src.available_size(), 8)
            self.assertTrue(prefill.pop_pd_hidden_request_done(room))
            state = PDHiddenRequestState.full(4, 6)
            state.mark_kv_done()
            self.assertTrue(state.request_done())
            sender.clear()
            receiver.clear()
            return
        state = PDHiddenRequestState.streaming_state(4, 4 + 2 * chunks_count)
        for chunk_id in range(chunks_count):
            is_last = chunk_id == chunks_count - 1
            hidden_start = 4 + 2 * chunk_id
            rows = source_rows[2 * chunk_id : 2 * chunk_id + 2]
            sender.set_pd_hidden_chunk_meta(hidden_start, 2, is_last, release_indices=rows)
            sender.send(np.array([chunk_id], dtype=np.int32), state_indices=[rows])
            chunks = eventually(lambda: decode.pop_pd_hidden_ready_chunks(room))
            self.assertEqual(len(chunks), 1)
            chunk = PDHiddenChunk(**chunks[0])
            self.assertEqual(chunk.dst_indices, destination_rows)
            torch.testing.assert_close(dst.buffer[destination_rows], values[2 * chunk_id : 2 * chunk_id + 2], rtol=0, atol=0)
            self.assertEqual(src.available_size(), 8 - 2 * chunks_count + 2 * chunk_id)
            self.assertFalse(sender.is_transfer_drained())
            self.assertEqual(state.accept_chunk(chunk, defer_hidden_done=True), "accepted")
            self.assertFalse(state.hidden_request_done())
            decode.submit_pd_hidden_chunk_ack(event=None, remote=chunk.ack_host, dst_port=chunk.ack_port, room=room, prefill_rank=chunk.prefill_rank, hidden_start=chunk.hidden_start, is_last_hidden_chunk=chunk.is_last_hidden_chunk)
            eventually(lambda: sender.is_transfer_drained())
            eventually(lambda: src.available_size() == 8 - 2 * chunks_count + 2 * (chunk_id + 1))
            self.assertEqual(prefill.pop_pd_hidden_request_done(room), is_last)
            acked = eventually(lambda: decode.pop_pd_hidden_acked_chunks(room))
            self.assertEqual(acked[0]["hidden_start"], hidden_start)
        eventually(lambda: decode.check_status(room) == KVPoll.Success)
        state.mark_kv_done()
        state.mark_hidden_done()
        self.assertTrue(state.request_done())
        for source, destination in zip(src_args._kv_fixture, dst_args._kv_fixture):
            torch.testing.assert_close(destination[:chunks_count], source[:chunks_count])
        torch.testing.assert_close(dst.buffer[2:], torch.zeros_like(dst.buffer[2:]))
        self.assertGreater(len(source_engine.copies), 0)
        sender.clear()
        receiver.clear()

    def test_mha_mtp_pp_pointer_plan_keeps_separate_main_and_draft_ranges(self):
        manager = NC.CommonKVManager.__new__(NC.CommonKVManager)
        manager.kv_args = S(total_main_kv_layers=4, total_draft_kv_layers=2, prefill_main_start_layer=2, prefill_main_end_layer=4, prefill_draft_start_layer=1, prefill_draft_end_layer=2)
        result = manager.get_mha_kv_ptrs_with_pp([11, 12, 13, 21, 22, 23], list(range(100, 112)))
        self.assertEqual(result, ([11, 12, 13], [21, 22, 23], [102, 103, 105], [108, 109, 111], 3))
        manager.kv_args.prefill_main_end_layer = None
        with self.assertRaisesRegex(ValueError, "requires prefill main/draft"):
            manager.get_mha_kv_ptrs_with_pp([11, 12, 13, 21, 22, 23], list(range(100, 112)))

    def test_ambiguous_source_publication_cannot_report_drained(self):
        from hcu_sglang.disagg.transport_source_drain import SourceDrainTracker
        tracker = SourceDrainTracker()
        chunk = S(room=18)
        published = []
        class Queue:
            def put(self, item):
                published.append(item)
                raise RuntimeError("ambiguous publication")
        with self.assertRaisesRegex(RuntimeError, "ambiguous publication"):
            tracker.publish(chunk, Queue(), lambda: True)
        self.assertEqual(published, [chunk])
        self.assertFalse(tracker.drained(18))
        tracker.complete(chunk)
        self.assertFalse(tracker.drained(18))

    def test_real_hcu_kda_state_producer_to_heterogeneous_tp_byte_consumer(self):
        from sglang.srt.configs.mamba_utils import KimiLinearStateShape, KimiLinearCacheParams, Mamba2StateDType
        from sglang.srt.mem_cache.memory_pool import MambaPool, HybridLinearKVPool
        from sglang.srt.disaggregation import utils as U
        from hcu_sglang.disagg.protocol import KVArgsRegisterInfo, TransferInfo
        def producer(tp):
            shape = KimiLinearStateShape.create(tp_world_size=tp, num_heads=4, head_dim=4, num_k_heads=2, head_k_dim=4, conv_kernel_size=4)
            params = KimiLinearCacheParams(shape=shape, layers=[2, 3], dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32))
            pool = MambaPool(size=2, spec_state_size=0, cache_params=params, mamba_layer_ids=params.layers, device="cpu", enable_memory_saver=False)
            full_fixture = S(get_kv_size_bytes=lambda: (0, 0))
            hybrid = HybridLinearKVPool(size=2, dtype=torch.bfloat16, page_size=1, head_num=1, head_dim=4, full_attention_layer_ids=[], device="cpu", mamba_pool=pool, full_kv_pool=full_fixture)
            args = KVArgs()
            args.engine_rank = 0
            U.setup_state_kv_args(args, hybrid)
            self.assertTrue(args.hcu_kda_state_sharded)
            self.assertEqual(args.state_types, [StateType.MAMBA])
            return pool, args
        source, args = producer(1)
        source.mamba_cache.conv[0].copy_(torch.arange(source.mamba_cache.conv[0].numel()).reshape(source.mamba_cache.conv[0].shape).to(torch.bfloat16))
        source.mamba_cache.temporal.copy_(torch.arange(source.mamba_cache.temporal.numel()).reshape(source.mamba_cache.temporal.shape).float())
        source_conv = source.mamba_cache.conv[0].clone()
        source_temporal = source.mamba_cache.temporal.clone()
        manager = NM.MooncakeKVManager.__new__(NM.MooncakeKVManager)
        manager.kv_args = args
        manager.attn_cp_size = 1
        manager.attn_cp_rank = 0
        manager.attn_tp_size = manager.state_tp_size = 1
        manager.attn_tp_rank = 0
        manager.pp_size = 1
        manager.cp_cache_layer_split = False
        engine = BoundedCPUTransport("kda-source:1")
        manager.engine = engine
        for ptrs, sizes in zip(args.state_data_ptrs, args.state_data_lens):
            engine.batch_register(ptrs, sizes)
        req = TransferInfo(room=32, endpoint="127.0.0.1", dst_port=1, mooncake_session_id="kda-dest:2", dst_kv_indices=np.empty(0, dtype=np.int32), dst_aux_index=0, dst_state_indices=[[2]], required_dst_info_num=1, is_dummy=False)
        try:
            for rank in [0, 1]:
                destination, dst = producer(2)
                destination.mamba_cache.conv[0].fill_(-7)
                destination.mamba_cache.temporal.fill_(-9)
                for ptrs, sizes in zip(dst.state_data_ptrs, dst.state_data_lens):
                    engine.batch_register(ptrs, sizes)
                info = KVArgsRegisterInfo(room="None", endpoint="127.0.0.1", dst_port=1, mooncake_session_id=req.mooncake_session_id, dst_kv_ptrs=[], dst_aux_ptrs=[], dst_state_data_ptrs=dst.state_data_ptrs, dst_tp_rank=rank, dst_attn_tp_size=2, dst_kv_item_len=0, dst_state_item_lens=dst.state_item_lens, dst_state_dim_per_tensor=dst.state_dim_per_tensor, dst_kv_layer_ids=[], dst_state_layer_ids=dst.state_layer_ids, dst_state_types=dst.state_types)
                with patch.object(manager, "_send_hcu_kda_state", wraps=manager._send_hcu_kda_state) as consumer:
                    self.assertEqual(manager.maybe_send_extra(req, [[1]], None, info), 0)
                consumer.assert_called_once()
                groups = [16, 8, 8]
                expected = []
                offset = 0
                for width in groups:
                    expected.append(source_conv[:, 1, :, offset + rank * (width // 2):offset + (rank + 1) * (width // 2)])
                    offset += width
                torch.testing.assert_close(destination.mamba_cache.conv[0][:, 2], torch.cat(expected, dim=-1), rtol=0, atol=0)
                torch.testing.assert_close(destination.mamba_cache.temporal[:, 2], source_temporal[:, 1, rank*2:(rank+1)*2], rtol=0, atol=0)
                torch.testing.assert_close(destination.mamba_cache.conv[0][:, :2], torch.full_like(destination.mamba_cache.conv[0][:, :2], -7))
                torch.testing.assert_close(destination.mamba_cache.temporal[:, :2], torch.full_like(destination.mamba_cache.temporal[:, :2], -9))
                for ptrs in dst.state_data_ptrs:
                    engine.batch_deregister(ptrs)
        finally:
            for ptrs in args.state_data_ptrs:
                engine.batch_deregister(ptrs)
        destination, dst = producer(1)
        destination.mamba_cache.conv[0].fill_(-17)
        destination.mamba_cache.temporal.fill_(-19)
        for ptrs, sizes in zip(dst.state_data_ptrs, dst.state_data_lens):
            engine.batch_register(ptrs, sizes)
        info = KVArgsRegisterInfo(room="None", endpoint="127.0.0.1", dst_port=1, mooncake_session_id=req.mooncake_session_id, dst_kv_ptrs=[], dst_aux_ptrs=[], dst_state_data_ptrs=dst.state_data_ptrs, dst_tp_rank=0, dst_attn_tp_size=1, dst_kv_item_len=0, dst_state_item_lens=dst.state_item_lens, dst_state_dim_per_tensor=dst.state_dim_per_tensor, dst_kv_layer_ids=[], dst_state_layer_ids=dst.state_layer_ids, dst_state_types=dst.state_types)
        for cp_rank in [0, 1]:
            shard, shard_args = producer(2)
            blocks = []
            offset = 0
            for width in [16, 8, 8]:
                blocks.append(source_conv[:, 1, :, offset + cp_rank * (width // 2):offset + (cp_rank + 1) * (width // 2)])
                offset += width
            shard.mamba_cache.conv[0][:, 1].copy_(torch.cat(blocks, dim=-1))
            shard.mamba_cache.temporal[:, 1].copy_(source_temporal[:, 1, cp_rank*2:(cp_rank+1)*2])
            manager.kv_args = shard_args
            manager.attn_cp_size = 2
            manager.attn_cp_rank = cp_rank
            manager.attn_tp_size = manager.state_tp_size = 1
            self.assertFalse(manager._should_skip_cp_replicated_state_transfer())
            for ptrs, sizes in zip(shard_args.state_data_ptrs, shard_args.state_data_lens):
                engine.batch_register(ptrs, sizes)
            self.assertEqual(manager.maybe_send_extra(req, [[1]], None, info), 0)
            for ptrs in shard_args.state_data_ptrs:
                engine.batch_deregister(ptrs)
        torch.testing.assert_close(destination.mamba_cache.conv[0][:, 2], source_conv[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(destination.mamba_cache.temporal[:, 2], source_temporal[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(destination.mamba_cache.conv[0][:, :2], torch.full_like(destination.mamba_cache.conv[0][:, :2], -17))
        for ptrs in dst.state_data_ptrs:
            engine.batch_deregister(ptrs)
        torch.testing.assert_close(source.mamba_cache.conv[0], source_conv)
        torch.testing.assert_close(source.mamba_cache.temporal, source_temporal)

    @staticmethod
    def http_ready(port):
        try:
            return requests.get(f"http://127.0.0.1:{port}/health", timeout=.2).status_code == 200
        except requests.RequestException:
            return False


if __name__ == "__main__":
    unittest.main()
