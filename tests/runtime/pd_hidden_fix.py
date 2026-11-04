from __future__ import annotations

import gc
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import weakref

import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.disagg import hidden_state as H, integration
from hcu_sglang.disagg import prefill as P
from hcu_sglang.disagg import utils as U
from sglang.srt.disaggregation import prefill as N
from sglang.srt.disaggregation import utils as NU
from sglang.srt.plugins.hook_registry import HookRegistry


class Request:
    pass


class PDHiddenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset()
        integration.register_hooks()
        HookRegistry.apply_hooks()

    def request(self):
        req=Request()
        req.rid='owned-pd-hidden'
        req.origin_input_ids=[1,2,3,4]
        req.pending_bootstrap=False
        req.metadata_buffer_index=-1
        req.disagg_kv_sender=None
        return req

    def queue(self):
        buffers=NU.MetadataBuffers(2,4,torch.bfloat16,1,pd_hidden_pool_size=4,pd_hidden_size=4)
        pool=buffers.pd_hidden_pool
        queue=N.PrefillBootstrapQueue.__new__(N.PrefillBootstrapQueue)
        queue.pp_rank=0
        queue.metadata_buffers=buffers
        queue.scheduler=S(model_config=S(hidden_size=4),
            tp_worker=S(model_runner=S(spec_aux_config=S(dflash_target_layer_ids=[1]))),
            server_args=S(disable_radix_cache=True))
        return queue,pool

    def metadata(self,stream=False):
        return {'hidden_start':2,'hidden_len':2,'target_layer_ids':[1],
                'dst_indices':[8,9],'decode_radix_cache_enabled':False,'streaming_hidden':stream}

    def test_native_bootstrap_capture_write_payload_and_release_share_one_state(self):
        req=self.request();queue,pool=self.queue()
        self.assertTrue(queue._finalize_pd_hidden_bootstrap(req,self.metadata(),2))
        state=H.get_pd_hidden_req_state(req)
        self.assertEqual(H.get_pd_hidden_capture_layer_ids([req]),[1])
        self.assertIs(P.pd_hidden_state(req),state)
        self.assertEqual(pool.available_size(),2)
        scheduler=N.SchedulerDisaggregationPrefillMixin()
        scheduler.disagg_metadata_buffers=queue.metadata_buffers
        batch=S(reqs=[req],prefix_lens=[2],extend_lens=[2],seq_lens_cpu=torch.tensor([2]))
        values=torch.arange(8,dtype=torch.bfloat16).view(2,4)
        result=S(logits_output=S(hidden_states=values),pp_hidden_states_proxy_tensors=None)
        with patch.object(P,'get_parallel',return_value=S(pp_rank=0)):
            scheduler._write_pd_hidden_rows_for_batch(batch,result)
        self.assertTrue(torch.equal(pool.buffer[state.src_indices],values))
        payload=N._select_pd_hidden_payload_indices(rid=req.rid,owner_direct_sent=False,
            src_indices=state.src_indices,capture_layer_ids=state.capture_layer_ids,
            current_src_indices=state.current_src_indices,has_current_pd_hidden=True,
            streaming_hidden=False,written=state.written)
        self.assertEqual(payload.tolist(),state.src_indices)
        self.assertEqual(state.current_start,2)
        self.assertTrue(state.current_is_last)
        N.maybe_release_metadata_buffer(req,None,pool)
        self.assertEqual(pool.available_size(),4)
        self.assertIsNone(H.get_pd_hidden_capture_layer_ids([req]))
        N.maybe_release_metadata_buffer(req,None,pool)
        self.assertEqual(pool.available_size(),4)

    def test_public_state_assembly_registers_real_hidden_buffer_bytes(self):
        from sglang.srt.disaggregation.base.conn import StateType
        pool=NU.PDHiddenRowPool(4,4,torch.bfloat16)
        args=S()
        NU.setup_state_kv_args(args,object(),pd_hidden_pool=pool)
        index=args.state_types.index(StateType.PD_HIDDEN)
        self.assertEqual(args.state_data_ptrs[index],[pool.buffer.data_ptr()])
        self.assertEqual(args.state_data_lens[index],[pool.buffer.nbytes])
        self.assertEqual(args.state_item_lens[index],[8])

    def test_incomplete_payload_rejected_and_weak_state_released(self):
        req=self.request();state=H.get_pd_hidden_req_state(req)
        state.capture_layer_ids=[1]
        with self.assertRaisesRegex(RuntimeError,'incomplete'):
            N._select_pd_hidden_payload_indices(rid=req.rid,owner_direct_sent=False,
                src_indices=[0,1],capture_layer_ids=[1],current_src_indices=[0],
                has_current_pd_hidden=True,streaming_hidden=False,written=[True,False])
        ref=weakref.ref(req)
        before=len(H._pd_hidden_req_states)
        del req
        gc.collect()
        self.assertIsNone(ref())
        self.assertLess(len(H._pd_hidden_req_states),before)

    def test_owned_frontend_uses_final_descriptor_wrapper(self):
        from unittest.mock import Mock
        effective=Mock(return_value='descriptor')
        args=object();pool=object();hidden=object()
        with patch.object(NU,'setup_state_kv_args',effective):
            self.assertEqual(P.setup_state_kv_args(args,pool,pd_hidden_pool=hidden),'descriptor')
        effective.assert_called_once_with(args,pool,pd_hidden_pool=hidden)

    def test_decode_full_injection_observes_copied_rows_before_done(self):
        from sglang.srt.disaggregation import decode as D
        from hcu_sglang.disagg.common_utils import PDHiddenRequestState
        req=self.request()
        pool=NU.PDHiddenRowPool(4,4,torch.bfloat16)
        rows=pool.alloc(2)
        values=torch.arange(8,dtype=torch.bfloat16).reshape(2,4)
        pool.write(rows,values)
        decode_req=D.DecodeRequest(req,None,pd_hidden_dst_indices=rows,pd_hidden_state=PDHiddenRequestState.full(2,4))
        queue=D.DecodeTransferQueue.__new__(D.DecodeTransferQueue)
        queue.metadata_buffers=S(pd_hidden_pool=pool)
        observed=[]
        class Event:
            def synchronize(self):observed.append('done')
        def inject(request,hidden,start):
            self.assertIs(request,req)
            self.assertEqual(start,2)
            self.assertTrue(torch.equal(hidden,values))
            observed.append('read')
            return Event()
        queue.scheduler=S(draft_worker=S(inject_pd_hidden_chunk=inject))
        queue._inject_full_pd_hidden(decode_req)
        self.assertEqual(observed,['read','done'])
        self.assertTrue(decode_req.pd_hidden_state.hidden_done)

    def test_decode_streaming_defers_done_until_matching_ack(self):
        from sglang.srt.disaggregation import decode as D
        from hcu_sglang.disagg.common_utils import PDHiddenRequestState
        req=self.request();req.bootstrap_room=17
        pool=NU.PDHiddenRowPool(4,4,torch.bfloat16)
        rows=pool.alloc(2);values=torch.arange(8,dtype=torch.bfloat16).reshape(2,4);pool.write(rows,values)
        decode_req=D.DecodeRequest(req,None,pd_hidden_dst_indices=rows,pd_hidden_state=PDHiddenRequestState.streaming_state(2,4))
        queue=D.DecodeTransferQueue.__new__(D.DecodeTransferQueue)
        queue.metadata_buffers=S(pd_hidden_pool=pool)
        from unittest.mock import Mock
        event=object()
        inject=Mock(return_value=event)
        queue.scheduler=S(draft_worker=S(inject_pd_hidden_chunk=inject))
        packet={'room':17,'prefill_rank':0,'hidden_start':2,'row_len':2,'is_last_hidden_chunk':True,'dst_indices':rows,'ack_host':'localhost','ack_port':1234}
        queue.kv_manager=S(pop_pd_hidden_ready_chunks=Mock(return_value=[packet]),submit_pd_hidden_chunk_ack=Mock(),pop_pd_hidden_acked_chunks=Mock(return_value=[packet]))
        queue._drain_pd_hidden_ready_chunks(decode_req)
        self.assertFalse(decode_req.pd_hidden_state.hidden_done)
        self.assertTrue(torch.equal(inject.call_args.args[1],values))
        self.assertIs(queue.kv_manager.submit_pd_hidden_chunk_ack.call_args.kwargs['event'],event)
        queue._consume_pd_hidden_acked_chunks(decode_req)
        self.assertTrue(decode_req.pd_hidden_state.hidden_done)

    def test_actual_decode_request_pool_constructor_uses_current_allocation_lifecycle(self):
        from sglang.srt.disaggregation import decode as D
        pool=D.DecodeReqToTokenPool(2,8,'cpu',False,2)
        self.assertEqual(pool.req_to_token.shape,(5,8))
        self.assertEqual(pool.free_slots,[1,2,3,4])



if __name__=='__main__':unittest.main()
