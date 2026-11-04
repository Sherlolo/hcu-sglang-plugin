from __future__ import annotations
from contextlib import nullcontext
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.model_executor.runner_utils import deepep_adapter as A
BASE_CLASS=[A.DeepEPCudaGraphRunnerAdapter]
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class DeepEPGraphCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def buffer(self):
        class Buffer:
            layout=None
            events=[]
            @classmethod
            def get_low_latency_layout(cls):return cls.layout
            @classmethod
            def set_low_latency_layout(cls,*layout):cls.layout=layout;cls.events.append(('layout',layout))
            @classmethod
            def set_dispatch_mode(cls,mode):cls.events.append(('mode',mode))
        return Buffer

    def test_native_class_alias_remains_class_with_compatible_owned_instances(self):
        from hcu_sglang import deepep_graph as O
        self.assertIs(A.DeepEPCudaGraphRunnerAdapter,O.DeepEPCudaGraphRunnerAdapter)
        self.assertTrue(issubclass(A.DeepEPCudaGraphRunnerAdapter,BASE_CLASS[0]))
        self.assertIsInstance(A.DeepEPCudaGraphRunnerAdapter(),BASE_CLASS[0])

    def test_replay_restores_captured_target_layout_before_dispatch_mode(self):
        from hcu_sglang import deepep_graph as O
        from sglang.srt.layers.moe.utils import MoeA2ABackend,DeepEPMode
        B=self.buffer()
        with patch.object(O,'DeepEPBuffer',B),patch.object(O,'get_moe_a2a_backend',return_value=MoeA2ABackend.DEEPEP),patch.object(O,'get_deepep_mode',return_value=DeepEPMode.AUTO):
            obj=A.DeepEPCudaGraphRunnerAdapter();obj.capture(False)
            B.layout=(128,7168,264);obj.capture_end()
            B.layout=(128,7168,256);B.events.clear();obj.replay()
        self.assertEqual(B.events,[('layout',(128,7168,264)),('mode',DeepEPMode.LOW_LATENCY)])

    def test_normal_or_non_deepep_do_not_restore_absent_layout(self):
        from hcu_sglang import deepep_graph as O
        from sglang.srt.layers.moe.utils import MoeA2ABackend,DeepEPMode
        B=self.buffer()
        with patch.object(O,'DeepEPBuffer',B),patch.object(O,'get_moe_a2a_backend',return_value=MoeA2ABackend.DEEPEP),patch.object(O,'get_deepep_mode',return_value=DeepEPMode.NORMAL):
            obj=A.DeepEPCudaGraphRunnerAdapter();obj.capture(True);obj.capture_end();B.events.clear();obj.replay()
            self.assertEqual(B.events,[('mode',DeepEPMode.NORMAL)])
        B.events.clear()
        with patch.object(O,'DeepEPBuffer',B),patch.object(O,'get_moe_a2a_backend',return_value=MoeA2ABackend.NONE):
            obj=A.DeepEPCudaGraphRunnerAdapter();obj.capture(False);obj.capture_end();obj.replay()
        self.assertEqual(B.events,[])

    def test_public_decode_capture_producer_records_layout_after_backend_capture(self):
        from hcu_sglang import deepep_graph as O
        from sglang.srt.model_executor.runner import decode_cuda_graph_runner as N
        from sglang.srt.layers.moe.utils import MoeA2ABackend,DeepEPMode
        B=self.buffer();events=[]
        fb=S(lora_ids=None,input_ids=torch.tensor([1]),positions=torch.tensor([0]),global_dp_buffer_len=1,dp_padding_mode=S(is_max_len=lambda:True),global_num_tokens_cpu=[1])
        attn=S(init_forward_metadata_out_graph=lambda *a,**k:events.append('out-metadata'),init_forward_metadata_in_graph=lambda *a:events.append('in-metadata'))
        def backend_capture(key,run_once,**kw):
            run_once();B.layout=(128,7168,264);events.append('backend-captured')
        runner=S(captured_req_width=1,ragged_verify_mode=False,capture_prepare=lambda *a,**k:(fb,attn,None),tbo_plugin=S(capture_one_batch_size=lambda *a,**k:None),_record_in_graph_metadata_prep_done=lambda:None,pp_size=1,dllm_uses_input_embeds=False,model_runner=S(spec_algorithm=S(is_dflash_family=lambda:False),is_draft_worker=False,capture_tail_hooks=[],canary_manager=None),_make_graph_key=lambda *a:'key',_capture_graph_size=lambda **kw:1,backend=S(capture_one=backend_capture))
        with patch.object(O,'DeepEPBuffer',B),patch.object(O,'get_moe_a2a_backend',return_value=MoeA2ABackend.DEEPEP),patch.object(O,'get_deepep_mode',return_value=DeepEPMode.AUTO),patch.object(O.N,'get_exec',return_value=S(graph=S(debug_cuda_graph=False))),patch.object(O.N,'forward_context',return_value=nullcontext()),patch.object(O.N,'ForwardContext',return_value=object()),patch.object(O.N,'set_dp_buffer_len'),patch.object(O.N,'set_is_extend_in_batch'),patch.object(O.N,'maybe_flashinfer_autotune_speculative_draft'):
            runner.deepep_adapter=A.DeepEPCudaGraphRunnerAdapter()
            N.DecodeCudaGraphRunner.capture_one_shape(runner,1,lambda *a,**k:events.append('forward'))
        self.assertEqual(runner.deepep_adapter._captured_low_latency_layout,(128,7168,264))
        self.assertEqual(events,['out-metadata','in-metadata','forward','backend-captured'])

    def test_non_hcu_capture_keeps_effective_original(self):
        from hcu_sglang import deepep_graph as O
        token=object();seen=[]
        with patch.object(O,'is_hcu',return_value=False):
            self.assertIs(O.around_capture(lambda *a,**k:seen.append((a,k)) or token,token,4,variant_label='x'),token)
        self.assertEqual(seen,[((token,4),{'variant_label':'x'})])


if __name__=='__main__':unittest.main()
