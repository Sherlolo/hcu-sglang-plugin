from __future__ import annotations
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.model_executor.runner import base_cuda_graph_runner as B
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as D
from sglang.srt.model_executor import runner as R
BASE=[B.get_batch_sizes_to_capture]
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class GraphBatchCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset();register();HookRegistry.apply_hooks()

    def compute(self, algorithm=None, cap=None, pool=64, width=1, alignment=1, overlap=False, gathered=None):
        from sglang.srt.dllm import config as C
        cfg=S(dllm_algorithm=algorithm,model_path='fixture',revision=None,max_running_requests=cap,dllm_algorithm_config=None,dllm_fdfo=False)
        hf=S(architectures=['DiffusionGemmaForBlockDiffusion' if algorithm=='Gemma4Renoise' else 'LLaDA2MoeModelLM'])
        model=S(server_args=cfg,req_to_token_pool=S(size=pool))
        execution=S(graph=S(cuda_graph_config=S(decode=S(bs=[1,2,4,8,16,32,64])),torch_compile_max_bs=8),overlap=S(enable_two_batch_overlap=overlap))
        with patch.object(C,'resolving_view',side_effect=lambda x:x),patch.object(C.ModelConfig,'from_server_args',return_value=S(hf_config=hf)),patch.object(B,'get_exec',return_value=execution),patch.object(B,'get_flags',return_value=S(capture=S(enable_torch_compile=True))),patch.object(B,'get_cuda_graph_batch_size_alignment',return_value=alignment) as align:
            old=BASE[0](model,width,gathered_buffer_required=gathered)
            result=D.get_batch_sizes_to_capture(model,width,gathered_buffer_required=gathered)
            self.assertEqual(align.call_args.kwargs,{'gathered_buffer_required':gathered})
        return old,result

    def test_preimported_public_aliases_resolve_to_hook(self):
        self.assertIs(B.get_batch_sizes_to_capture,D.get_batch_sizes_to_capture)
        self.assertIs(B.get_batch_sizes_to_capture,R.get_batch_sizes_to_capture)

    def test_real_gemma_config_limits_capture_and_compile(self):
        old,new=self.compute('Gemma4Renoise',cap=3)
        self.assertEqual(old,([1,2,4,8,16,32,64],[1,2,4,8]))
        self.assertEqual(new,([1,2,3],[1,2,3]))

    def test_default_one_request_and_pool_bound(self):
        self.assertEqual(self.compute('Gemma4Renoise')[1],([1],[1]))
        self.assertEqual(self.compute('Gemma4Renoise',cap=32,pool=3)[1],([1,2,3],[1,2,3]))

    def test_alignment_width_overlap_and_gathered_argument(self):
        self.assertEqual(self.compute('Gemma4Renoise',cap=5,alignment=4,gathered=True)[1],([4,8],[4,8]))
        self.assertEqual(self.compute('Gemma4Renoise',cap=5,alignment=4,width=4)[1],([1,2,4,8],[1,2,4,8]))
        self.assertEqual(self.compute('Gemma4Renoise',cap=5,alignment=4,width=4,overlap=True)[1],([4,8],[4,8]))

    def test_ordinary_and_non_separate_diffusion_keep_native_result(self):
        for algorithm in [None,'LowConfidence']:
            old,new=self.compute(algorithm,cap=3)
            self.assertEqual(old,new)

    def test_non_hcu_uses_effective_wrapped_original(self):
        from hcu_sglang import graph_batch as O
        seen=[];token=object()
        with patch.object(O,'is_hcu',return_value=False):
            result=O.around_batch_sizes(lambda *a,**k:seen.append((a,k)) or token,token,3,gathered_buffer_required=True)
        self.assertIs(result,token)
        self.assertEqual(seen,[((token,3),{'gathered_buffer_required':True})])


if __name__=='__main__':unittest.main()
