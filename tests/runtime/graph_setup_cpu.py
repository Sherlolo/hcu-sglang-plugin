from __future__ import annotations
from contextlib import ExitStack
from types import SimpleNamespace as S
import os
import unittest
from unittest.mock import patch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.model_executor.model_runner_components import cuda_graph_setup as N
from sglang.srt.model_executor import model_runner as M
BASE_INDEX=[N.index_attention_layers_by_global_id]
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class GraphSetupCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        HookRegistry.reset();register();HookRegistry.apply_hooks()

    def test_preimported_public_model_runner_aliases(self):
        self.assertIs(M.capture_cuda_graphs,N.capture_cuda_graphs)
        self.assertIs(M.capture_prefill_graph,N.capture_prefill_graph)

    def test_trailing_pure_moe_layers_preserve_global_index_space(self):
        a=S(layer_id=0);b=S(layer_id=1);model=S(layers=[a,b,object(),object()])
        old=BASE_INDEX[0]([a,b],[None,None],model)
        new=N.index_attention_layers_by_global_id([a,b],[None,None],model)
        self.assertEqual(len(old[0]),2)
        self.assertEqual(new,([a,b,None,None],[None]*4))
        a0=[a,a];c0=[None,None]
        self.assertIs(N.index_attention_layers_by_global_id(a0,c0,model)[0],a0)
        with self.assertRaisesRegex(ValueError,'parallel'):N.index_attention_layers_by_global_id([a],[],model)

    def test_hybrid_handler_requires_full_local_range_coverage(self):
        from hcu_sglang import graph_setup as O
        a=[None,object(),None,None];moe=[None,None,object(),object()]
        self.assertTrue(O.has_hybrid_graph_handlers(attention_layers=a,moe_layers=moe,start_layer=1,end_layer=4))
        for start,end in [(0,4),(-1,4),(1,5),(2,2)]:
            self.assertFalse(O.has_hybrid_graph_handlers(attention_layers=a,moe_layers=moe,start_layer=start,end_layer=end))

    def budget(self,pp=2,tp=2,only=False,initialized=False,join=None,device='cuda',backend='deep_gemm',draft=False):
        from sglang.srt.layers.moe.moe_runner import deep_gemm as G
        parallel=S(pp_size=pp,tp_group=S(world_size=tp,cpu_group='tp-cpu'),world_group=S(cpu_group='world-cpu'),launch_world_size=pp*tp)
        runner=S(device=device,gpu_id=0,is_draft_worker=draft,model_config=S(quantization='fp8'))
        with patch.dict(os.environ,{'SGLANG_DEEPGEMM_STANDARD_LAYOUT':'auto','SGLANG_DEEPGEMM_MASKED_MEMORY_BUDGET_FRACTION':'0.25'}),patch.object(N,'get_parallel',return_value=parallel),patch.object(N,'get_exec',return_value=S(moe=S(ep_join_mode=join,moe_runner_backend=backend,moe_a2a_backend='none'))),patch.object(N,'get_spec',return_value=S(speculative_moe_runner_backend='deep_gemm',speculative_moe_a2a_backend=None)),patch.object(N,'_deep_gemm_layout_memory_budget_initialized',initialized),patch.object(N,'get_available_gpu_memory',return_value=12.0) as mem,patch.object(G,'_masked_standard_layout_memory_budget_bytes',None):
            N.refresh_deep_gemm_layout_memory_budget(runner,only_if_initialized=only)
            return mem.call_args,N._deep_gemm_layout_memory_budget_initialized,G._masked_standard_layout_memory_budget_bytes

    def test_pp_budget_uses_tp_group_and_native_mutable_budget(self):
        call,flag,budget=self.budget()
        self.assertEqual(call.kwargs,{'distributed':True,'cpu_group':'tp-cpu'})
        self.assertTrue(flag);self.assertEqual(budget,3*(1<<30))
        self.assertEqual(self.budget(tp=1)[0].kwargs,{'distributed':False,'cpu_group':'tp-cpu'})
        self.assertEqual(self.budget(pp=1)[0].kwargs,{'distributed':True,'cpu_group':'world-cpu'})

    def test_budget_guards_and_draft_backend_override(self):
        for kwargs in [{'only':True},{'only':True,'initialized':True,'join':'scale'},{'device':'cpu'},{'backend':'triton'}]:
            self.assertIsNone(self.budget(**kwargs)[0])
        self.assertIsNotNone(self.budget(only=True,initialized=True)[0])
        self.assertIsNotNone(self.budget(backend='triton',draft=True)[0])

    def test_h16_cache_release_requires_both_feature_and_backend_and_sync_order(self):
        from hcu_sglang import graph_setup as O
        events=[];platform=S(synchronize=lambda:events.append('sync'),empty_cache=lambda:events.append('empty'))
        with patch.object(N,'current_platform',platform),patch.object(O,'is_hcu',return_value=True),patch.dict(os.environ,{'SGLANG_HCU_FLASHMLA_DECODE_H16':'1'}):
            self.assertFalse(O.release_hcu_flashmla_decode_h16_capture_cache(S()))
            self.assertTrue(O.release_hcu_flashmla_decode_h16_capture_cache(S(attn_backend=S(_hcu_flashmla_decode_h16_enabled=True))))
        self.assertEqual(events,['sync','empty'])
        with patch.object(O,'is_hcu',return_value=False):self.assertFalse(O.release_hcu_flashmla_decode_h16_capture_cache(S(attn_backend=S(_hcu_flashmla_decode_h16_enabled=True))))

    def test_public_capture_lifecycle_releases_cache_after_finalization(self):
        from hcu_sglang import graph_setup as O
        for decode in [True,False]:
            events=[];eager=object();runner=S(is_draft_worker=False,attn_backend=S(_hcu_flashmla_decode_h16_enabled=True))
            platform=S(capabilities=S(graph_capture=True),synchronize=lambda:events.append('sync'),empty_cache=lambda:events.append('empty'))
            with ExitStack() as stack:
                for obj,key,value in [(N,'current_platform',platform),(N,'EagerRunner',lambda *a,**k:events.append('eager') or eager),(N,'capture_decode_graph',lambda **k:events.append('decode') or N.GraphCapture(runner=None,memory_phase='decode',memory_usage_gb=0,capture_time=0)),(N,'finalize_cuda_graph_capture',lambda *a:events.append('finalize')),(O,'refresh_deep_gemm_layout_memory_budget',lambda *a:events.append('budget')),(O,'capture_prefill_graph',lambda **k:events.append('prefill') or N.GraphCapture(runner=None,memory_phase='prefill',memory_usage_gb=0,capture_time=0))]:stack.enter_context(patch.object(obj,key,value))
                stack.enter_context(patch.object(N.GraphSharedOutput,'create_for_model_runner',return_value=object()))
                stack.enter_context(patch.dict(os.environ,{'SGLANG_HCU_FLASHMLA_DECODE_H16':'1'}))
                out=N.capture_cuda_graphs(model_runner=runner,capture_decode_cuda_graph=decode)
            self.assertEqual(events,['eager','budget','prefill']+(['decode'] if decode else [])+['finalize']+(['sync','empty'] if decode else []))
            self.assertIs(out.eager_runner,eager)

    def test_public_hybrid_prefill_retains_trailing_moe_and_rejects_holes(self):
        a=S(layer_id=0);layers=[object() for _ in range(3)];lm=S(layers=layers,supports_hybrid_prefill_graph=True)
        for hole in [False,True]:
            runner=S(is_draft_worker=False,spec_algorithm=S(is_eagle=lambda:False),lora_manager=None,model=S(model=lm),model_config=S(context_len=64),req_to_token_pool=S(size=4),layer_info=S(start_layer=0,end_layer=3),device='cuda',gpu_id=0)
            runner.get_cuda_graph_layers=lambda model:([a],[None,object(),None if hole else object()],[],[],[None])
            cfg=S(bs=[4,16],backend=N.Backend.BREAKABLE)
            token=object()
            with patch.object(N,'check_cuda_graph_backend',return_value=False),patch.object(N,'get_exec',return_value=S(graph=S(cuda_graph_config=S(prefill=cfg)))),patch.object(N,'get_parallel',return_value=S(enable_prefill_cp=False,pp_size=1)),patch.object(N,'get_available_gpu_memory',return_value=10),patch.object(N,'PrefillCudaGraphRunner',return_value=token) as create:
                result=M.capture_prefill_graph(model_runner=runner,eager_runner=object())
            self.assertEqual(runner.attention_layers,[a,None,None])
            self.assertIs(result.runner,None if hole else token)
            self.assertEqual(create.call_count,0 if hole else 1)

    def test_non_hcu_preserves_effective_original_all_wrappers(self):
        from hcu_sglang import graph_setup as O
        for name in ['index_attention_layers_by_global_id','refresh_deep_gemm_layout_memory_budget','capture_cuda_graphs','capture_prefill_graph']:
            seen=[];token=object()
            with patch.object(O,'is_hcu',return_value=False):self.assertIs(getattr(O,'around_'+name)(lambda *a,**k:seen.append((a,k)) or token,token,option=7),token)
            self.assertEqual(seen,[((token,),{'option':7})])


if __name__=='__main__':unittest.main()
