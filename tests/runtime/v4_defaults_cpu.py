from __future__ import annotations
from contextlib import ExitStack,contextmanager
from types import SimpleNamespace as S
from unittest.mock import patch
import os,unittest
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from sglang.srt.arg_groups import deepseek_v4_hook as N
BASE=[N.apply_deepseek_v4_defaults,N.validate_deepseek_v4_cp]
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class V4DefaultsCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def config(self,**kw):
        d=dict(device='cuda',dsv4_attn_backend='auto',kv_cache_dtype='fp8_e4m3',max_running_requests=32,speculative_algorithm=None,speculative_eagle_topk=1,enable_prefill_cp=False,cp_strategy='interleave',attn_dp_size=1,attn_cp_size=1,tp_size=8,dp_size=1,moe_dense_tp_size=1,ep_size=8,moe_a2a_backend='deepep',nnodes=1,enable_decoder_swa_bounded_replay=False,enable_two_batch_overlap=False)
        d.update(kw);return S(**d)

    @contextmanager
    def context(self,cfg,hip=True,hcu=True,sm100=False,model='deepseek_v41'):
        from hcu_sglang import v4_defaults as O
        from sglang.srt.utils import common
        declarations=[]
        def declare(seed,owner,**kw):
            declarations.append(kw)
            for key,value in kw.items():setattr(cfg,key,value)
        with ExitStack() as stack:
            for obj,key,value in [(O,'_is_hcu',hcu),(N,'resolving_view',lambda x:cfg),(N,'resolved_view',lambda x:cfg),(N,'model_config_of',lambda x:S(hf_config=S(model_type=model))),(N,'get_platform',lambda:S(is_hip=hip)),(N,'run_post_process_pass',lambda *a:None),(N,'declare_resolution',declare),(N,'context_parallel_attn_dp_size',lambda *a:1),(N,'is_npu',lambda:False),(N,'attention_backends_of',lambda x:('hcu_mla','hcu_mla')),(common,'is_sm100_supported',lambda:sm100)]:stack.enter_context(patch.object(obj,key,value))
            yield declarations

    def test_old_defaults_clear_explicit_hcu_sparse_new_public_handler_preserves(self):
        cfg=self.config(dsv4_attn_backend='flashmla')
        with self.context(cfg),patch.dict(os.environ,{'SGLANG_OPT_FLASHMLA_SPARSE_PREFILL':'1'}):
            BASE[0](object(),'DeepseekV4ForCausalLM');self.assertFalse(N.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get());N.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(True)
            N.apply_deepseek_v4_defaults(object(),'DeepseekV4ForCausalLM');self.assertTrue(N.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get())

    def test_unspecified_and_explicit_false_do_not_enable_sparse_implicitly(self):
        for value in [None,'0']:
            with patch.dict(os.environ,{},clear=False):
                os.environ.pop('SGLANG_OPT_FLASHMLA_SPARSE_PREFILL',None)
                if value is not None:os.environ['SGLANG_OPT_FLASHMLA_SPARSE_PREFILL']=value
                with self.context(self.config()):N.apply_deepseek_v4_defaults(object(),'DeepseekV4ForCausalLM')
                self.assertFalse(N.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get())

    def test_source_backend_default_and_request_limit_are_declared(self):
        cfg=self.config(max_running_requests=None)
        with self.context(cfg),patch.dict(os.environ,{'SGLANG_OPT_FLASHMLA_SPARSE_PREFILL':'1'}):
            N.apply_deepseek_v4_defaults(object(),'DeepseekV4ForCausalLM')
        self.assertEqual(cfg.dsv4_attn_backend,'flashmla');self.assertEqual(cfg.max_running_requests,256)

    def test_sm100_mega_selector_respects_model_dtype_and_explicit_layout(self):
        from hcu_sglang import v4_defaults as O
        cfg=self.config()
        with self.context(cfg,hip=False,sm100=True),patch.dict(os.environ,{'SGLANG_DSV4_KV_LAYOUT':'auto'}):self.assertEqual(O._default_dsv4_attention_backend(object()),'flashmla_mega')
        with self.context(cfg,hip=False,sm100=True),patch.dict(os.environ,{'SGLANG_DSV4_KV_LAYOUT':'legacy'}):self.assertEqual(O._default_dsv4_attention_backend(object()),'flashmla')
        cfg.kv_cache_dtype='bf16'
        with self.context(cfg,hip=False,sm100=True):self.assertEqual(O._default_dsv4_attention_backend(object()),'flashmla')
        with self.context(self.config(),hip=False,sm100=True,model='deepseek_v4'):self.assertEqual(O._default_dsv4_attention_backend(object()),'flashmla')

    def test_hcu_cp_preserves_explicit_sparse_and_declares_local_topology(self):
        cfg=self.config(enable_prefill_cp=True)
        with self.context(cfg,model='deepseek_v4'),patch.dict(os.environ,{'SGLANG_OPT_FLASHMLA_SPARSE_PREFILL':'1'}):
            N.validate_deepseek_v4_cp(object());self.assertTrue(N.envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get())
        self.assertEqual((cfg.attn_dp_size,cfg.dp_size,cfg.attn_cp_size,cfg.moe_dense_tp_size),(1,1,8,1))

    def test_non_hcu_hip_keeps_backend_guard_and_invalid_cp_is_rejected(self):
        with self.context(self.config(enable_prefill_cp=True),hcu=False):
            with self.assertRaisesRegex(ValueError,'requires the dsv4 attention'):N.validate_deepseek_v4_cp(object())
        for kw in [{'cp_strategy':'zigzag'},{'tp_size':16},{'moe_a2a_backend':'invalid'}]:
            with self.context(self.config(enable_prefill_cp=True,**kw)):
                with self.assertRaises((AssertionError,ValueError)):N.validate_deepseek_v4_cp(object())

    def test_invalid_speculation_and_mega_hardware_are_not_silently_changed(self):
        with self.context(self.config(speculative_algorithm='EAGLE',speculative_eagle_topk=2)):
            with self.assertRaises(AssertionError):N.apply_deepseek_v4_defaults(object(),'DeepseekV4ForCausalLM')
        with self.context(self.config(dsv4_attn_backend='flashmla_mega'),sm100=False):
            with self.assertRaisesRegex(AssertionError,'requires DeepSeek-V4.1'):N.apply_deepseek_v4_defaults(object(),'DeepseekV4ForCausalLM')

    def test_legacy_cp_export_and_registered_target_share_canonical_owner(self):
        from hcu_sglang import v4_defaults as O
        from hcu_sglang.models.deepseek.current_v4.hcu_backend import arg_groups as A
        self.assertIs(A.validate_deepseek_v4_cp,O.validate_deepseek_v4_cp)
        self.assertIs(HookRegistry._hooks['sglang.srt.arg_groups.deepseek_v4_hook.validate_deepseek_v4_cp'][0][1],O.validate_deepseek_v4_cp)
        self.assertEqual(len(HookRegistry._hooks['sglang.srt.arg_groups.deepseek_v4_hook.validate_deepseek_v4_cp']),1)

    def test_non_hcu_effective_original(self):
        from hcu_sglang import v4_defaults as O
        for fn in [O.around_defaults]:
            token=object();seen=[]
            with patch.object(O,'is_hcu',return_value=False):self.assertIs(fn(lambda *a,**k:seen.append((a,k)) or token,token,flag=1),token)
            self.assertEqual(seen,[((token,),{'flag':1})])


if __name__=='__main__':unittest.main()
