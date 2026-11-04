from __future__ import annotations
from types import SimpleNamespace as S
from unittest.mock import patch
import unittest
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import replay_validation as O,model_replay_budget as C
from sglang.srt.model_executor.cuda_graph_config import Backend


class ReplayValidationCpuTests(unittest.TestCase):
    def config(self,**kw):
        data=dict(moe_a2a_backend='none',enable_encoder_swa_bounded_replay=False,dsv4_attn_backend='flashmla',cuda_graph_config=S(prefill=S(backend=Backend.BREAKABLE,max_seq_len=None)),attn_cp_size=1,enable_unified_cache_external_linker=False,enable_unified_memory=False,disaggregation_mode='null',enable_mixed_chunk=False,enable_lora=False,enable_session_radix_cache=False,speculative_algorithm=None,enable_hisparse=False,enable_two_batch_overlap=False,pp_size=1,enable_decoder_swa_bounded_replay=False,enable_prefill_cp=False,cp_strategy=None,tp_size=8,attn_dp_size=8,dp_size=1,dcp_size=1,enable_return_hidden_states=False,page_size=16,max_running_requests=4,chunked_prefill_size=144)
        data.update(kw);return S(**data)

    def features(self,cfg,hip=False,cuda=True,dtype='deepseek_v41'):
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import env_gate
        declarations=[]
        with patch.object(O.N,'resolving_view',return_value=cfg),patch.object(O.N,'model_config_of',return_value=S(hf_config=S(model_type=dtype))),patch.object(O.N,'get_platform',return_value=S(is_cuda=cuda,is_hip=hip)),patch.object(O.N,'is_gfx95_supported',return_value=False),patch.object(O.N,'attn_dp_enabled_of',return_value=True),patch.object(O.N,'declare_resolution',side_effect=lambda *a,**kw:declarations.append(kw)),patch.object(env_gate,'is_unified_kv_triton',return_value=False):O.validate_deepseek_v41_features(object())
        return declarations

    def test_replay_budget_requires_window_plus_one_page_and_running_slots(self):
        cfg=self.config(enable_encoder_swa_bounded_replay=True)
        with patch.object(O.N,'resolving_view',return_value=cfg):
            O.validate_deepseek_v41_encoder_replay_budget(object());cfg.chunked_prefill_size=143
            with self.assertRaisesRegex(ValueError,'at least 144'):O.validate_deepseek_v41_encoder_replay_budget(object())
            cfg.chunked_prefill_size=144;cfg.max_running_requests=0
            with self.assertRaises(ValueError):O.validate_deepseek_v41_encoder_replay_budget(object())

    def test_source_cuda_bcg_encoder_allowed_and_hip_hardware_guard_retained(self):
        cfg=self.config(enable_encoder_swa_bounded_replay=True)
        self.features(cfg)
        with self.assertRaisesRegex(ValueError,'hardware other than CUDA or gfx950'):self.features(cfg,hip=True,cuda=False)
        with self.assertRaisesRegex(ValueError,'requires DeepSeek-V4.1'):self.features(cfg,dtype='deepseek_v4')
        cfg.cuda_graph_config.prefill.backend=Backend.FULL
        with self.assertRaisesRegex(ValueError,'other than breakable'):self.features(cfg)

    def test_decoder_auto_declares_source_default_without_rewriting_prefill_limit(self):
        cfg=self.config(enable_decoder_swa_bounded_replay=None,moe_a2a_backend='none')
        declarations=self.features(cfg);self.assertEqual(declarations,[{'enable_decoder_swa_bounded_replay':True}]);self.assertIsNone(cfg.cuda_graph_config.prefill.max_seq_len)
        cfg.enable_decoder_swa_bounded_replay=False;self.assertEqual(self.features(cfg),[])

    def test_named_capability_seam_preserves_previous_and_instance_early_return(self):
        cfg=self.config(model_path='fixture',enable_encoder_swa_bounded_replay=True);calls=[];token=object()
        with patch.object(C,'is_hcu',return_value=True),patch.object(C.N,'resolving_view',return_value=cfg),patch.object(C.N,'parse_connector_type',return_value=object()),patch.object(C.N,'model_config_of',return_value=S(hf_config=S(model_type='deepseek_v41'))),patch.object(O.N,'resolving_view',return_value=cfg):
            self.assertIs(C.capability_replay_budget(object(),lambda seed:calls.append('previous') or token),token)
            cfg.chunked_prefill_size=128
            with self.assertRaises(ValueError):C.capability_replay_budget(object(),lambda seed:calls.append('previous'))
        self.assertEqual(calls,['previous','previous'])
        with patch.object(C,'is_hcu',return_value=True),patch.object(C.N,'resolving_view',return_value=cfg),patch.object(C.N,'parse_connector_type',return_value=C.N.ConnectorType.INSTANCE),patch.object(C.N,'model_config_of',side_effect=AssertionError('must not read model config')):self.assertIs(C.capability_replay_budget(object(),lambda seed:token),token)

    def test_real_named_resolution_dispatcher_reaches_budget_after_previous(self):
        from sglang.srt.arg_groups import resolution_hooks as H
        cfg=self.config(model_path='fixture',enable_encoder_swa_bounded_replay=True,chunked_prefill_size=128);calls=[]
        def handle_model_capability_adjustments(seed):calls.append('builtin')
        with patch.dict(H._HOOKS,{},clear=True),patch.object(C,'_REGISTERED',False),patch.object(C,'is_hcu',return_value=True),patch.object(C.N,'resolving_view',return_value=cfg),patch.object(C.N,'parse_connector_type',return_value=object()),patch.object(C.N,'model_config_of',return_value=S(hf_config=S(model_type='deepseek_v41'))),patch.object(O.N,'resolving_view',return_value=cfg):
            C.register_hooks();C.register_hooks();self.assertEqual(len(H._HOOKS['handle_model_capability_adjustments']),1)
            with self.assertRaisesRegex(ValueError,'at least 144'):H.run_hook(handle_model_capability_adjustments,object())
        self.assertEqual(calls,['builtin'])


if __name__=='__main__':unittest.main()
