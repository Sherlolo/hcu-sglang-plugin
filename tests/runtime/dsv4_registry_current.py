from types import SimpleNamespace
from unittest.mock import patch
import unittest
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.attention import dsv4_registry as R
from sglang.srt.plugins.hook_registry import HookRegistry
import sglang.srt.layers.attention.attention_registry as A
import sglang.srt.layers.attention.deepseek_v4_trtllm_backend as F
from sglang.srt.speculative.draft_utils import DraftBackendFactory

class Dsv4RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();R.register();HookRegistry.apply_hooks()

    def test_real_registry_dictionary_selects_base_factory_for_hcu(self):
        runner=object();marker=object()
        with patch.object(R,'is_dcu',return_value=True),patch.object(F,'is_dsv4_trtllm_attn_enabled',return_value=False),patch.object(F,'DeepseekV4AttnBackend',return_value=marker) as ctor:
            self.assertIs(A.ATTENTION_BACKENDS['dsv4'](runner),marker)
        ctor.assert_called_once_with(runner)

    def factory(self):
        f=object.__new__(DraftBackendFactory);f.draft_model_runner=object();f.topk=1;f.speculative_num_steps=3;return f

    def test_real_draft_decode_uses_base_multistep_and_preserves_sizes(self):
        f=self.factory();marker=object()
        with patch.object(R,'is_dcu',return_value=True),patch.object(F,'is_dsv4_trtllm_attn_enabled',return_value=False),patch.object(F,'DeepseekV4MultiStepBackend',return_value=marker) as ctor:
            name,backend=f._create_dsv4_decode_backend()
        self.assertEqual(name,'dsv4');self.assertIs(backend,marker);ctor.assert_called_once_with(f.draft_model_runner,topk=1,speculative_num_steps=3)

    def test_real_draft_prefill_preserves_skip_prefill_false(self):
        f=self.factory();marker=object()
        with patch.object(R,'is_dcu',return_value=True),patch.object(F,'is_dsv4_trtllm_attn_enabled',return_value=False),patch.object(F,'DeepseekV4AttnBackend',return_value=marker) as ctor:
            name,backend=f._create_dsv4_prefill_backend()
        self.assertEqual(name,'dsv4');self.assertIs(backend,marker);ctor.assert_called_once_with(f.draft_model_runner,skip_prefill=False)

    def test_non_hcu_delegates_all_original_factories(self):
        marker=object();arg=object()
        with patch.object(R,'is_dcu',return_value=False):
            for fn in [R.around_create_dsv4_backend,R.around_create_dsv4_decode_backend,R.around_create_dsv4_prefill_backend]:self.assertIs(fn(lambda owner:marker,arg),marker)

    def test_owned_hcu_metadata_matches_real_flashmla_consumer(self):
        from hcu_sglang.models.deepseek.current_v4.hcu_backend import backend as B
        from flash_mla.flash_mla_interface import FlashMLASchedMeta
        a=B._create_flashmla_metadata();b=B._create_flashmla_metadata();self.assertIsInstance(a,FlashMLASchedMeta);self.assertIsInstance(b,FlashMLASchedMeta);self.assertIsNot(a,b)

if __name__=='__main__':unittest.main()
