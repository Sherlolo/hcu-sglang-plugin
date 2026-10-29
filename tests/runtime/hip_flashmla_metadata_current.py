import unittest
from unittest.mock import patch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.layers.attention.debug_flash_mla_adapter import around_hip_create_flashmla_metadata

class HipFlashMLAMetadataTests(unittest.TestCase):
    def test_real_hcu_factory_returns_fresh_consumer_metadata(self):
        from flash_mla.flash_mla_interface import FlashMLASchedMeta
        from hcu_sglang.utils import detection
        with patch.object(detection,'is_dcu',return_value=True):
            a=around_hip_create_flashmla_metadata(lambda:None)
            b=around_hip_create_flashmla_metadata(lambda:None)
        self.assertIsInstance(a,FlashMLASchedMeta);self.assertIsInstance(b,FlashMLASchedMeta)
        self.assertIsNot(a,b)

    def test_non_hcu_preserves_native_factory(self):
        from hcu_sglang.utils import detection
        marker=object()
        with patch.object(detection,'is_dcu',return_value=False):
            self.assertIs(around_hip_create_flashmla_metadata(lambda:marker),marker)

if __name__=='__main__':unittest.main()
