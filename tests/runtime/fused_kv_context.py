from types import SimpleNamespace
from unittest.mock import patch
import unittest

import torch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from hcu_sglang.models import fused_rotary


class Pool:
    def __init__(self):
        self.k = torch.zeros(2, 4)
        self.v = torch.zeros(2, 4)
        self.layers = []

    def get_kv_buffer(self, layer_id):
        self.layers.append(layer_id)
        return self.k, self.v


class FusedKVContextTests(unittest.TestCase):
    def test_fused_store_uses_active_backend_pool_and_follows_nested_contexts(self):
        outer, inner = Pool(), Pool()
        module = SimpleNamespace(rotary_emb=SimpleNamespace(cos_sin_cache=torch.zeros(2, 4)),
                                 layer_id=7, head_dim=4, kv_cache_dtype=torch.float32)
        norm = SimpleNamespace(weight=torch.ones(4), variance_epsilon=1e-6)
        q, k, v = (torch.ones(1, 4) * n for n in (1, 2, 3))
        batch = SimpleNamespace(out_cache_loc=torch.tensor([0]))
        received = []
        def kernel(positions, q, k, v, cache, head_dim, page_size, k_buffer, v_buffer, locations, **kwargs):
            received.append((k_buffer.data_ptr(), v_buffer.data_ptr()))
            k_buffer[locations] = k
            v_buffer[locations] = v
            return q, k, v
        def run():
            return fused_rotary.fused_rms_rotary_kv_store(kernel,module,torch.tensor([0]),q,k,v,batch,norm,norm)
        with patch.object(fused_rotary, 'get_model', return_value=SimpleNamespace(kv_cache_dtype='auto')):
            with forward_context(ForwardContext(attn_backend=SimpleNamespace(token_to_kv_pool=outer))):
                run()
                with forward_context(ForwardContext(attn_backend=SimpleNamespace(token_to_kv_pool=inner))):
                    run()
                run()
        self.assertEqual(received, [(outer.k.data_ptr(),outer.v.data_ptr()),
                                    (inner.k.data_ptr(),inner.v.data_ptr()),
                                    (outer.k.data_ptr(),outer.v.data_ptr())])
        self.assertEqual(outer.layers, [7,7])
        self.assertEqual(inner.layers, [7])
        self.assertTrue(torch.equal(outer.k[0], k[0]))
        self.assertTrue(torch.equal(inner.v[0], v[0]))
        self.assertTrue(fused_rotary.has_fused_kv_store(batch, 7))
        self.assertFalse(hasattr(batch, 'token_to_kv_pool'))


if __name__ == '__main__':
    unittest.main()
