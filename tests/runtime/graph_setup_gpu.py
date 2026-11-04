from __future__ import annotations
import os
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import graph_setup as O


@unittest.skipUnless(os.environ.get('HCU_PLUGIN_RUN_GPU_TESTS')=='1','explicit GPU opt-in required')
class GraphSetupGpuTests(unittest.TestCase):
    def test_real_graph_replays_changed_input_after_native_cache_release(self):
        self.assertTrue(O.is_hcu())
        x=torch.arange(256,dtype=torch.float32,device='cuda');out=torch.empty_like(x)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):out.copy_(x*3+2)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out.copy_(x*3+2)
        runner=S(attn_backend=S(_hcu_flashmla_decode_h16_enabled=True))
        with patch.dict(os.environ,{'SGLANG_HCU_FLASHMLA_DECODE_H16':'1'}):self.assertTrue(O.release_hcu_flashmla_decode_h16_capture_cache(runner))
        for shift in [1,7]:
            x.add_(shift);graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(out,x*3+2,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
