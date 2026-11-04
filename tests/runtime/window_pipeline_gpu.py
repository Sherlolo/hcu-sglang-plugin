from __future__ import annotations
import os,unittest
import torch
import window_pipeline_cpu as F


@unittest.skipUnless(os.environ.get('HCU_PLUGIN_RUN_GPU_TESTS')=='1','explicit GPU opt-in required')
class WindowPipelineGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):F.WindowPipelineCpuTests.setUpClass()

    def test_public_two_window_h2d_from_pinned_host_then_ack_releases_full_bounces(self):
        f=F.Fixture(device='cuda');f.pipe.prepare_staged_prefetch(f.req)
        result=f.pipe.init_load_back(F.InitLoadBackParams(best_match_node=f.req.last_node,host_hit_length=2,req=f.req));self.assertEqual(result[0],2)
        self.assertFalse(any(e[0]=='host_free' for e in f.events));self.assertEqual(f.data[F.PN.KV][86:88].cpu().tolist(),[-1,-1])
        f.drain();self.assertTrue(f.copy_complete_event.query());self.assertTrue(all(p.is_pinned() for p in f.pinned_payloads))
        self.assertEqual(f.data[F.PN.KV][86:88].cpu().tolist(),[1006,1007]);self.assertEqual(f.data[F.PN.SWA][20:24].cpu().tolist(),[2102,2103,2106,2107]);self.assertEqual(f.data[F.PN.AUXILIARY_SWA][40:44].cpu().tolist(),[3204,3205,3206,3207])
        self.assertTrue(f.pipe.try_finish_load_back(-10));self.assertEqual([len(e[2]) for e in f.events if e[0]=='host_free'],[8,8,8]);self.assertEqual(f.cc.prefetch_tokens_occupied,0)


if __name__=='__main__':unittest.main()
