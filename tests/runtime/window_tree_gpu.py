from __future__ import annotations
import os,unittest
import torch
import window_tree_cpu as F
N = F.N


@unittest.skipUnless(os.environ.get('HCU_PLUGIN_RUN_GPU_TESTS')=='1','explicit GPU opt-in required')
class WindowTreeGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):F.WindowTreeCpuTests.setUpClass()

    def test_actual_gpu_window_storage_clone_split_and_component_ledgers(self):
        helper=F.WindowTreeCpuTests();tree,key,node=helper.tree(device='cuda')
        rows=torch.tensor([22,23,24,25],device='cuda',dtype=torch.int64)
        tree.attach_swa_window(key,2,6,rows,component_type=N.ComponentType.AUXILIARY_SWA);rows.fill_(-1);torch.cuda.synchronize()
        nodes=[child for child,_,_ in tree._walk_span(key,8)];actual=nodes[1].component_data[N.ComponentType.AUXILIARY_SWA].value
        self.assertEqual(actual.device.type,'cuda');self.assertEqual(actual.cpu().tolist(),[22,23,24,25]);self.assertNotEqual(actual.data_ptr(),rows.data_ptr())
        self.assertEqual(torch.cat([n.component_data[N.ComponentType.SWA].value for n in nodes]).cpu().tolist(),list(range(100,108)))
        self.assertEqual(tree.component_evictable_size_[N.ComponentType.AUXILIARY_SWA],4);self.assertEqual(tree.swa_tombstone_ranges(key,0,8,component_type=N.ComponentType.AUXILIARY_SWA),[(0,2),(6,8)])


if __name__=='__main__':unittest.main()
