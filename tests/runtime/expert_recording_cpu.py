from __future__ import annotations
from types import SimpleNamespace as S
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry


class ExpertRecordingCpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):HookRegistry.reset();register();HookRegistry.apply_hooks()

    def test_auto_factory_uses_dispatch_counts_from_both_modes(self):
        from sglang.srt.eplb import expert_distribution as N
        from hcu_sglang.eplb import recording as O
        cfg=S(moe=S(expert_distribution_recorder_mode='stat',moe_a2a_backend='deepep',deepep_mode='auto',elastic_ep_backend=None))
        metadata=S(num_layers=2,num_local_physical_experts=2,num_physical_experts=4)
        with patch.object(N,'get_exec',return_value=cfg),patch.object(N,'get_device',return_value='cpu'):
            gatherer=N._SinglePassGatherer.init_new(metadata,rank=0)
        self.assertIs(type(gatherer),O._DeepepAutoSinglePassGatherer)
        gatherer.on_deepep_dispatch_normal(0,[1,2],None,None,None)
        gatherer.on_deepep_dispatch_low_latency(0,torch.tensor([3,4],dtype=torch.int32))
        gatherer.on_select_experts(0,torch.zeros(64,8,dtype=torch.int32))
        torch.testing.assert_close(gatherer._data,torch.tensor([[4,6],[0,0]],dtype=torch.int32),rtol=0,atol=0)
        with self.assertRaisesRegex(ValueError,'shape mismatch'):gatherer.on_deepep_dispatch_normal(0,[1,2,3],None,None,None)

    def test_forward_capture_does_not_collect_or_append(self):
        from sglang.srt.eplb import expert_distribution as N
        events=[]
        rec=object.__new__(N._ExpertDistributionRecorderReal)
        rec._recording=True;rec._record_current_pass=True
        rec._single_pass_gatherers={'g':S(collect=lambda:events.append('collect') or {'count':1})}
        rec._accumulator=S(append=lambda *args:events.append(args))
        with patch.object(torch,'get_device_module',return_value=S(is_current_stream_capturing=lambda:True)):
            rec._on_forward_pass_end(7,{})
        self.assertEqual(events,[])
        with patch.object(torch,'get_device_module',return_value=S(is_current_stream_capturing=lambda:False)):
            rec._on_forward_pass_end(7,{})
        self.assertEqual(events[0],'collect');self.assertEqual(events[1],(7,'g',{'count':1},{}))

    def accumulator(self,steps):
        from sglang.srt.eplb import expert_distribution as N
        obj=object.__new__(N._StatAccumulator);obj._rank=0;obj._first_dump=False;obj._enable=False
        obj._expert_location_metadata=S(num_layers=1,num_logical_experts=2,physical_to_logical_map=torch.tensor([[0,1,0,1]]))
        obj._global_physical_count_of_buffered_step=N._Buffer.init_new((1,4),-1,torch.int32,'cpu')
        for value in steps:obj._global_physical_count_of_buffered_step.append(torch.tensor([value],dtype=torch.int32))
        return obj

    def test_local_sum_is_int64_fixed_shape_and_keeps_large_totals(self):
        from sglang.srt.eplb import expert_distribution as N
        rows=[[2**30,3,2**30,4],[2**30,5,2**30,6],[7,8,9,10]]
        obj=self.accumulator(rows);seen=[]
        def reduce(value,op):seen.append((tuple(value.shape),value.dtype));value.mul_(2)
        with patch.object(N.torch.distributed,'all_reduce',side_effect=reduce):result=obj.dump('object')
        expected=torch.tensor(rows,dtype=torch.int64).sum(0);expected=torch.tensor([[int(expected[0]+expected[2]),int(expected[1]+expected[3])]])*2
        self.assertEqual(seen,[((1,2),torch.int64)])
        torch.testing.assert_close(result['logical_count'],expected,rtol=0,atol=0)
        self.assertIsNone(result['average_utilization_rate_over_window'])

    def test_empty_window_has_same_collective_shape(self):
        from sglang.srt.eplb import expert_distribution as N
        obj=self.accumulator([]);seen=[]
        with patch.object(N.torch.distributed,'all_reduce',side_effect=lambda value,op:seen.append((tuple(value.shape),value.dtype))):result=obj.dump('object')
        self.assertEqual(seen,[((1,2),torch.int64)])
        torch.testing.assert_close(result['logical_count'],torch.zeros(1,2,dtype=torch.int64),rtol=0,atol=0)


    def test_public_accumulator_file_dump_writes_reduced_payload(self):
        import os
        from sglang.srt.eplb import expert_distribution as N
        obj=self.accumulator([[1,2,3,4]])
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,{'SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR':root}),patch.object(N.torch.distributed,'all_reduce'):
            self.assertIsNone(obj.dump('file'))
            files=list(Path(root).glob('expert_distribution_recorder_*.pt'));self.assertEqual(len(files),1)
            data=torch.load(files[0],map_location='cpu',weights_only=True)
            torch.testing.assert_close(data['logical_count'],torch.tensor([[4,6]],dtype=torch.int64),rtol=0,atol=0)


    def test_moved_source_method_global_names_resolve(self):
        import builtins
        import symtable
        from hcu_sglang.eplb import recording as O
        table=symtable.symtable(Path(O.__file__).read_text(),O.__file__,'exec')
        missing=[]
        for child in table.get_children():
            if child.get_type()!='function':continue
            for symbol in child.get_symbols():
                name=symbol.get_name()
                if symbol.is_global() and symbol.is_referenced() and name not in vars(O) and not hasattr(builtins,name):missing.append((child.get_name(),name))
        self.assertEqual(missing,[])

    def test_dump_file_respects_configured_directory_and_preserves_payload(self):
        import os
        from sglang.srt.eplb import expert_distribution as N
        with tempfile.TemporaryDirectory() as root,patch.dict(os.environ,{'SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR':str(Path(root)/'nested')}):
            N._dump_to_file('sample.pt',{'count':torch.tensor([2**34],dtype=torch.int64)})
            data=torch.load(Path(root)/'nested/sample.pt',weights_only=True)
            torch.testing.assert_close(data['count'],torch.tensor([2**34],dtype=torch.int64),rtol=0,atol=0)


if __name__=='__main__':unittest.main()
