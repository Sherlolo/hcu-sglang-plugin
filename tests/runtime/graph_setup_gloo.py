from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
import torch
import torch.distributed as dist
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import graph_setup as O
from sglang.srt.model_executor.model_runner_components import cuda_graph_setup as N
from sglang.srt.layers.moe.moe_runner import deep_gemm as G
from sglang.srt.utils import common as C
from sglang.srt.plugins.hook_registry import HookRegistry


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1);HookRegistry.reset();O.register_hooks();HookRegistry.apply_hooks()
    dist.init_process_group('gloo',timeout=timedelta(seconds=60));rank=dist.get_rank();assert dist.get_world_size()==4
    groups=[dist.new_group(ranks=ranks,backend='gloo',timeout=timedelta(seconds=30)) for ranks in [[0,1],[2,3]]]
    records=[]
    try:
        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ,{'SGLANG_DEEPGEMM_STANDARD_LAYOUT':'auto','SGLANG_DEEPGEMM_MASKED_MEMORY_BUDGET_FRACTION':'0.25'}))
            stack.enter_context(patch.object(N,'get_exec',return_value=S(moe=S(ep_join_mode=None,moe_runner_backend='deep_gemm',moe_a2a_backend='none'))))
            stack.enter_context(patch.object(N,'get_spec',return_value=S(speculative_moe_runner_backend='deep_gemm',speculative_moe_a2a_backend=None)))
            for key,val in [('device_count',lambda:4),('current_device',lambda:rank),('get_device_properties',lambda i:S(is_integrated=False))]:stack.enter_context(patch.object(torch.cuda,key,val))
            stack.enter_context(patch.object(C,'empty_device_cache',lambda *a:None))
            runner=S(device='cuda',gpu_id=rank,is_draft_worker=True,model_config=S(quantization='fp8'))
            parallel=S(pp_size=2,tp_group=S(world_size=2,cpu_group=groups[rank//2]),world_group=S(cpu_group=dist.group.WORLD),launch_world_size=4)
            with patch.object(N,'get_parallel',return_value=parallel),patch.object(torch.cuda,'mem_get_info',return_value=((16-rank*2)*(1<<30),32*(1<<30))):
                if rank>=2:
                    N.refresh_deep_gemm_layout_memory_budget(runner)
                    assert N._deep_gemm_layout_memory_budget_initialized
                    assert G._masked_standard_layout_memory_budget_bytes==int(10*(1<<30)*.25)
                    records.append({'case':'pp2_last_stage_draft','budget_bytes':G._masked_standard_layout_memory_budget_bytes,'participants':[2,3]})
                else:
                    value=torch.tensor(rank+1,dtype=torch.int64);dist.all_reduce(value,group=groups[0]);assert value.item()==3
                    records.append({'case':'pp2_earlier_stage_other_collective','sum':value.item(),'participants':[0,1]})
            dist.barrier()
            parallel.pp_size=1
            with patch.object(N,'get_parallel',return_value=parallel),patch.object(torch.cuda,'mem_get_info',return_value=((16-rank)*(1<<30),32*(1<<30))):
                N.refresh_deep_gemm_layout_memory_budget(runner)
                assert G._masked_standard_layout_memory_budget_bytes==int(13*(1<<30)*.25)
                records.append({'case':'pp1_world_budget','budget_bytes':G._masked_standard_layout_memory_budget_bytes,'participants':[0,1,2,3]})
        args.output_dir.mkdir(parents=True,exist_ok=True);out=args.output_dir/('rank'+str(rank)+'.json');assert not out.exists()
        out.write_text(json.dumps({'status':'passed','rank':rank,'backend':'gloo','world_size':4,'cases':records,'scope':'Real process-group collectives and native budget setter; GPU memory inquiry is synthetic, not GPU collective or model execution.'},indent=2)+'\n')
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
