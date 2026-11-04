from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace as S
import json
import os
import torch
import torch.distributed as dist
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang.eplb import recording
from sglang.srt.eplb import expert_distribution as N
from sglang.srt.plugins.hook_registry import HookRegistry


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--backend',choices=['gloo','nccl'],default='gloo');args=parser.parse_args()
    torch.set_num_threads(1)
    if args.backend=='nccl':torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    HookRegistry.reset();recording.register_hooks();HookRegistry.apply_hooks()
    dist.init_process_group(args.backend);rank=dist.get_rank();world=dist.get_world_size();assert world in (2,8)
    device=torch.device('cuda',int(os.environ['LOCAL_RANK'])) if args.backend=='nccl' else torch.device('cpu')
    if world==2:
        rows_by_case=[([[1,2,3,4]] if rank==0 else [[5,6,7,8]]*3),([] if rank==0 else [[2**30,1,2**30,2]]*2)]
        expected=[torch.tensor([[40,48]],dtype=torch.int64,device=device),torch.tensor([[2**32,6]],dtype=torch.int64,device=device)]
    else:
        rows_by_case=[[[rank+1,2,rank+3,4]]*(rank+1),([] if rank!=7 else [[2**30,1,2**30,2]]*2)]
        expected=[torch.tensor([[480,216]],dtype=torch.int64,device=device),torch.tensor([[2**32,6]],dtype=torch.int64,device=device)]
    records=[]
    try:
        for index,rows in enumerate(rows_by_case):
            obj=object.__new__(N._StatAccumulator);obj._rank=rank;obj._first_dump=False;obj._enable=False
            obj._expert_location_metadata=S(num_layers=1,num_logical_experts=2,physical_to_logical_map=torch.tensor([[0,1,0,1]],device=device))
            obj._global_physical_count_of_buffered_step=N._Buffer.init_new((1,4),-1,torch.int32,device)
            for row in rows:obj._global_physical_count_of_buffered_step.append(torch.tensor([row],dtype=torch.int32,device=device))
            result=obj.dump('object')['logical_count']
            torch.testing.assert_close(result,expected[index],rtol=0,atol=0)
            records.append({'case':index,'local_steps':len(rows),'dtype':str(result.dtype),'shape':list(result.shape),'global_counts':result.cpu().tolist()})
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/('rank'+str(rank)+'.json')).write_text(json.dumps({'status':'passed','rank':rank,'world_size':world,'backend':args.backend,'cases':records},indent=2)+'\n')
        dist.barrier()
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
