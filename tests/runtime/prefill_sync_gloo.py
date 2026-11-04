from __future__ import annotations
import argparse,json,os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
import torch
import torch.distributed as dist
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang import prefill_graph_sync as O
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.srt.managers.scheduler_components import dp_attn as N


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    torch.set_num_threads(1);HookRegistry.reset();O.register_hooks();HookRegistry.apply_hooks();dist.init_process_group('gloo',timeout=timedelta(seconds=60));rank=dist.get_rank();assert dist.get_world_size()==4
    subgroup=dist.new_group(ranks=[1,2,3],backend='gloo',timeout=timedelta(seconds=30));results=[]
    try:
        parallel=S(tp_group=S(active_ranks_cpu=torch.ones(4,dtype=torch.int32)))
        with patch.object(N,'get_parallel',return_value=parallel),patch.object(N,'_ENABLE_METRICS_DP_ATTENTION',False):
            for case,cp,reduce in [('dp4_cpu_gather_broadcast',1,False),('dp2_cp2_mixed',2,False),('dp4_world_reduce',1,True)]:
                tokens=[0,8,5,17] if cp==1 else [8,8,5,5];counts=[0]*4 if cp==1 else [4,4,0,0]
                obj=N.MLPSyncBatchInfo(num_dp_ranks=4//cp,tp_size=1,cp_size=cp,num_tokens=tokens[rank],num_tokens_for_logprob=1,cp_num_tokens=counts[rank],can_run_decode_cuda_graph=False,can_run_draft_cuda_graph=True,can_run_prefill_cuda_graph=rank!=2,is_extend_in_batch=True,local_can_run_tbo=True,local_forward_mode=N.ForwardMode.EXTEND.value)
                obj.all_gather(device='cpu',group=dist.group.WORLD,use_all_reduce=reduce)
                expected=tokens if cp==1 else [8,5]
                assert obj.global_num_tokens==expected and obj.global_cp_num_tokens==[0]*(4//cp)
                assert not obj.can_run_prefill_cuda_graph and obj.tp0_info_cpu.shape==(4//cp,12)
                results.append({'case':case,'global_tokens':obj.global_num_tokens,'cp_tokens':obj.global_cp_num_tokens,'shape':list(obj.tp0_info_cpu.shape),'prefill_vote':obj.can_run_prefill_cuda_graph})
        if rank!=0:
            local=torch.tensor([rank,rank+10],dtype=torch.int64);global_flat=torch.empty(6,dtype=torch.int64)
            O.exchange_sync_rows(global_flat,local,subgroup);assert global_flat.tolist()==[1,11,2,12,3,13]
            results.append({'case':'nonzero_global_root_subgroup','rows':global_flat.tolist()})
        dist.barrier();args.output_dir.mkdir(parents=True,exist_ok=True);p=args.output_dir/('rank'+str(rank)+'.json');assert not p.exists();p.write_text(json.dumps({'status':'passed','rank':rank,'world_size':4,'backend':'gloo','cases':results},indent=2)+'\n');dist.barrier()
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
