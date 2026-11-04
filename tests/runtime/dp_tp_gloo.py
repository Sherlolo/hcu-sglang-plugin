from __future__ import annotations

import socket
from types import SimpleNamespace as S
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank,port):
    from hcu_sglang.compat.aiter import bootstrap_aiter_compat
    bootstrap_aiter_compat()
    from hcu_sglang import distributed_backend as T
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=4)
    groups=[]
    class Group:
        def __init__(self,group,ranks):self.group=group;self.ranks=ranks
        def all_reduce(self,tensor):dist.all_reduce(tensor,group=self.group);return tensor
    def create(group_ranks,local_rank,backend,**kwargs):
        selected=None
        for ranks in group_ranks:
            group=dist.new_group(ranks,backend='gloo')
            groups.append(group)
            if rank in ranks:selected=Group(group,ranks)
        return selected
    with patch.object(T,'is_hcu',return_value=True),patch.object(T.P,'get_parallel',return_value=S(tp_size=4,pp_size=1,attn_cp_size=2,attn_dp_size=2)),patch.object(T.P,'get_world_group',return_value=S(local_rank=rank)),patch.object(T.P,'init_model_parallel_group',side_effect=create),patch.object(T.P,'_ATTN_DP_TP',None,create=True):
        T.around_initialize_model_parallel(Mock(),backend='gloo')
        tensor=torch.tensor([rank+1],dtype=torch.int64)
        group=T.get_attn_dp_tp_group()
        group.all_reduce(tensor)
        assert group.ranks==([0,2] if rank%2==0 else [1,3]),(rank,group.ranks)
        assert tensor.item()==(4 if rank%2==0 else 6),(rank,tensor)
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    mp.spawn(worker,args=(port,),nprocs=4,join=True)
    print('PASS: four real Gloo ranks, DP+TP groups exclude CP peers')
