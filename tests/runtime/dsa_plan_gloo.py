from __future__ import annotations

import socket
from types import SimpleNamespace as S
from unittest.mock import patch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class Batch:
    pass


def worker(rank,port):
    from hcu_sglang.compat.aiter import bootstrap_aiter_compat
    bootstrap_aiter_compat()
    from hcu_sglang.environ import register_envs
    register_envs()
    from hcu_sglang.mem_cache import dsa,dsa_cache_layer_split as P
    from sglang.srt.plugins.hook_registry import HookRegistry
    HookRegistry.reset();dsa.register_hooks();HookRegistry.apply_hooks()
    from hcu_sglang import dsa_utils as U
    U.register_hooks();HookRegistry.apply_hooks()
    from sglang.srt.mem_cache import memory_pool as N
    from sglang.srt.mem_cache.dsa_cache_layer_split import LayerSplitDSATokenToKVPool
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.model_executor import forward_context as F
    dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2)
    cp=S(world_size=2,pynccl_comm=None,ranks=[0,1],device_group=dist.group.WORLD)
    with get_parallel().override(attn_dcp_size=1),patch.object(P,'get_parallel',return_value=S(attn_cp_group=cp)):
        pool=LayerSplitDSATokenToKVPool(size=128,page_size=64,kv_lora_rank=2,dtype=torch.bfloat16,qk_rope_head_dim=2,layer_num=2,device='cpu',index_head_dim=128,enable_memory_saver=False,kv_cache_dim=4,layer_shard_rank=rank,layer_shard_size=2)
        reqs=N.ReqToTokenPool(2,192,'cpu',False)
        reqs.req_to_token[1,:64]=torch.arange(128,192)
        batch=Batch();batch.forward_mode=S(is_extend_without_speculative=lambda:True);batch.can_run_tbo=False;batch.tbo_parent_token_range=None
        batch.extend_prefix_lens_cpu=[64];batch.out_cache_loc=torch.tensor([64,65]);batch.req_pool_indices=torch.tensor([1]);batch.dsa_layer_split_main_kv_page_plan=None
        with patch.object(U.U,'get_token_to_kv_pool',return_value=pool),patch.object(U,'dsa_use_prefill_cp',return_value=True),patch.object(F,'get_req_to_token_pool',return_value=reqs):
            U.U.maybe_configure_main_kv_page_plan(batch)
        values=torch.arange(192*4,dtype=torch.float32).view(192,1,4).to(torch.bfloat16)
        if rank==0:pool.kv_buffer[0].copy_(values)
        else:assert pool.kv_buffer[0].numel()==0
        pool.prefetch_kv_buffer(0)
        assert torch.equal(pool.remote_kv_buffer[64:128],values[128:192])
        current=torch.full((2,1,4),9,dtype=torch.bfloat16)
        def scatter(buffer,loc,nope,rope):buffer[loc]=torch.cat((nope,rope),dim=-1)
        def gather(buffer,loc,nope,rope):
            nope.copy_(buffer[loc,:,:2]);rope.copy_(buffer[loc,:,2:])
        with patch.object(pool,'_write_mla_kv_buffer',side_effect=scatter):
            pool.set_mla_kv_buffer(S(layer_id=0),batch.out_cache_loc,current[...,:2],current[...,2:])
        with patch.object(N,'get_mla_kv_buffer_triton',side_effect=gather):
            nope,rope=pool.get_mla_kv_buffer(S(layer_id=0),torch.tensor([128,129,64,65]))
        actual=torch.cat((nope,rope),dim=-1)
        expected=torch.cat((values[128:130],current),dim=0)
        assert torch.equal(actual,expected),(rank,actual,expected)
        assert pool.remote_kv_buffer[:64].count_nonzero().item()==0
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    mp.spawn(worker,args=(port,),nprocs=2,join=True)
    print('PASS: two real Gloo ranks, public batch-plan producer -> Source owned pool -> compact history payload -> current mirrored writes -> translated public MLA reader; only GPU scatter/gather math is a CPU fixture')
