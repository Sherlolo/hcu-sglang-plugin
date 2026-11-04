from __future__ import annotations

import json
import torch
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang import register
from sglang.srt.plugins.hook_registry import HookRegistry
HookRegistry.reset()
register()
HookRegistry.apply_hooks()
from sglang.srt.configs.mamba_utils import KimiLinearStateShape,KimiLinearCacheParams,Mamba2StateDType
from sglang.srt.mem_cache.memory_pool import MambaPool

shape=KimiLinearStateShape.create(tp_world_size=1,num_heads=2,head_dim=4,conv_kernel_size=4,num_spec=2)
params=KimiLinearCacheParams(shape=shape,layers=[2,3],dtype=Mamba2StateDType(conv=torch.bfloat16,temporal=torch.float32))
pool=MambaPool(size=3,spec_state_size=2,cache_params=params,mamba_layer_ids=params.layers,
               device='cuda:0',enable_memory_saver=False,speculative_num_draft_tokens=2,
               speculative_eagle_topk=1,enable_linear_replayssm_spec=True,linear_replayssm_cache_len=4)
assert pool.use_hcu_kda_replayssm
assert not pool._should_fuse_slot_ops()
src=torch.tensor([1],device='cuda:0');dst=torch.tensor([2],device='cuda:0')
state=pool.mamba_cache
for _ in range(2):
    pool.clear_slots(dst)
    pool.copy_from(src,dst)
torch.cuda.synchronize()
graph=torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    pool.clear_slots(dst)
    pool.copy_from(src,dst)
for value in (2.5,7.5):
    state.temporal[:,1].fill_(value)
    state.conv[0][:,1].fill_(value+1)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(state.temporal[:,2],torch.full_like(state.temporal[:,2],value))
    assert torch.equal(state.conv[0][:,2],torch.full_like(state.conv[0][:,2],value+1))
print(json.dumps({'status':'passed','device':torch.cuda.get_device_name(0),
                  'replays':2,'source_values':[2.5,7.5],
                  'scope':'Actual HCU pool allocation plus HIP Graph clear/copy/replay, with changed persistent state; no complete KDA kernel/model claim.'}))
