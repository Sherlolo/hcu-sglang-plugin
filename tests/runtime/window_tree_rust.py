from __future__ import annotations
import argparse,json,os
from pathlib import Path
from array import array


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();env=json.loads(args.config.read_text())['env'];os.environ.update(env)
    from hcu_sglang.compat.aiter import bootstrap_aiter_compat
    bootstrap_aiter_compat()
    from hcu_sglang import register
    from sglang.srt.plugins.hook_registry import HookRegistry
    register();HookRegistry.apply_hooks()
    import torch
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as CT
    from sglang.srt.mem_cache.unified_cache import tree_core_registry as G
    params=CacheInitParams(disable=False,req_to_token_pool=None,token_to_kv_pool_allocator=None,page_size=2,sliding_window_size=4,tree_components=(CT.FULL,CT.SWA),tree_core_backend='rust')
    resolved=G.select_tree_core_backend(params);assert resolved=='rust',resolved
    tree=G.create_tree_core('rust',params,{CT.FULL:None,CT.SWA:None});assert tree.swa_tombstone_ranges.__module__=='hcu_sglang.mem_cache.window_rust'
    key=RadixKey(array('q',range(8)),None);step=tree.begin_insert(InsertParams(key=key,value=torch.arange(8),component_evicted_seqlens={CT.SWA:8}))
    while step.result is None:
        assert not step.actions,step.actions
        step=tree.resume_insert()
    node_id=step.result.last_device_node;assert not tree.end_insert()
    assert tree.swa_tombstone_ranges(key,0,8)==[(0,8)]
    actions=tree.attach_swa_window(key,2,6,torch.arange(22,26));assert not actions
    assert tree.swa_tombstone_ranges(key,0,8)==[(0,2),(6,8)]
    tree.attach_swa_window(key,0,2,torch.arange(20,22));tree.attach_swa_window(key,6,8,torch.arange(26,28));assert tree.swa_tombstone_ranges(key,0,8)==[]
    assert tree.component_evictable_size(CT.SWA)==8
    try:tree.swa_tombstone_ranges(key,0,8,component_type=CT.AUXILIARY_SWA)
    except NotImplementedError:pass
    else:raise AssertionError('Source Rust single-window guard missing')
    args.output.parent.mkdir(parents=True,exist_ok=True);assert not args.output.exists();args.output.write_text(json.dumps({'status':'passed','backend':resolved,'adapter_type':type(tree).__module__+'.'+type(tree).__qualname__,'method_owner':tree.swa_tombstone_ranges.__module__,'restored_swa_tokens':tree.component_evictable_size(CT.SWA),'aux_guard':'NotImplementedError','scope':'Actual native Rust CPU binding via selected factory; no GPU Rust/AUX/storage/model claim.'},indent=2)+'\n');print('actual Rust window pass',node_id)


if __name__=='__main__':main()
