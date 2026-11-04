from __future__ import annotations

import argparse
from array import array
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as S


def oracle(tokens, page, bigram=False, prior=None):
    logical = len(tokens) - 1 if bigram else len(tokens)
    digest = bytes.fromhex(prior) if prior else b''
    result = []
    for start in range(0, logical, page):
        stop = min(start + page, logical)
        words = [word for i in range(start, stop) for word in (tokens[i:i+2] if bigram else [tokens[i]])]
        wide = any(word > 2**32 - 1 for word in words)
        encoded = (b'\xff' if wide else b'') + b''.join(word.to_bytes(8 if wide else 4, 'little') for word in words)
        digest = hashlib.sha256(digest + encoded).digest()
        result.append(digest.hex())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['auto', 'never'], default='auto')
    args = parser.parse_args()
    os.environ.update(json.loads(args.config.read_text())['env'])
    os.environ['SGLANG_RUST_BUILD_MODE'] = args.mode
    from hcu_sglang.compat.aiter import bootstrap_aiter_compat
    bootstrap_aiter_compat()
    from hcu_sglang import register
    from sglang.srt.plugins.hook_registry import HookRegistry
    register()
    HookRegistry.apply_hooks()
    assert 'sglang.srt.mem_cache.rust_tree_core.extension' not in sys.modules
    import torch
    from hcu_sglang.mem_cache.rust_tree_core import provider as P
    from hcu_sglang.mem_cache.rust_tree_core.events import _kv_event_from_tagged
    from hcu_sglang.mem_cache.kv_events import KvEventLoraNames, KVCacheEventRecorder
    from hcu_sglang.mem_cache.cpp_utils import native_hash as H
    from sglang.srt.mem_cache.cpp_utils import native_hash as N
    from sglang.srt.mem_cache import utils as U
    assert U.get_native_hash is N.get_native_hash
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType as CT
    from sglang.srt.mem_cache.unified_cache import tree_core_registry as G
    names = KvEventLoraNames()
    for namespace, name in [('A', 'first'), ('B', 'second')]:
        names.register(S(lora_id=namespace, extra_key=namespace), name)
    def tree(dynamo=False, swa=False):
        params = CacheInitParams(False, None, None, 2, tree_components=(CT.FULL, CT.SWA) if swa else (CT.FULL,),
            sliding_window_size=4 if swa else None, tree_core_backend='rust',
            enable_kv_cache_events=True, dynamo_kv_event_format=dynamo, kv_event_lora_names=names)
        from unittest.mock import patch
        with patch.object(G, '_RUST_TREE_CORE_MANIFEST', Path('/missing-native-sources/Cargo.toml')):
            assert G.select_tree_core_backend(params) == 'rust'
        core = G.create_tree_core('rust', params, {ct:None for ct in params.tree_components})
        assert core._bindings is sys.modules[P._OWNED_MODULE]
        assert core.kv_events.__class__.__module__ == 'hcu_sglang.mem_cache.rust_tree_core.events'
        core.kv_events.take()
        return core
    def insert(core, key, evicted=None):
        params = InsertParams(key=key, value=torch.arange(len(key)))
        if evicted is not None:
            params.component_evicted_seqlens = {CT.SWA:evicted}
        step = core.begin_insert(params)
        while step.result is None:
            assert not step.actions, step.actions
            step = core.resume_insert()
        assert not core.end_insert()
        return step.result.last_device_node
    module = None
    cases = []
    event_ids = []
    for dynamo in [False, True]:
        for namespace in ['A', 'B', None]:
            core = tree(dynamo)
            module = core._bindings
            key = RadixKey(array('q', [1, 2, 3, 4]), namespace, cache_salt='salt')
            node_id = insert(core, key)
            raw = core._binding.take_events()
            assert raw and raw[0][0] == 'block_stored' and len(raw[0]) == 9, raw
            stored = [_kv_event_from_tagged(item, names, dynamo) for item in raw]
            root = TreeNode(); root.key = RadixKey(array('q')); root.hash_value = []
            node = TreeNode(); node.parent = root; node.key = key
            recorder = KVCacheEventRecorder(enabled=True, page_size=2, dynamo_format=dynamo, lora_names=names)
            recorder.record_store(node)
            python_event = recorder.take()[0]
            assert stored[0] == python_event, (stored, python_event)
            eviction = core.evict_device_leaf(node_id, False)
            values = eviction.device_frees.pop(CT.FULL)
            assert torch.equal(torch.cat(values), torch.arange(4))
            assert not eviction.device_frees and not eviction.host_frees
            assert eviction.tracker[CT.FULL] == 4
            raw_remove = core._binding.take_events()
            assert raw_remove and len(raw_remove[0]) == 4, raw_remove
            assert raw_remove[0][3] == namespace
            removed = [_kv_event_from_tagged(item, names, dynamo) for item in raw_remove]
            assert removed[0].block_hashes == stored[0].block_hashes
            if not dynamo:
                event_ids.append(stored[0].block_hashes)
            cases.append({'dynamo':dynamo, 'namespace':namespace, 'store_remove_python_parity':True, 'removed_tuple_items':4})
    assert event_ids[0] != event_ids[1] != event_ids[2]
    hash_cases = []
    prior = 'ab' * 32
    for bigram in [False, True]:
        for tokens in [[0, 1, 2, 3], [2**32-1, 7, 2**32, 9, 2**40], [2**63-1, 0, 3, 4]]:
            for seed in [None, prior]:
                expected = oracle(tokens, 2, bigram, seed)
                got = module.get_hash_str(array('q', tokens), seed, 2, is_bigram=bigram)
                key = RadixKey(array('q', tokens), is_bigram=bigram)
                cpp = N.get_native_hash(key, bytes.fromhex(seed) if seed else None, 2)
                assert got == cpp == expected, (got, cpp, expected)
                hash_cases.append({'bigram':bigram, 'prior':seed is not None, 'wide':max(tokens)>2**32-1})
    for tokens in [[-1, 2], [1, -2]]:
        for operation in [lambda:module.get_hash_str(array('q', tokens), None, 2),
                          lambda:N.get_native_hash(array('q', tokens), None, 2)]:
            try:operation()
            except (ValueError, RuntimeError, OverflowError, IndexError):pass
            else:raise AssertionError('negative token accepted')
    swa = tree(swa=True)
    key = RadixKey(array('q', range(8)))
    insert(swa, key, 8)
    assert swa.swa_tombstone_ranges(key, 0, 8) == [(0, 8)]
    swa.attach_swa_window(key, 0, 8, torch.arange(8))
    assert swa.swa_tombstone_ranges(key, 0, 8) == []
    assert swa.component_evictable_size(CT.SWA) == 8
    try:swa.swa_tombstone_ranges(key, 0, 8, component_type=CT.AUXILIARY_SWA)
    except NotImplementedError:pass
    else:raise AssertionError('Source single-window guard missing')
    result = {'status':'passed', 'mode':args.mode, 'binding_module':module.__name__,
        'binding_file':module.__file__, 'binding_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        'event_cases':cases, 'hash_cases':hash_cases, 'negative_rejections':4, 'swa_tokens_restored':8,
        'scope':'Actual native Rust factory and owned compiled Source crate, native insert/leaf eviction and raw tuple ABI, full Source event decoder/Python recorder parity, Source C++ hash/Rust/hashlib oracle. CPU tensors only; no Rust GPU/L3/model/event publisher claim.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','mode','binding_module','swa_tokens_restored']}))


if __name__ == '__main__':
    main()
