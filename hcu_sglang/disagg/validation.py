"""Validate the complete descriptor snapshot before publishing a CP peer."""
from sglang.srt.runtime_context import get_memory
from hcu_sglang.mem_cache.cp_cache_layer_split.transfer import match_transfer_entries
from .protocol import resolve_state_component_dst_index, StateType


def _check_vectors(keys,pointers,sizes,label):
    if len(keys)!=len(pointers) or len(keys)!=len(sizes):
        raise RuntimeError(f'{label} descriptors, pointers and byte strides are incomplete')


def around_validate_peer_state_layout(original,manager,peer):
    if not getattr(manager,'cp_cache_layer_split',False):
        return original(manager,peer)
    try:
        source=manager.kv_args.v4_transfer_metadata
        destination=peer.v4_transfer_metadata
        if source.get('version')!=1 or destination.get('version')!=1:
            raise RuntimeError('LayerSplit requires V4 descriptors on both prefill and decode')
        source_keys=source['kv'];destination_keys=destination['kv']
        source_sizes=manager.kv_args.kv_item_lens
        destination_sizes=peer.dst_kv_item_lens
        if list(destination.get('kv_sizes',[]))!=list(destination_sizes):
            raise RuntimeError('LayerSplit destination KV sizes disagree with registration fields')
        _check_vectors(source_keys,manager.kv_args.kv_data_ptrs,source_sizes,'Source KV')
        _check_vectors(destination_keys,peer.dst_kv_ptrs,destination_sizes,'Destination KV')
        match_transfer_entries(source_keys,destination_keys,source_sizes,destination_sizes)
        source_types=manager.kv_args.state_types
        destination_types=peer.dst_state_types
        if len(source['states'])!=len(source_types) or len(destination['states'])!=len(destination_types):
            raise RuntimeError('LayerSplit state descriptors/types are incomplete')
        for i,state_type in enumerate(source_types):
            j=resolve_state_component_dst_index(source_types,destination_types,i)
            pointers=manager.kv_args.state_data_ptrs[i]
            sizes=manager.kv_args.state_item_lens[i]
            dst_pointers=peer.dst_state_data_ptrs[j]
            dst_sizes=peer.dst_state_item_lens[j]
            keys=source['states'][i];dst_keys=destination['states'][j]
            if keys:
                _check_vectors(keys,pointers,sizes,'Source state')
                _check_vectors(dst_keys,dst_pointers,dst_sizes,'Destination state')
                match_transfer_entries(keys,dst_keys,sizes,dst_sizes)
            elif manager._is_generic_kvcache_state_type(state_type):
                mismatch=manager._state_stride_mismatch(
                    src_data_ptrs=pointers,dst_data_ptrs=dst_pointers,
                    item_lens=sizes,dst_item_lens=dst_sizes,state_type=state_type,
                    force_flat=(state_type in (StateType.QSA_PENDING,StateType.QSA_COMPRESSED)
                        or get_memory().enable_unified_memory),
                    src_layer_ids=manager.kv_args.state_layer_ids[i],
                    dst_layer_ids=peer.dst_state_layer_ids[j])
                if mismatch is not None:
                    raise RuntimeError(mismatch)
        return True,None
    except (RuntimeError,KeyError,IndexError,TypeError,ValueError) as error:
        return False,'LayerSplit peer layout rejected: '+str(error)
