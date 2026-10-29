"""Canonical DSpark handshake for sharded-prefill and unsharded-decode peers."""
from __future__ import annotations
from typing import Optional
from sglang.srt.runtime_context import get_spec, get_parallel

def get_dsv41_spec_layout(kv_args: KVArgs) -> Optional[dict]:
    """Describe the positional transfer layout without pool capacities or pointers."""
    ratios = getattr(kv_args, "mla_compression_ratios", None) or []
    if 2 not in ratios or str(get_spec().speculative_algorithm).upper() != "DSPARK":
        return None

    from sglang.srt.disaggregation.base.conn import StateType

    if kv_args.state_types.count(StateType.SWA) != 2:
        raise RuntimeError(
            "DeepSeek-V4.1 DSpark PD requires target and draft SWA state"
        )

    layout = {
        "num_draft_tokens": get_spec().speculative_num_draft_tokens,
        "compression_ratios": list(ratios),
        "kv_layer_ids": list(kv_args.kv_layer_ids),
        "kv_item_lens": list(kv_args.kv_item_lens),
        "state_types": [state_type.value for state_type in kv_args.state_types],
        "state_item_lens": [list(items) for items in kv_args.state_item_lens],
    }
    if getattr(kv_args, "cp_cache_layer_split", False):
        # Handshake against the global logical layout, not a rank's owned
        # buffers. Actual copies still use v4_transfer_metadata descriptors.
        metadata = kv_args.v4_transfer_metadata
        descriptors = [metadata["kv"], *metadata["states"]]
        item_lens = [kv_args.kv_item_lens, *kv_args.state_item_lens]
        if len(descriptors) != len(item_lens) or any(
            len(keys) != len(sizes) for keys, sizes in zip(descriptors, item_lens)
        ):
            raise RuntimeError("Incomplete LayerSplit DSPARK transfer descriptors")
        local = [list(zip(keys, sizes)) for keys, sizes in zip(descriptors, item_lens)]
        ranks = get_parallel().attn_cp_group.all_gather_object(local)
        merged = [{} for _ in local]
        for rank in ranks:
            if len(rank) != len(merged):
                raise RuntimeError("Incomplete LayerSplit DSPARK transfer descriptors")
            for entries, destination in zip(rank, merged):
                for descriptor, size in entries:
                    key = tuple(descriptor)
                    if key in destination and destination[key] != size:
                        raise RuntimeError(
                            f"LayerSplit DSPARK page ABI differs across CP ranks: {key}"
                        )
                    destination[key] = size
        # V4.1's unsharded pool registers ratio-1 then ratio-2, each with
        # KV, indexer and optional separate scales, in source-layer order.
        family_order = {
            f"c{ratio}_{family}": order
            for order, (ratio, family) in enumerate(
                (ratio, family)
                for ratio in (1, 2)
                for family in ("kv", "indexer", "indexer_scale")
            )
        }
        kv_keys = sorted(merged[0], key=lambda key: (family_order[key[0]], key[1]))
        # Layer IDs must describe the same global positional ABI as sizes,
        # not this CP rank's subset of owned KV and replicated indexer pages.
        layout["kv_layer_ids"] = [key[1] for key in kv_keys]
        layout["kv_item_lens"] = [merged[0][key] for key in kv_keys]
        layout["state_item_lens"] = [
            [entries[key] for key in sorted(entries)] for entries in merged[1:]
        ]
    elif metadata := getattr(kv_args, "v4_transfer_metadata", None):
        # The unsharded V4.1 pool's legacy layer IDs only cover C4/C128
        # (and are empty for C1/C2). Use its complete transfer descriptors
        # so decode and LayerSplit prefill compare the same logical layout.
        if len(metadata["kv"]) != len(kv_args.kv_item_lens):
            raise RuntimeError("Incomplete DSPARK transfer descriptors")
        layout["kv_layer_ids"] = [key[1] for key in metadata["kv"]]
    return layout

