from __future__ import annotations

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared, BlockRemoved, BlockStored, StorageMedium,
    kv_event_namespace_seed, namespaced_block_hash,
)
from hcu_sglang.mem_cache.kv_events import KvEventLoraNames


def _kv_event_from_tagged(
    event: tuple, lora_names: KvEventLoraNames, dynamo_format: bool = False
):
    """Build the Python KV cache event for one of the binding's tagged tuples.

    Rust hashes are LoRA-agnostic; the trailing extra_key selects the LoRA
    namespace, applied here as in KVCacheEventRecorder.
    """
    tag = event[0]
    if tag == "block_stored":
        lora_name = (
            lora_names.resolve(event[8]) if dynamo_format else lora_names.get(event[8])
        )
        seed = (
            kv_event_namespace_seed(extra_key=event[8], cache_salt=None)
            if not dynamo_format and lora_name is not None
            else None
        )
        parent = event[2]
        return BlockStored(
            block_hashes=[
                namespaced_block_hash(h, namespace_seed=seed) for h in event[1]
            ],
            parent_block_hash=(
                None
                if parent is None
                else namespaced_block_hash(parent, namespace_seed=seed)
            ),
            token_ids=event[3],
            block_size=event[4],
            lora_id=None,
            medium=StorageMedium(event[5]),
            cache_salt=event[6],
            session_id=event[7],
            lora_name=lora_name,
        )
    if tag == "block_removed":
        seed = (
            kv_event_namespace_seed(extra_key=event[3], cache_salt=None)
            if not dynamo_format and lora_names.get(event[3]) is not None
            else None
        )
        return BlockRemoved(
            block_hashes=[
                namespaced_block_hash(h, namespace_seed=seed) for h in event[1]
            ],
            medium=StorageMedium(event[2]),
        )
    if tag == "all_blocks_cleared":
        return AllBlocksCleared()
    raise ValueError(f"unknown kv event tag: {tag}")


class _RustKVCacheEventRecorder:
    """Expose the Rust event queue through the Python recorder interface."""

    def __init__(
        self,
        binding,
        enabled: bool,
        lora_names: KvEventLoraNames,
        dynamo_format: bool = False,
    ):
        self._binding = binding
        self.enabled = enabled
        self._lora_names = lora_names
        self._dynamo_format = dynamo_format

    def record_all_cleared(self) -> None:
        self._binding.record_all_cleared_event()

    def take(self) -> list:
        events = [
            _kv_event_from_tagged(event, self._lora_names, self._dynamo_format)
            for event in self._binding.take_events()
        ]
        return events
