# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
"""HCU descriptor-aware Mooncake registration and byte-transfer methods."""
from __future__ import annotations
import concurrent.futures
import json
import struct
import threading
from typing import List, Optional, Tuple
import numpy as np
import numpy.typing as npt
import zmq
from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import _uses_flat_entry_layout, logger, MooncakeKVManager
from sglang.srt.disaggregation.utils import compute_mamba_state_slice_byte_blocks
from sglang.srt.disaggregation.utils import (
    build_transfer_entry_pairs, is_elided_entry, should_send_replicated_state,
    build_dsa_tail_transfer_blocks, slice_dsa_tail_dst_ptrs_for_pp,
)
from sglang.srt.disaggregation.common.utils import group_concurrent_contiguous, pack_int_lists
from sglang.srt.runtime_context import get_memory
from hcu_sglang.mem_cache.cp_cache_layer_split.transfer import match_transfer_entries
from .protocol import (StateType, TransferInfo, KVArgsRegisterInfo, pack_string_list,
    pack_state_types, encode_v4_transfer_metadata, resolve_state_component_dst_index)

def send_kvcache_generic(
    self,
    mooncake_session_id: str,
    src_data_ptrs: list[int],
    dst_data_ptrs: list[int],
    item_lens: list[int],
    prefill_data_indices: npt.NDArray[np.int32],
    dst_data_indices: npt.NDArray[np.int32],
    executor: concurrent.futures.ThreadPoolExecutor,
    state_type: Optional[StateType] = None,
    force_flat: bool = False,
    src_layer_ids: Optional[List[int]] = None,
    dst_layer_ids: Optional[List[int]] = None,
    dst_device_data_indices: Optional[npt.NDArray[np.int32]] = None,
    dst_device_data_ptrs: Optional[set[int]] = None,
    src_layout: Optional[list] = None,
    dst_layout: Optional[list] = None,
    dst_item_lens: list[int] | None = None,
    bootstrap_room: Optional[int] = None,
) -> int:
    """
    Generic KV cache transfer supporting both MHA and MLA architectures.
    This method is used by both send_kvcache (full pool) and maybe_send_extra.

    ``force_flat`` uses the MLA-style flat (single-buffer-per-layer) layout
    even on a non-MLA backend, for K-only state buffers (e.g. MiniMax sparse
    index) whose per-layer list must not be half-split into K/V.
    """
    # Host and device buffers may use different destination page spaces.
    # Build both transfer plans once, then select per destination buffer.
    prefill_kv_blocks, dst_kv_blocks = group_concurrent_contiguous(
        prefill_data_indices, dst_data_indices
    )
    device_prefill_kv_blocks = device_dst_kv_blocks = None
    if dst_device_data_indices is not None:
        device_prefill_kv_blocks, device_dst_kv_blocks = (
            group_concurrent_contiguous(
                prefill_data_indices, dst_device_data_indices
            )
        )

    layers_params = None

    if src_layout is not None:
        if len(src_layout) != len(src_data_ptrs):
            raise RuntimeError("LayerSplit source descriptors/pointers are incomplete")
        if dst_layout is None or len(dst_layout) != len(dst_data_ptrs):
            raise RuntimeError("LayerSplit destination descriptors/pointers are incomplete")
    # Descriptor matching below validates the byte ABI by identity. The legacy
    # positional pre-check cannot describe owner shards and reordered peers.
    if (
        src_layout is None
        and dst_item_lens is not None
        and mooncake_session_id not in self.state_strides_validated
    ):
        mismatch = self._state_stride_mismatch(
            src_data_ptrs=src_data_ptrs,
            dst_data_ptrs=dst_data_ptrs,
            item_lens=item_lens,
            dst_item_lens=dst_item_lens,
            state_type=state_type,
            force_flat=force_flat,
            src_layer_ids=src_layer_ids,
            dst_layer_ids=dst_layer_ids,
        )
        if mismatch is not None:
            assert bootstrap_room is not None
            logger.error(mismatch)
            self.conclude_failure(
                bootstrap_room=bootstrap_room,
                failure_reason=mismatch,
            )
            return -1

    # Decode pp size should be equal to prefill pp size or 1
    # Published layer IDs give exact pairing; plain-MHA peers publish none
    # and keep positional slicing.
    has_layer_ids = bool(src_layer_ids or dst_layer_ids)
    if src_layout is not None:
        if dst_layout is None or dst_item_lens is None:
            raise RuntimeError("LayerSplit PD requires V4 descriptors on both P and D")
        pairs = match_transfer_entries(
            src_layout, dst_layout, item_lens, dst_item_lens
        )
        layers_params = [
            (src_data_ptrs[i], dst_data_ptrs[j], item_lens[i]) for i, j in pairs
        ]
    elif _uses_flat_entry_layout(
        mla_backend=self.is_mla_backend or self.is_hybrid_mla_backend,
        force_flat=force_flat,
        has_layer_ids=has_layer_ids,
        state_type=state_type,
        num_src_entries=len(src_data_ptrs),
        num_dst_entries=len(dst_data_ptrs),
    ):
        # Layer IDs map PP-local buffers to global decode entries.
        # Registrations without them retain the existing PP mapping.
        if has_layer_ids:
            pairs = build_transfer_entry_pairs(
                src_layer_ids,
                dst_layer_ids,
                len(src_data_ptrs),
                len(dst_data_ptrs),
                allow_positional_fallback=self.pp_size == 1,
            )
            # An entry elided on either side moves no bytes and has no
            # registered buffer to write into.
            layers_params = [
                (src_data_ptrs[i], dst_data_ptrs[j], item_lens[i])
                for i, j in pairs
                if not is_elided_entry(
                    item_lens[i],
                    dst_item_lens,
                    j,
                    allow_dst_elision=state_type == StateType.DSA,
                )
            ]
        else:
            src_kv_ptrs, dst_kv_ptrs, layers_current_pp_stage = (
                self.get_mla_kv_ptrs_with_pp(
                    src_data_ptrs, dst_data_ptrs, state_type
                )
            )
            mapped_dst_lens = None
            if dst_item_lens is not None:
                _, mapped_dst_lens, _ = self.get_mla_kv_ptrs_with_pp(
                    item_lens, dst_item_lens, state_type
                )
            layers_params = [
                (
                    src_kv_ptrs[layer_id],
                    dst_kv_ptrs[layer_id],
                    item_lens[layer_id],
                )
                for layer_id in range(layers_current_pp_stage)
                if not is_elided_entry(
                    item_lens[layer_id],
                    mapped_dst_lens,
                    layer_id,
                    allow_dst_elision=state_type == StateType.DSA,
                )
            ]
    else:
        src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage = (
            self.get_mha_kv_ptrs_with_pp(src_data_ptrs, dst_data_ptrs)
        )
        # item_lens structure: [k_layer0, k_layer1, ..., k_layerN, v_layer0, v_layer1, ..., v_layerN]
        # Use correct item lengths for K and V separately
        if layers_current_pp_stage > len(dst_k_ptrs):
            logger.error(
                "Prefill transfer kvcache error, layers_current_pp_stage is out of range: "
                f"layers_current_pp_stage={layers_current_pp_stage}, len(dst_k_ptrs)={len(dst_k_ptrs)}"
            )
            return -1
        layers_params = [
            (
                src_k_ptrs[layer_id],
                dst_k_ptrs[layer_id],
                item_lens[layer_id],  # K item length
            )
            for layer_id in range(layers_current_pp_stage)
        ] + [
            (
                src_v_ptrs[layer_id],
                dst_v_ptrs[layer_id],
                item_lens[layers_current_pp_stage + layer_id],  # V item length
            )
            for layer_id in range(layers_current_pp_stage)
        ]
    assert layers_params is not None

    def set_transfer_blocks(
        src_ptr: int, dst_ptr: int, item_len: int
    ) -> List[Tuple[int, int, int]]:
        transfer_blocks = []
        if dst_device_data_ptrs and int(dst_ptr) in dst_device_data_ptrs:
            assert (
                device_prefill_kv_blocks is not None
                and device_dst_kv_blocks is not None
            )
            src_blocks, dst_blocks = (
                device_prefill_kv_blocks,
                device_dst_kv_blocks,
            )
        else:
            src_blocks, dst_blocks = prefill_kv_blocks, dst_kv_blocks
        for prefill_index, decode_index in zip(src_blocks, dst_blocks):
            src_addr = src_ptr + int(prefill_index[0]) * item_len
            dst_addr = dst_ptr + int(decode_index[0]) * item_len
            length = item_len * len(prefill_index)
            transfer_blocks.append((src_addr, dst_addr, length))
        return transfer_blocks

    # Worker function for processing a single layer
    def process_layer(src_ptr: int, dst_ptr: int, item_len: int) -> int:
        transfer_blocks = set_transfer_blocks(src_ptr, dst_ptr, item_len)
        return self._transfer_data(mooncake_session_id, transfer_blocks)

    # Worker function for processing all layers in a batch
    def process_layers(layers_params: List[Tuple[int, int, int]]) -> int:
        transfer_blocks = []
        for src_ptr, dst_ptr, item_len in layers_params:
            transfer_blocks.extend(set_transfer_blocks(src_ptr, dst_ptr, item_len))
        return self._transfer_data(mooncake_session_id, transfer_blocks)

    if (
        self.enable_custom_mem_pool
        and self.custom_mem_pool_type != "INTRA_NODE_NVLINK"
    ):
        futures = [
            executor.submit(
                process_layer,
                src_ptr,
                dst_ptr,
                item_len,
            )
            for (src_ptr, dst_ptr, item_len) in layers_params
        ]
        return self._await_transfer_futures(futures)
    else:
        # Combining all layers' params in one batch transfer is more efficient
        # compared to using multiple threads. Preserve this legacy path unless
        # users explicitly opt in to bounded index batches.
        max_batch_indices = self.max_transfer_batch_indices
        if max_batch_indices <= 0 or prefill_data_indices.size <= max_batch_indices:
            return process_layers(layers_params)

        def process_index_batch(
            prefill_blocks,
            dst_blocks,
            device_prefill_blocks=None,
            device_dst_blocks=None,
        ) -> int:
            transfer_blocks = []
            for src_ptr, dst_ptr, item_len in layers_params:
                if dst_device_data_ptrs and int(dst_ptr) in dst_device_data_ptrs:
                    assert (
                        device_prefill_blocks is not None
                        and device_dst_blocks is not None
                    )
                    src_blocks, target_blocks = (
                        device_prefill_blocks,
                        device_dst_blocks,
                    )
                else:
                    src_blocks, target_blocks = prefill_blocks, dst_blocks
                for prefill_index, decode_index in zip(src_blocks, target_blocks):
                    src_addr = src_ptr + int(prefill_index[0]) * item_len
                    dst_addr = dst_ptr + int(decode_index[0]) * item_len
                    length = item_len * len(prefill_index)
                    transfer_blocks.append((src_addr, dst_addr, length))
            return self._transfer_data(mooncake_session_id, transfer_blocks)

        for start in range(
            0,
            prefill_data_indices.size,
            max_batch_indices,
        ):
            batch_prefill_blocks, batch_dst_blocks = group_concurrent_contiguous(
                prefill_data_indices[start : start + max_batch_indices],
                dst_data_indices[start : start + max_batch_indices],
            )
            batch_device_prefill_blocks = batch_device_dst_blocks = None
            if dst_device_data_indices is not None:
                batch_device_prefill_blocks, batch_device_dst_blocks = (
                    group_concurrent_contiguous(
                        prefill_data_indices[start : start + max_batch_indices],
                        dst_device_data_indices[start : start + max_batch_indices],
                    )
                )
            ret = process_index_batch(
                batch_prefill_blocks,
                batch_dst_blocks,
                batch_device_prefill_blocks,
                batch_device_dst_blocks,
            )
            if ret != 0:
                return ret
        return 0


def send_kvcache(
    self,
    mooncake_session_id: str,
    prefill_kv_indices: npt.NDArray[np.int32],
    dst_kv_ptrs: list[int],
    dst_kv_indices: npt.NDArray[np.int32],
    executor: concurrent.futures.ThreadPoolExecutor,
    dst_layer_ids: Optional[List[int]] = None,
    dst_device_kv_indices: Optional[npt.NDArray[np.int32]] = None,
    dst_kv_item_len: Optional[int] = None,
    dst_attn_tp_size: Optional[int] = None,
    v4_transfer_metadata: Optional[dict] = None,
):
    self._validate_envelope_kv_layout(
        dst_kv_ptrs, dst_kv_item_len, dst_attn_tp_size
    )
    dst_device_kv_ptrs = None
    if dst_device_kv_indices is not None:
        compression_ratios = self.kv_args.mla_compression_ratios
        if compression_ratios is not None:
            if len(dst_kv_ptrs) == len(self.kv_args.kv_data_ptrs):
                start = self.kv_args.prefill_start_layer
                end = self.kv_args.prefill_end_layer
                assert end is not None
                compression_ratios = compression_ratios[start:end]
            host_backed_kv_count = sum(ratio == 4 for ratio in compression_ratios)
        else:
            # The remote target prefix is host-backed, even across PP stages;
            # only the draft suffix uses logical device page IDs.
            num_draft_entries = self.kv_args.num_draft_entries
            if not 0 < num_draft_entries < len(dst_kv_ptrs):
                raise ValueError(
                    "HiSparse spec PD transfer requires an appended device-only "
                    "draft KV buffer"
                )
            host_backed_kv_count = len(dst_kv_ptrs) - num_draft_entries
        dst_device_kv_ptrs = set(dst_kv_ptrs[host_backed_kv_count:])
    return self._send_kvcache_generic(
        mooncake_session_id=mooncake_session_id,
        src_data_ptrs=self.kv_args.kv_data_ptrs,
        dst_data_ptrs=dst_kv_ptrs,
        item_lens=self.kv_args.kv_item_lens,
        prefill_data_indices=prefill_kv_indices,
        dst_data_indices=dst_kv_indices,
        executor=executor,
        force_flat=get_memory().enable_unified_memory,
        src_layer_ids=self.kv_args.kv_layer_ids,
        dst_layer_ids=dst_layer_ids,
        dst_device_data_indices=dst_device_kv_indices,
        dst_device_data_ptrs=dst_device_kv_ptrs,
        src_layout=(
            self.kv_args.v4_transfer_metadata["kv"]
            if self.cp_cache_layer_split
            else None
        ),
        dst_layout=(v4_transfer_metadata or {}).get("kv"),
        dst_item_lens=(v4_transfer_metadata or {}).get("kv_sizes"),
    )


def is_generic_kvcache_state_type(self, st: StateType) -> bool:
    """State types sent via the page-indexed ``_send_kvcache_generic`` path
    (not the mamba-state path); subclasses extend for hardware components."""
    return st in (
        StateType.SWA,
        StateType.DSA,
        StateType.QSA_PENDING,
        StateType.QSA_COMPRESSED,
        StateType.SWA_RING,
        StateType.DSV4_REQUEST_STATE,
        StateType.REQUEST_ROW,
        StateType.BLOCK_SCALE,
        StateType.BLOCK_SCALE_SWA,
    )


def requires_exact_state_index_match(self, st: StateType) -> bool:
    """State types whose page lists are positional and must not be truncated."""
    return st in (
        StateType.QSA_PENDING,
        StateType.QSA_COMPRESSED,
        StateType.SWA_RING,
        StateType.DSV4_REQUEST_STATE,
        StateType.REQUEST_ROW,
    )


def maybe_send_extra(
    self,
    req: TransferInfo,
    prefill_state_indices: List,
    executor: concurrent.futures.ThreadPoolExecutor,
    target_rank_registration_info: Optional[KVArgsRegisterInfo] = None,
):
    rc = 0
    state_types = getattr(self.kv_args, "state_types", [])
    for i, st in enumerate(state_types):
        indices = (
            prefill_state_indices[i] if i < len(prefill_state_indices) else None
        )
        if indices is None:
            continue
        src_data_ptrs = self.kv_args.state_data_ptrs[i]
        src_item_lens = self.kv_args.state_item_lens[i]
        src_dim_per_tensor = (
            self.kv_args.state_dim_per_tensor[i]
            if i < len(self.kv_args.state_dim_per_tensor)
            else []
        )
        src_conv_shard_groups = getattr(self.kv_args, "state_conv_shard_groups", [])
        src_conv_shard_groups = (
            src_conv_shard_groups[i] if i < len(src_conv_shard_groups) else []
        )
        src_slice_outer_counts = getattr(
            self.kv_args, "state_slice_outer_counts", []
        )
        src_slice_outer_counts = (
            src_slice_outer_counts[i] if i < len(src_slice_outer_counts) else []
        )
        src_state_layer_ids = self.kv_args.state_layer_ids
        src_state_layer_ids = (
            src_state_layer_ids[i] if i < len(src_state_layer_ids) else []
        )
        dst_component_index = i
        if target_rank_registration_info is not None:
            dst_component_index = resolve_state_component_dst_index(
                state_types,
                target_rank_registration_info.dst_state_types,
                i,
            )
            dst_data_ptrs = (
                target_rank_registration_info.dst_state_data_ptrs[
                    dst_component_index
                ]
                if dst_component_index
                < len(target_rank_registration_info.dst_state_data_ptrs)
                else []
            )
            dst_item_lens = (
                target_rank_registration_info.dst_state_item_lens[
                    dst_component_index
                ]
                if dst_component_index
                < len(target_rank_registration_info.dst_state_item_lens)
                else []
            )
            dst_dim_per_tensor = (
                target_rank_registration_info.dst_state_dim_per_tensor[
                    dst_component_index
                ]
                if dst_component_index
                < len(target_rank_registration_info.dst_state_dim_per_tensor)
                else []
            )
            dst_state_layer_ids = (
                target_rank_registration_info.dst_state_layer_ids[
                    dst_component_index
                ]
                if dst_component_index
                < len(target_rank_registration_info.dst_state_layer_ids)
                else []
            )
        else:
            dst_data_ptrs, dst_item_lens, dst_dim_per_tensor = [], [], []
            dst_state_layer_ids = []
        dst_indices = (
            req.dst_state_indices[dst_component_index]
            if dst_component_index < len(req.dst_state_indices)
            else []
        )

        if st == StateType.MAMBA:
            if (not src_dim_per_tensor or not dst_dim_per_tensor) and list(
                src_item_lens
            ) != list(dst_item_lens):
                raise RuntimeError(
                    "Mamba state layouts differ between prefill and decode "
                    f"(src item_lens={src_item_lens}, dst item_lens="
                    f"{dst_item_lens}) and no per-tensor dim metadata is "
                    "available to reslice. With --enable-unified-memory, "
                    "prefill and decode must both enable it and use equal "
                    "attention TP sizes."
                )
            if (
                getattr(self.kv_args, "hcu_kda_state_sharded", False)
                and target_rank_registration_info is not None
            ):
                rc = (
                    self._send_hcu_kda_state(
                        req,
                        indices,
                        src_data_ptrs,
                        src_item_lens,
                        src_dim_per_tensor,
                        dst_data_ptrs,
                        dst_indices,
                        dst_item_lens,
                        dst_dim_per_tensor,
                        target_rank_registration_info,
                        src_conv_shard_groups,
                        src_slice_outer_counts,
                        src_state_layer_ids,
                        dst_state_layer_ids,
                    )
                    or rc
                )
            elif (
                target_rank_registration_info is not None
                and self.state_tp_size
                != target_rank_registration_info.dst_attn_tp_size
            ):
                rc = (
                    self._send_mamba_state_slice(
                        req,
                        indices,
                        src_data_ptrs,
                        src_item_lens,
                        src_dim_per_tensor,
                        dst_data_ptrs,
                        dst_indices,
                        dst_item_lens,
                        dst_dim_per_tensor,
                        target_rank_registration_info.dst_tp_rank,
                        target_rank_registration_info.dst_attn_tp_size,
                        src_conv_shard_groups,
                        src_slice_outer_counts,
                        src_state_layer_ids,
                        dst_state_layer_ids,
                    )
                    or rc
                )
            else:
                rc = (
                    self._send_mamba_state(
                        req,
                        indices,
                        src_data_ptrs,
                        src_item_lens,
                        dst_data_ptrs,
                        dst_indices,
                        src_state_layer_ids,
                        dst_state_layer_ids,
                        dst_item_lens,
                    )
                    or rc
                )
        elif st == StateType.DSA_TAIL:
            rc = (
                self._send_slot_state(
                    req,
                    src_data_ptrs,
                    src_item_lens,
                    dst_data_ptrs,
                    dst_item_lens,
                    list(indices),
                    list(dst_indices),
                    st.value,
                    src_state_layer_ids,
                    dst_state_layer_ids,
                )
                or rc
            )
        elif self._is_generic_kvcache_state_type(st):
            is_qwen4_qsa_state = st in (
                StateType.QSA_PENDING,
                StateType.QSA_COMPRESSED,
            )
            has_heterogeneous_attn_tp = (
                target_rank_registration_info is not None
                and self.attn_tp_size
                != target_rank_registration_info.dst_attn_tp_size
            )
            slice_kv_heads = (
                has_heterogeneous_attn_tp
                and not self.is_mla_backend
                and not self.is_hybrid_mla_backend
                and not is_qwen4_qsa_state
            )
            if slice_kv_heads and st != StateType.SWA:
                raise RuntimeError(
                    f"PD Disaggregation does NOT support PD different TP sizes for non-MLA {st.upper()} hybrid models yet."
                )
            if is_qwen4_qsa_state and target_rank_registration_info is not None:
                # Reject QSA state layout mismatches up front.
                if len(dst_item_lens) != len(dst_data_ptrs):
                    raise RuntimeError(
                        f"Replicated {st.upper()} destination pointer/item-length "
                        "metadata is inconsistent: "
                        f"dst ptrs={len(dst_data_ptrs)} lens={len(dst_item_lens)}"
                    )
                qsa_entry_pairs = build_transfer_entry_pairs(
                    src_state_layer_ids,
                    dst_state_layer_ids,
                    len(src_data_ptrs),
                    len(dst_data_ptrs),
                    allow_positional_fallback=self.pp_size == 1,
                )
                layout_mismatches = [
                    (i, j, src_item_lens[i], dst_item_lens[j])
                    for i, j in qsa_entry_pairs
                    if src_item_lens[i] != dst_item_lens[j]
                ]
                if layout_mismatches:
                    raise RuntimeError(
                        f"Replicated {st.upper()} layout differs between mapped "
                        "prefill and decode entries: "
                        f"{layout_mismatches}"
                    )
            if has_heterogeneous_attn_tp and is_qwen4_qsa_state:
                local_tp_rank_in_group = (
                    self.kv_args.engine_rank % self.attn_tp_size
                )
                if not should_send_replicated_state(
                    src_attn_tp_size=self.attn_tp_size,
                    dst_attn_tp_size=(
                        target_rank_registration_info.dst_attn_tp_size
                    ),
                    local_tp_rank_in_group=local_tp_rank_in_group,
                ):
                    continue
            src_indices = list(indices)
            dst_indices_local = list(dst_indices)
            if (
                st == StateType.DSV4_REQUEST_STATE
                and len(src_indices) == 0
                and len(dst_indices_local) == 0
            ):
                continue
            if len(src_indices) != len(dst_indices_local):
                # These components are position- or request-indexed:
                # truncating silently misaligns rows and corrupts KV.
                # Paged SWA/DSA tolerate a 1-page drift -> keep the
                # lenient truncation below.
                if self._requires_exact_state_index_match(st):
                    raise RuntimeError(
                        f"{st.upper()} state index length mismatch: "
                        f"prefill={len(src_indices)}, dst={len(dst_indices_local)}"
                    )
                logger.warning(
                    f"len(prefill_state_indices) = {len(src_indices)}, len(dst_state_indices) = {len(dst_indices_local)}"
                )
                if len(src_indices) > len(dst_indices_local):
                    src_indices = src_indices[: len(dst_indices_local)]
                else:
                    dst_indices_local = dst_indices_local[: len(src_indices)]
            if slice_kv_heads:
                rc = (
                    self._send_swa_state_slice(
                        req=req,
                        prefill_swa_page_indices=src_indices,
                        src_state_data_ptrs=src_data_ptrs,
                        src_state_item_lens=src_item_lens,
                        dst_state_data_ptrs=dst_data_ptrs,
                        dst_swa_page_indices=dst_indices_local,
                        dst_state_item_lens=dst_item_lens,
                        dst_tp_rank=target_rank_registration_info.dst_tp_rank,
                        dst_attn_tp_size=(
                            target_rank_registration_info.dst_attn_tp_size
                        ),
                        executor=executor,
                    )
                    or rc
                )
                continue
            rc = (
                self._send_kvcache_generic(
                    mooncake_session_id=req.mooncake_session_id,
                    src_data_ptrs=src_data_ptrs,
                    dst_data_ptrs=dst_data_ptrs,
                    item_lens=src_item_lens,
                    prefill_data_indices=np.array(src_indices, dtype=np.int32),
                    dst_data_indices=np.array(dst_indices_local, dtype=np.int32),
                    executor=executor,
                    state_type=st,
                    # Two independent reasons to keep the flat layout.
                    # QSA's per-layer list must not be half-split into K/V;
                    # neither must a unified sub-pool's single region, which
                    # holds every layer's K and V per slot envelope -- the
                    # MHA branch would compute zero layers and ship nothing
                    # (same reason as in `send_kvcache`).
                    force_flat=(
                        st in (StateType.QSA_PENDING, StateType.QSA_COMPRESSED)
                        or get_memory().enable_unified_memory
                    ),
                    src_layer_ids=src_state_layer_ids,
                    dst_layer_ids=dst_state_layer_ids,
                    src_layout=(
                        self.kv_args.v4_transfer_metadata["states"][i]
                        if self.cp_cache_layer_split
                        and self.kv_args.v4_transfer_metadata["states"][i]
                        else None
                    ),
                    dst_layout=(
                        target_rank_registration_info.v4_transfer_metadata.get(
                            "states", []
                        )[dst_component_index]
                        if self.cp_cache_layer_split
                        and target_rank_registration_info is not None
                        and dst_component_index < len(
                            target_rank_registration_info.v4_transfer_metadata.get(
                                "states", []
                            )
                        )
                        else None
                    ),
                    dst_item_lens=dst_item_lens,
                    bootstrap_room=req.room,
                )
                or rc
            )
            if self.request_status.get(req.room, KVPoll.Failed) == KVPoll.Failed:
                return rc
        elif st in (StateType.MINIMAX_INDEX_K, StateType.MINIMAX_DENSE_KV):
            # Compacted layer lists require equal TP and PP=1 on both peers.
            if self.pp_size is not None and self.pp_size > 1:
                raise RuntimeError(
                    "PD disagg: PP>1 not supported for MiniMax state yet."
                )
            if (
                target_rank_registration_info is not None
                and self.attn_tp_size
                != target_rank_registration_info.dst_attn_tp_size
            ):
                raise RuntimeError(
                    "PD disagg: heterogeneous TP not supported for MiniMax "
                    "state yet."
                )
            src_indices = list(indices)
            dst_indices_local = list(dst_indices)
            if st == StateType.MINIMAX_DENSE_KV:
                if len(src_indices) != len(dst_indices_local):
                    raise RuntimeError(
                        f"{st.value} state index length mismatch: "
                        f"prefill={len(src_indices)}, dst={len(dst_indices_local)}"
                    )
            elif len(src_indices) > len(dst_indices_local):
                src_indices = src_indices[: len(dst_indices_local)]
            elif len(src_indices) < len(dst_indices_local):
                dst_indices_local = dst_indices_local[: len(src_indices)]
            rc = (
                self._send_kvcache_generic(
                    mooncake_session_id=req.mooncake_session_id,
                    src_data_ptrs=src_data_ptrs,
                    dst_data_ptrs=dst_data_ptrs,
                    item_lens=src_item_lens,
                    prefill_data_indices=np.array(src_indices, dtype=np.int32),
                    dst_data_indices=np.array(dst_indices_local, dtype=np.int32),
                    executor=executor,
                    force_flat=True,
                )
                or rc
            )
    return rc


def send_slot_state(
    self,
    req: TransferInfo,
    src_ptrs: list[int],
    src_item_lens: list[int],
    dst_ptrs: list[int],
    dst_item_lens: list[int],
    src_indices: list[int],
    dst_indices: list[int],
    label: str,
    src_layer_ids: Optional[List[int]] = None,
    dst_layer_ids: Optional[List[int]] = None,
) -> int:
    try:
        if src_layer_ids or dst_layer_ids:
            pairs = build_transfer_entry_pairs(
                src_layer_ids or [],
                dst_layer_ids or [],
                len(src_ptrs),
                len(dst_ptrs),
                allow_positional_fallback=False,
            )
            src_ptrs = [src_ptrs[i] for i, _ in pairs]
            src_item_lens = [src_item_lens[i] for i, _ in pairs]
            dst_ptrs = [dst_ptrs[j] for _, j in pairs]
            dst_item_lens = [dst_item_lens[j] for _, j in pairs]
        else:
            dst_ptrs = slice_dsa_tail_dst_ptrs_for_pp(
                src_ptrs,
                dst_ptrs,
                self.kv_args.prefill_start_layer,
                self.kv_args.prefill_end_layer,
            )
            dst_item_lens = slice_dsa_tail_dst_ptrs_for_pp(
                src_ptrs,
                dst_item_lens,
                self.kv_args.prefill_start_layer,
                self.kv_args.prefill_end_layer,
            )
        transfer_blocks = build_dsa_tail_transfer_blocks(
            src_ptrs,
            src_item_lens,
            dst_ptrs,
            src_indices,
            dst_indices,
            dst_item_lens,
        )
    except ValueError as exc:
        logger.error("%s: %s", label, exc)
        return -1
    return self._transfer_data(req.mooncake_session_id, transfer_blocks)


def await_transfer_futures(self, futures) -> int:
    """Drain every submitted write, including after failure or cancellation."""
    futures = tuple(futures)
    ret = 0
    first_exception = None
    for future in concurrent.futures.as_completed(futures):
        try:
            status = future.result()
        except concurrent.futures.CancelledError:
            continue
        except Exception as exc:
            if first_exception is None:
                first_exception = exc
            for other in futures:
                other.cancel()
            continue
        if status != 0 and ret == 0:
            ret = status
            for other in futures:
                other.cancel()
    if first_exception is not None:
        raise first_exception
    return ret


def register_kv_args(self) -> bool:
    for bootstrap_info in self.bootstrap_infos:
        packed_kv_data_ptrs = b"".join(
            struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.kv_data_ptrs
        )
        packed_aux_data_ptrs = b"".join(
            struct.pack("Q", ptr) for ptr in self.kv_mgr.kv_args.aux_data_ptrs
        )
        packed_state_data_ptrs = pack_int_lists(
            self.kv_mgr.kv_args.state_data_ptrs, "Q"
        )
        packed_state_item_lens = pack_int_lists(
            self.kv_mgr.kv_args.state_item_lens, "I"
        )
        packed_state_dim_per_tensor = pack_int_lists(
            getattr(self.kv_mgr.kv_args, "state_dim_per_tensor", []) or [], "I"
        )
        packed_state_data_formats = pack_string_list(
            getattr(self.kv_mgr.kv_args, "state_data_formats", []) or []
        )
        packed_state_layer_ids = pack_int_lists(
            self.kv_mgr.kv_args.state_layer_ids, "I"
        )
        packed_state_types = pack_state_types(self.kv_mgr.kv_args.state_types)
        packed_kv_layer_ids = b"".join(
            struct.pack("I", layer_id)
            for layer_id in self.kv_mgr.kv_args.kv_layer_ids
        )
        # Note(shangming): No need to add pp rank here since decode pp size should be equal to prefill pp size or 1
        tp_rank = self.kv_mgr.kv_args.engine_rank
        # Some pools have no full-token contiguous KV (kv_item_lens empty)
        # and ship per-pool instead, so report 0.
        kv_item_len = (
            self.kv_mgr.kv_args.kv_item_lens[0]
            if self.kv_mgr.kv_args.kv_item_lens
            else 0
        )
        dst_tp_rank = str(tp_rank).encode("ascii")
        dst_attn_tp_size = str(self.kv_mgr.attn_tp_size).encode("ascii")
        dst_kv_item_len = str(kv_item_len).encode("ascii")
        dst_dcp_size = str(self.kv_mgr.dcp_size).encode("ascii")
        dst_dcp_rank = str(self.kv_mgr.dcp_rank).encode("ascii")
        if (
            self.kv_mgr.enable_staging
            and self.kv_mgr._staging_ctx.allocator is not None
        ):
            _alloc = self.kv_mgr._staging_ctx.allocator
            packed_staging_base_ptr = struct.pack("Q", _alloc.get_base_ptr())
            staging_total_size_str = str(_alloc.get_total_size()).encode("ascii")
        else:
            packed_staging_base_ptr = b""
            staging_total_size_str = b""
        staging_slots = getattr(self.kv_mgr, "kv_buffer_tensors", None) or {}
        packed_staging_slot_layer_ids = b"".join(
            struct.pack("Q", layer_id)
            for layer_id in (staging_slots.get("slot_layer_ids") or [])
        )

        try:
            sock, lock = self._connect_to_bootstrap_server(bootstrap_info)
            with lock:
                sock.send_multipart(
                    [
                        "None".encode("ascii"),
                        self.kv_mgr.local_ip.encode("ascii"),
                        str(self.kv_mgr.rank_port).encode("ascii"),
                        self.session_id.encode("ascii"),
                        packed_kv_data_ptrs,
                        packed_aux_data_ptrs,
                        packed_state_data_ptrs,
                        dst_tp_rank,
                        dst_attn_tp_size,
                        dst_kv_item_len,
                        packed_state_item_lens,
                        packed_state_dim_per_tensor,
                        packed_kv_layer_ids,
                        packed_state_layer_ids,
                        packed_staging_base_ptr,
                        staging_total_size_str,
                        dst_dcp_size,
                        dst_dcp_rank,
                        packed_staging_slot_layer_ids,
                        struct.pack(
                            f"{len(self.kv_mgr.kv_args.kv_item_lens)}Q",
                            *self.kv_mgr.kv_args.kv_item_lens,
                        ),
                        packed_state_data_formats,
                        packed_state_types,
                        encode_v4_transfer_metadata(self.kv_mgr.kv_args),
                    ]
                )
        except zmq.ZMQError:
            self.kv_mgr.record_failure(
                self.bootstrap_room,
                f"_register_kv_args to prefill {bootstrap_info.get('rank_ip')}:{bootstrap_info.get('rank_port')} failed",
            )
            self.conclude_state = KVPoll.Failed
            self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Failed)
            return False
    return True




def send_hcu_kda_state(
    self,
    req,
    indices,
    src_ptrs,
    src_items,
    src_dims,
    dst_ptrs,
    dst_indices,
    dst_items,
    dst_dims,
    info,
    conv_groups,
    outer_counts,
    src_ids,
    dst_ids,
):
    src_size, src_rank = (
        (self.attn_cp_size, self.attn_cp_rank)
        if self.attn_cp_size > 1
        else (self.attn_tp_size, self.attn_tp_rank)
    )
    dst_size = info.dst_attn_tp_size
    dst_rank = info.dst_tp_rank % dst_size
    overlaps = (
        (src_rank * dst_size // src_size == dst_rank)
        if src_size >= dst_size
        else (dst_rank * src_size // dst_size == src_rank)
    )
    if not overlaps:
        return 0
    if src_size == dst_size:
        return self._send_mamba_state(
            req,
            indices,
            src_ptrs,
            src_items,
            dst_ptrs,
            dst_indices,
            src_ids,
            dst_ids,
        )
    return self._send_mamba_state_slice(
        req,
        indices,
        src_ptrs,
        src_items,
        src_dims,
        dst_ptrs,
        dst_indices,
        dst_items,
        dst_dims,
        info.dst_tp_rank,
        dst_size,
        conv_groups,
        outer_counts,
        src_ids,
        dst_ids,
        src_shard_size=src_size,
        src_shard_rank=src_rank,
    )



def send_mamba_state_slice(
    self,
    req: TransferInfo,
    prefill_mamba_index: list,
    src_state_data_ptrs: list[int],
    src_state_item_lens: list[int],
    src_state_dim_per_tensor: list[int],
    dst_state_data_ptrs: list[int],
    dst_mamba_index: list,
    dst_state_item_lens: list[int],
    dst_state_dim_per_tensor: list[int],
    dst_tp_rank: int,
    dst_attn_tp_size: int,
    src_state_conv_shard_groups: list = None,
    src_state_slice_outer_counts: list[int] = None,
    src_layer_ids: Optional[List[int]] = None,
    dst_layer_ids: Optional[List[int]] = None,
    *,
    src_shard_size: Optional[int] = None,
    src_shard_rank: Optional[int] = None,
):
    """Transfer Mamba states with TP slice support.

    Mamba state layout:
    - conv_state: [num_layers, size+1, conv_dim/tp, conv_kernel-1]
    - temporal_state: [num_layers, size+1, num_heads/tp, head_dim, state_size]

    The 3rd dimension is sliced by TP. When prefill and decode have different
    attn_tp_size, we slice the state accordingly. GDN conv_state is the
    concatenation [query | key | value] with each sub-block head-sharded
    independently, so on the scatter path it is sliced per sub-block via
    ``src_state_conv_shard_groups`` (see
    compute_mamba_state_slice_byte_blocks).
    """
    logger.warning_once(
        "Using Mamba state slice transfer for different runtime attention TP "
        f"sizes: prefill={self.state_tp_size}, decode={dst_attn_tp_size}. "
        "Performance may be affected."
    )
    assert len(prefill_mamba_index) == 1, "Mamba should have single state index"

    # If no dimension info available, fall back to regular transfer
    if not src_state_dim_per_tensor or not dst_state_dim_per_tensor:
        return self._send_mamba_state(
            req,
            prefill_mamba_index,
            src_state_data_ptrs,
            src_state_item_lens,
            dst_state_data_ptrs,
            dst_mamba_index,
            src_layer_ids,
            dst_layer_ids,
            dst_state_item_lens,
        )

    source_size = self.state_tp_size if src_shard_size is None else src_shard_size
    local_tp_rank_in_group = (
        self.kv_args.engine_rank % source_size
        if src_shard_rank is None
        else src_shard_rank
    )
    dst_tp_rank_in_group = dst_tp_rank % dst_attn_tp_size

    transfer_blocks = []
    pairs = build_transfer_entry_pairs(
        src_layer_ids or [],
        dst_layer_ids or [],
        len(src_state_data_ptrs),
        len(dst_state_data_ptrs),
        allow_positional_fallback=self.pp_size == 1,
    )
    for i, j in pairs:
        dst_state_ptr = dst_state_data_ptrs[j]
        src_item_len = src_state_item_lens[i]
        dst_item_len = dst_state_item_lens[j]
        src_dim = src_state_dim_per_tensor[i]
        dst_dim = dst_state_dim_per_tensor[j]

        conv_shard_groups = (
            src_state_conv_shard_groups[i]
            if src_state_conv_shard_groups and i < len(src_state_conv_shard_groups)
            else None
        )
        outer_count = (
            src_state_slice_outer_counts[i]
            if src_state_slice_outer_counts
            and i < len(src_state_slice_outer_counts)
            else 1
        )
        for (
            src_offset,
            dst_offset,
            bytes_to_send,
        ) in compute_mamba_state_slice_byte_blocks(
            src_item_len=src_item_len,
            dst_item_len=dst_item_len,
            src_dim=src_dim,
            dst_dim=dst_dim,
            outer_count=outer_count,
            src_attn_tp_size=source_size,
            dst_attn_tp_size=dst_attn_tp_size,
            dst_tp_rank_in_group=dst_tp_rank_in_group,
            local_tp_rank_in_group=local_tp_rank_in_group,
            conv_shard_groups=conv_shard_groups,
        ):
            src_addr = (
                src_state_data_ptrs[i]
                + src_item_len * int(prefill_mamba_index[0])
                + src_offset
            )
            dst_addr = (
                dst_state_ptr + dst_item_len * int(dst_mamba_index[0]) + dst_offset
            )
            transfer_blocks.append((src_addr, dst_addr, bytes_to_send))

    return self._transfer_data(req.mooncake_session_id, transfer_blocks)



def should_skip_cp_replicated_state_transfer(self):
    if getattr(self.kv_args, "hcu_kda_state_sharded", False):
        return False
    return super(MooncakeKVManager, self)._should_skip_cp_replicated_state_transfer()

