# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
"""Current HCU PD wire types and optional append-only metadata codecs."""
from __future__ import annotations
import dataclasses
import enum
import json
import struct
from typing import List, Optional
import numpy as np
import numpy.typing as npt
from sglang.srt.disaggregation.common.staging_handler import StagingRegisterInfo, StagingTransferInfo
from sglang.srt.disaggregation.common.utils import pack_list_of_buffers, unpack_list_of_buffers, pack_int_lists, unpack_int_lists
from hcu_sglang.mem_cache.cp_cache_layer_split.transfer import decode_v4_transfer_metadata, encode_v4_transfer_metadata

class StateType(str, enum.Enum):
    MAMBA = "mamba"
    QSA_PENDING = "qsa_pending"
    QSA_COMPRESSED = "qsa_compressed"
    SWA = "swa"
    DSA = "dsa"
    # DSA kpool-compress tail: one per-request ring row. The indices encode
    # only the live subrange of that row for the current open pool.
    DSA_TAIL = "dsa_tail"
    MINIMAX_INDEX_K = "minimax_index_k"
    MINIMAX_DENSE_KV = "minimax_dense_kv"
    # DeepSeek-V4 unified_kv SWA ring: addressed per-row by ring slot
    # (req_pool_idx * ring_stride + pos % ring_stride), needs its own component.
    SWA_RING = "swa_ring"
    # DeepSeek-V4 request-scoped compression state; preserve the legacy wire value.
    DSV4_REQUEST_STATE = "c128_state"
    # One item per request row (req_pool_idx), sent row to row: the DeepSeek-V4
    # DSpark draft's SWA valid-from.
    REQUEST_ROW = "request_row"
    # A block-scaled KV dtype keeps its per-block scales in buffers parallel to
    # K/V, one component per sub-pool so each carries the index payload of the
    # KV it describes (whole sequence for full attention, window for SWA).
    BLOCK_SCALE = "block_scale"
    BLOCK_SCALE_SWA = "block_scale_swa"
    # Target aux hidden rows used to bootstrap decode-side draft KV.
    PD_HIDDEN = "pd_hidden"


@dataclasses.dataclass
class PrefillServerInfo:
    # Topology fields (fetched from bootstrap server)
    attn_tp_size: int
    attn_cp_size: int
    dp_size: int
    pp_size: int
    page_size: Optional[int]
    kv_cache_dtype: Optional[str]
    follow_bootstrap_room: bool
    enable_dsa_cache_layer_split: bool = False
    cp_cache_layer_split: bool = False
    dsv41_spec_layout: Optional[dict] = None
    kv_cache_layout: Optional[str] = None
    kv_layout: Optional[str] = None
    decode_allocation_policy: str = "early"

    # PD true-retraction rebootstrap: the prefill's HTTP API port. The decode
    # already knows the prefill host (the bootstrap_addr host), so it can POST
    # /generate to http://{bootstrap_host}:{prefill_http_port} to trigger a KV
    # recompute -- no router-injected pd_rebootstrap_prefill_url needed.
    prefill_http_port: Optional[int] = None

    # Pre-computed rank mapping (set by try_ensure_parallel_info on decode side)
    target_tp_rank: Optional[int] = None
    target_tp_ranks: Optional[List[int]] = None
    target_cp_ranks: Optional[List[int]] = None
    target_pp_ranks: Optional[List[int]] = None
    required_dst_info_num: Optional[int] = None
    required_prefill_response_num: Optional[int] = None

    def __post_init__(self):
        self.attn_tp_size = int(self.attn_tp_size)
        self.attn_cp_size = int(self.attn_cp_size)
        self.dp_size = int(self.dp_size)
        self.pp_size = int(self.pp_size)
        self.page_size = int(self.page_size) if self.page_size is not None else None
        self.kv_cache_dtype = (
            str(self.kv_cache_dtype) if self.kv_cache_dtype is not None else None
        )
        self.kv_cache_layout = (
            str(self.kv_cache_layout) if self.kv_cache_layout is not None else None
        )
        self.follow_bootstrap_room = bool(self.follow_bootstrap_room)
        self.enable_dsa_cache_layer_split = bool(self.enable_dsa_cache_layer_split)
        self.cp_cache_layer_split = bool(self.cp_cache_layer_split)
        self.prefill_http_port = (
            int(self.prefill_http_port) if self.prefill_http_port is not None else None
        )


def pack_string_list(values: List[str]) -> bytes:
    return pack_list_of_buffers([value.encode("utf-8") for value in values])


def unpack_string_list(buf: bytes) -> List[str]:
    return [value.decode("utf-8") for value in unpack_list_of_buffers(buf)]


def pack_state_types(state_types) -> bytes:
    return ",".join(state_type.value for state_type in (state_types or [])).encode(
        "ascii"
    )


def unpack_state_types(data: bytes):

    if not data:
        return []
    return [StateType(value) for value in data.decode("ascii").split(",") if value]


def resolve_state_component_dst_index(src_state_types, dst_state_types, src_index: int):
    if not dst_state_types:
        return src_index
    if not src_state_types:
        raise RuntimeError(
            "Destination state_types are present but source state_types are empty."
        )
    if src_index >= len(src_state_types):
        raise RuntimeError(
            f"Source state component index {src_index} exceeds "
            f"state_types length {len(src_state_types)}."
        )
    state_type = src_state_types[src_index]
    occurrence = sum(
        1 for item in src_state_types[: src_index + 1] if item == state_type
    )
    seen = 0
    for dst_index, dst_state_type in enumerate(dst_state_types):
        if dst_state_type == state_type:
            seen += 1
            if seen == occurrence:
                return dst_index
    raise RuntimeError(
        f"Decode peer is missing state component {state_type!s} "
        f"occurrence {occurrence}."
    )


@dataclasses.dataclass
class TransferInfo:
    room: int
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_indices: npt.NDArray[np.int32]
    dst_aux_index: int
    dst_state_indices: List[List[int]]  # parallel to receiver's state_types
    required_dst_info_num: int
    is_dummy: bool
    decode_prefix_len: Optional[int] = None
    dst_device_kv_indices: Optional[npt.NDArray[np.int32]] = None
    spec_metadata: Optional[dict] = None
    # Note: always put the optional staging field at the final (it will be set through 'STAGING_RSP' pkg when needed)
    staging: Optional[StagingTransferInfo] = None

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        if msg[4] == b"" and msg[5] == b"":
            is_dummy = True
            dst_kv_indices = np.array([], dtype=np.int32)
            dst_aux_index = None
            dst_state_indices = []
        else:
            dst_kv_indices = np.frombuffer(msg[4], dtype=np.int32)
            dst_aux_index = int(msg[5].decode("ascii"))
            dst_state_indices = unpack_int_lists(msg[6], "i")
            is_dummy = False
        return cls(
            room=int(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_indices=dst_kv_indices,
            dst_aux_index=dst_aux_index,
            dst_state_indices=dst_state_indices,
            required_dst_info_num=int(msg[7].decode("ascii")),
            is_dummy=is_dummy,
            decode_prefix_len=(
                int(msg[8].decode("ascii")) if len(msg) > 8 and msg[8] != b"" else None
            ),
            dst_device_kv_indices=(
                np.frombuffer(msg[9], dtype=np.int32)
                if len(msg) > 9 and msg[9] != b""
                else None
            ),
            spec_metadata=(
                json.loads(msg[10].decode("utf-8"))
                if len(msg) > 10 and msg[10] != b""
                else None
            ),
        )


@dataclasses.dataclass
class KVArgsRegisterInfo:
    room: str
    endpoint: str
    dst_port: int
    mooncake_session_id: str
    dst_kv_ptrs: list[int]
    dst_aux_ptrs: list[int]
    dst_state_data_ptrs: List[List[int]]  # parallel to state_types (same below)
    dst_tp_rank: int
    dst_attn_tp_size: int
    dst_kv_item_len: int
    # for mamba state different tp slice transfer
    dst_state_item_lens: List[List[int]]
    dst_state_dim_per_tensor: List[List[int]]
    dst_kv_layer_ids: List[int]
    dst_state_layer_ids: List[List[int]]
    dst_state_types: List[StateType] = dataclasses.field(default_factory=list)
    dst_state_data_formats: List[str] = dataclasses.field(default_factory=list)
    # Local-only validation result; this is never serialized on the wire.
    registration_error: Optional[str] = dataclasses.field(default=None, repr=False)
    dst_dcp_size: int = 1
    dst_dcp_rank: int = 0
    requires_dcp_relayout: bool = False
    dcp_token_item_lens: Optional[List[int]] = None
    dst_kv_item_lens: List[int] = dataclasses.field(default_factory=list)
    staging_base_ptr: int = 0
    staging_total_size: int = 0
    staging: Optional[StagingRegisterInfo] = None
    v4_transfer_metadata: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def from_zmq(cls, msg: List[bytes]):
        return cls(
            room=str(msg[0].decode("ascii")),
            endpoint=msg[1].decode("ascii"),
            dst_port=int(msg[2].decode("ascii")),
            mooncake_session_id=msg[3].decode("ascii"),
            dst_kv_ptrs=list(struct.unpack(f"{len(msg[4]) // 8}Q", msg[4])),
            dst_aux_ptrs=list(struct.unpack(f"{len(msg[5]) // 8}Q", msg[5])),
            dst_state_data_ptrs=unpack_int_lists(msg[6], "Q"),
            dst_tp_rank=int(msg[7].decode("ascii")),
            dst_attn_tp_size=int(msg[8].decode("ascii")),
            dst_kv_item_len=int(msg[9].decode("ascii")),
            dst_state_item_lens=(
                unpack_int_lists(msg[10], "I") if len(msg) > 10 else []
            ),
            dst_state_dim_per_tensor=(
                unpack_int_lists(msg[11], "I") if len(msg) > 11 else []
            ),
            dst_kv_layer_ids=(
                list(struct.unpack(f"{len(msg[12]) // 4}I", msg[12]))
                if len(msg) > 12 and msg[12] != b""
                else []
            ),
            dst_state_layer_ids=(
                unpack_int_lists(msg[13], "I")
                if len(msg) > 13 and msg[13] != b""
                else []
            ),
            dst_state_data_formats=(
                unpack_string_list(msg[20]) if len(msg) > 20 and msg[20] != b"" else []
            ),
            # Index 19 carries kv_item_lens; state types follow the data formats.
            dst_state_types=unpack_state_types(msg[21]) if len(msg) > 21 else [],
            staging_base_ptr=(
                struct.unpack("Q", msg[14])[0]
                if len(msg) > 14 and len(msg[14]) == 8
                else 0
            ),
            staging_total_size=(
                int(msg[15].decode("ascii")) if len(msg) > 15 and msg[15] != b"" else 0
            ),
            dst_dcp_size=(
                int(msg[16].decode("ascii")) if len(msg) > 16 and msg[16] != b"" else 1
            ),
            dst_dcp_rank=(
                int(msg[17].decode("ascii")) if len(msg) > 17 and msg[17] != b"" else 0
            ),
            dst_kv_item_lens=(
                list(struct.unpack(f"{len(msg[19]) // 8}Q", msg[19]))
                if len(msg) > 19 and msg[19]
                else []
            ),
            # Note: always put the staging field at the final
            staging=StagingRegisterInfo.from_zmq_fields(msg, 14, slot_ids_index=18),
            v4_transfer_metadata=decode_v4_transfer_metadata(msg),
        )


