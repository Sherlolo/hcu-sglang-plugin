# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
"""HCU common PD topology and bootstrap metadata contracts."""
from __future__ import annotations
import asyncio
import dataclasses
import concurrent.futures
from collections import defaultdict
import threading
from typing import Dict, List, Optional, Set, Tuple, Union
import requests
import zmq
from aiohttp import web
from transformers import PretrainedConfig
from sglang.srt.disaggregation.common.conn import (
    AckTarget, DeferredAbortAckState, DeferredBootstrap, PPConsensusStore,
    PrefillRankInfo, _get_bootstrap_session, get_local_ip_auto,
    get_zmq_socket_on_host, NetworkAddress, logger,
)
from sglang.srt.disaggregation.base.conn import KVArgs, KVPoll
from sglang.srt.disaggregation.utils import DisaggregationMode, get_dsv41_spec_layout
from sglang.srt.configs.model_config import ModelConfig, is_deepseek_dsa, dsa_layer_skips_topk
from sglang.srt.runtime_context import get_disagg, get_parallel, get_serving, linear_attn_tp_size
from sglang.srt.server_args import ServerArgs
from sglang.srt.environ import envs
from .protocol import PrefillServerInfo, StateType

def manager_init(
    self,
    args: KVArgs,
    disaggregation_mode: DisaggregationMode,
    server_args: ServerArgs,
    is_mla_backend: Optional[bool] = False,
):
    self.kv_args = args
    self.kv_cache_layout = getattr(args, "kv_cache_layout", None)
    self.cp_cache_layer_split = bool(getattr(args, "cp_cache_layer_split", False))
    self.kv_cache_dtype_str = args.kv_cache_dtype_str
    self.kv_layout = args.kv_layout
    self.dsv41_spec_layout = get_dsv41_spec_layout(args)
    self.kv_item_lens_sum = sum(args.kv_item_lens)
    self.state_item_lens_sum = sum(x for comp in args.state_item_lens for x in comp)
    self.is_mla_backend = is_mla_backend
    # Per-sender fan-out of a KV copy onto N decode destinations
    # (MLA under Prefill-CP + Decode-TP, or decode_tp > prefill_tp).
    # MLA is resolved lazily at bootstrap (see resolve_kv_replica_factor);
    # MHA never replicates, so it stays pinned at 1.
    self._kv_replica_factor: Optional[int] = None if is_mla_backend else 1
    self.is_hybrid_mla_backend = getattr(args, "is_hybrid_mla_backend", False)
    self.disaggregation_mode = disaggregation_mode
    self.server_args = server_args
    self.enable_deferred_decode_kv_release = (
        envs.SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE.get()
        and self.supports_deferred_decode_kv_release
    )
    self._dcp_pack_buffers = None
    self._dcp_pack_max_tokens: Optional[int] = None
    # for p/d multi node infer
    self.bootstrap_host = get_serving().host
    self.bootstrap_port = get_disagg().disaggregation_bootstrap_port
    self.dist_init_addr = get_parallel().dist_init_addr
    parallel = get_parallel()
    self.attn_tp_size = parallel.attn_tp_size
    self.attn_tp_rank = parallel.attn_tp_rank
    self.attn_cp_size = parallel.attn_cp_size
    self.attn_cp_rank = parallel.attn_cp_rank
    # The TP width the transferred state is laid out over, which decode
    # pairs with: attention TP, or the TP group under CP-TP group sharing,
    # where the GDN heads are partitioned over it and QSA/PLE state is whole
    # on every rank. KV heads keep following attention TP.
    self.state_tp_size = linear_attn_tp_size()
    self.dcp_size = parallel.attn_dcp_size
    self.dcp_rank = parallel.attn_dcp_rank
    self.attn_dp_size = parallel.attn_dp_size
    self.attn_dp_rank = parallel.attn_dp_rank
    self.system_dp_size = get_parallel().dp_size
    self.system_dp_rank = (
        self.kv_args.system_dp_rank if self.kv_args.system_dp_rank else 0
    )
    self.pp_size = get_parallel().pp_size
    self.pp_rank = self.kv_args.pp_rank
    self.local_ip = get_local_ip_auto()
    cp_sharded_prefill = self.attn_cp_size > 1 and (
        self.is_hybrid_mla_backend or get_parallel().enable_dsa_cache_layer_split
        or self.cp_cache_layer_split
    )

    hybrid_decode_pulls_all_ranks = (
        self.is_hybrid_mla_backend
        and disaggregation_mode == DisaggregationMode.DECODE
    )
    # CP-TP group sharing: CP rank r holds every KV head, the CP-replicated
    # state and linear-attention head shard r. It transfers as TP rank r of
    # a CP-1 prefill (see register_to_bootstrap), not as a CP replica.
    cp_tp_group_sharing = parallel.enable_cp_tp_group_sharing
    self.enable_all_cp_ranks_for_transfer = not cp_tp_group_sharing and (
        envs.SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER.get()
        or cp_sharded_prefill
        or hybrid_decode_pulls_all_ranks
    )
    self.model_config = ModelConfig.from_server_args(server_args)

    # bind zmq socket
    self._zmq_ctx = zmq.Context()
    # Raise libzmq's per-context socket cap because this manager caches two
    # sockets per decode endpoint, so large fleets can exceed the default.
    self._zmq_ctx.set(
        zmq.MAX_SOCKETS, envs.SGLANG_DISAGGREGATION_ZMQ_MAX_SOCKETS.get()
    )
    self.rank_port, self.server_socket = get_zmq_socket_on_host(
        self._zmq_ctx, zmq.PULL, host=self.local_ip
    )
    logger.debug(f"kv manager bind to {self.local_ip}:{self.rank_port}")

    self.request_status: Union[Dict[int, KVPoll], PPConsensusStore] = {}
    self._socket_cache: Dict[str, zmq.Socket] = {}
    self._monitor_cache: Dict[str, zmq.Socket] = {}
    self._socket_send_locks: Dict[str, threading.Lock] = {}
    self._socket_lock = threading.Lock()
    self.failure_records: Dict[int, str] = {}
    self.failure_lock = threading.Lock()
    self._staging_outstanding: Dict[int, int] = defaultdict(int)
    self._deferred_ack_targets: Dict[int, Dict[Tuple[str, int], AckTarget]] = {}
    # A poisoned room has a permanently outstanding transfer whose backend
    # can no longer prove quiescent. A new room lifecycle clears this state.
    self._deferred_ack_poisoned_rooms: Set[int] = set()
    self.defer_decode_allocation = (
        get_disagg().disaggregation_decode_allocation_policy == "prefill_complete"
    )
    self.deferred_bootstrap = None
    if self.defer_decode_allocation:
        if any(state != StateType.DSA for state in args.state_types):
            raise ValueError(
                "prefill_complete currently supports full-attention and DSA KV pools only"
            )
        if disaggregation_mode == DisaggregationMode.PREFILL:
            self.deferred_bootstrap = DeferredBootstrap(
                envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()
                + envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()
            )

    if self.disaggregation_mode == DisaggregationMode.PREFILL:
        # When SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER is True, all CP ranks
        # participate in KV transfer; Otherwise only CP rank 0 sends.
        self.is_dummy_cp_rank = (
            not self.enable_all_cp_ranks_for_transfer
            and not cp_tp_group_sharing
            and self.attn_cp_size > 1
            and self.attn_cp_rank != 0
        )
        # Sync the leader's bootstrap port to every rank before
        # registering: in multi-node prefill, registration targets
        # `dist_init_addr` (rank 0) but each rank's local port may
        # differ when the launcher auto-reserves a free port per host.
        self.bootstrap_port = self._sync_bootstrap_port_across_nodes(
            self.bootstrap_port
        )
        self.register_to_bootstrap()
        self.transfer_infos = {}
        self.req_to_decode_prefix_len: Dict[int, int] = {}
        self.req_to_pd_hidden_meta: Dict[int, dict] = {}
        self.decode_kv_args_table = {}
        self.pp_group = get_parallel().pp_group
        if self.pp_size > 1:
            self.request_status = PPConsensusStore(
                self.pp_size, self.pp_rank, self.pp_group
            )
        # If a timeout happens on the prefill side, it means prefill instances
        # fail to receive the KV indices from the decode instance of this request.
        # These timeout requests should be aborted to release the tree cache.
        self.bootstrap_timeout = envs.SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT.get()
    elif self.disaggregation_mode == DisaggregationMode.DECODE:
        self.enable_staging: bool = False
        self._staging_handler = None
        self.connection_pool: Dict[str, Dict[str, Union[str, int]]] = {}
        self.connection_lock = threading.Lock()
        self.required_prefill_response_num_table: Dict[int, int] = {}
        self.prefill_info_table: Dict[str, PrefillServerInfo] = {}
        self.heartbeat_failures: Dict[str, int] = {}
        self.session_pool: Dict = defaultdict(requests.Session)
        self.session_pool_lock = threading.Lock()
        self.addr_to_rooms_tracker: Dict[str, Set[int]] = defaultdict(set)
        self.prefill_response_tracker: Dict[int, Set[int]] = defaultdict(set)
        # Deferred KV release: room -> current generation and ranks that
        # ACKed. The generation rejects an old request's late ACK after room
        # reuse.
        self._deferred_abort_ack_tracker: Dict[int, DeferredAbortAckState] = {}
        self._deferred_abort_generation = 0
        # Heartbeat interval should be at least 2 seconds
        self.heartbeat_interval = max(
            envs.SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL.get(), 2.0
        )
        # Heartbeat failure should be at least 1
        self.max_failures = max(
            envs.SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE.get(), 1
        )
        # Event used to signal the heartbeat checker thread to exit
        # during teardown (e.g. runtime P<->D role switch).
        self._heartbeat_shutdown = threading.Event()
        # If a timeout happens on the decode side, it means decode instances
        # fail to receive the KV Cache transfer done signal after bootstrapping.
        # These timeout requests should be aborted to release the tree cache.
        self.waiting_timeout = envs.SGLANG_DISAGGREGATION_WAITING_TIMEOUT.get()
        # PD true-retraction rebootstrap: a shared executor + per-thread HTTP
        # sessions used to drive the original prefill worker's ``/generate``
        # endpoint so it recomputes a retracted request's prefix KV under the
        # current weights. Created lazily on first use so deployments that
        # never retract pay nothing.
        self._prefill_recompute_executor: Optional[
            concurrent.futures.ThreadPoolExecutor
        ] = None
        self._prefill_recompute_executor_lock = threading.Lock()
        self._prefill_recompute_sessions = threading.local()
    else:
        raise ValueError(
            f"Unsupported DisaggregationMode: {self.disaggregation_mode}"
        )


def manager_should_skip_cp_replicated_state_transfer(self) -> bool:
    """Whether this prefill rank should omit CP-replicated state.

    Prefill CP materializes global token order before writing state pools, so
    every CP rank holds the same state. When all CP ranks transfer their KV
    shards, only rank 0 needs to send that state. Cache layer split is the
    exception because each CP rank owns different state layers.
    """
    return (
        self.attn_cp_size > 1
        and self.attn_cp_rank != 0
        and not get_parallel().enable_dsa_cache_layer_split
        and not get_parallel().enable_cp_tp_group_sharing
        and not self.cp_cache_layer_split
    )


def manager_register_to_bootstrap(self):
    """Register prefill server info to bootstrap server via HTTP PUT."""
    if self.dist_init_addr:
        # Multi-node case: bootstrap server's host is dist_init_addr
        host = NetworkAddress.parse(self.dist_init_addr).resolved().host
    else:
        # Single-node case: bootstrap server's host is the same as http server's host
        host = self.bootstrap_host
        # A wildcard bind address (0.0.0.0 / ::) is not a valid HTTP Host
        # and can't be connected to; rewrite it to the same-family loopback,
        # which the wildcard listener also binds.  (self.local_ip is wrong
        # here — it can resolve to a different family than the listener,
        # e.g. IPv6 while the server is bound to 0.0.0.0.)
        host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(host, host)

    bootstrap_na = NetworkAddress(host, self.bootstrap_port)
    url = f"{bootstrap_na.to_url()}/route"
    attn_tp = (self.attn_tp_size, self.attn_tp_rank)
    attn_cp = (self.attn_cp_size, self.attn_cp_rank)
    if get_parallel().enable_cp_tp_group_sharing:
        # Decode pairs its TP rank r with this CP rank r (see __init__).
        attn_tp, attn_cp = attn_cp, (1, 0)
    payload = {
        "attn_tp_size": attn_tp[0],
        "attn_tp_rank": attn_tp[1],
        "attn_cp_size": attn_cp[0],
        "attn_cp_rank": attn_cp[1],
        "attn_dp_size": self.attn_dp_size,
        "attn_dp_rank": self.attn_dp_rank,
        "pp_size": self.pp_size,
        "pp_rank": self.pp_rank,
        "system_dp_size": self.system_dp_size,
        "system_dp_rank": self.system_dp_rank,
        "rank_ip": self.local_ip,
        "rank_port": self.rank_port,
        "page_size": self.kv_args.page_size,
        "kv_cache_layout": self.kv_cache_layout,
        "kv_cache_dtype": self.kv_cache_dtype_str,
        "kv_layout": self.kv_layout,
        "dsv41_spec_layout": self.dsv41_spec_layout,
        "load_balance_method": get_parallel().load_balance_method,
        "enable_dsa_cache_layer_split": get_parallel().enable_dsa_cache_layer_split,
        "cp_cache_layer_split": self.cp_cache_layer_split,
        # Self-register the HTTP API port so the decode can derive the PD
        # retract rebootstrap /generate URL from bootstrap info instead of a
        # router-injected pd_rebootstrap_prefill_url.
        "prefill_http_port": get_serving().port,
    }

    payload["decode_allocation_policy"] = (
        get_disagg().disaggregation_decode_allocation_policy
    )

    if envs.SGLANG_RUST_SERVER.get() and self.attn_dp_size > 1:
        topology_rows = get_parallel().world_group.all_gather_object(payload)
        # Every scheduler contributes a topology row. Only the scheduler
        # ranks that own a Rust listener populate their local registry.
        if self.kv_args.rust_http_port is None:
            return
        registry_host = {"0.0.0.0": "127.0.0.1", "::": "::1"}.get(
            self.bootstrap_host, self.bootstrap_host
        )
        registry = NetworkAddress(registry_host, self.kv_args.rust_http_port)
        url = f"{registry.to_url()}/route"
        for topology_row in topology_rows:
            registry_row = {
                **topology_row,
                "prefill_http_port": self.kv_args.rust_http_port,
            }
            self._register_topology_row(url, registry_row)
        return

    self._register_topology_row(url, payload)


def manager_try_ensure_parallel_info(self, bootstrap_addr: str) -> bool:
    """Single non-blocking attempt to fetch and cache prefill parallel info.
    Returns True if info is available (cached or freshly fetched)."""
    if bootstrap_addr in self.prefill_info_table:
        return True

    info: PrefillServerInfo = None
    try:
        url = (
            f"http://{bootstrap_addr}/route?"
            f"prefill_dp_rank={-1}&prefill_cp_rank={-1}&"
            f"target_tp_rank={-1}&target_pp_rank={-1}"
        )
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            info = PrefillServerInfo(**data)
        else:
            logger.error(
                f"Failed to get prefill server info: {response.status_code}, {response.text}"
            )
            return False
    except Exception as e:
        logger.error(f"Error fetching prefill server info from bootstrap: {e}")
        return False

    # Sanity checks
    if info.page_size is not None and info.page_size != self.kv_args.page_size:
        raise RuntimeError(
            f"Page size mismatch: prefill server has page_size={info.page_size}, "
            f"but decode server has page_size={self.kv_args.page_size}. "
            f"Both servers must use the same --page-size value."
        )

    if (
        info.kv_cache_layout is not None
        and info.kv_cache_layout != self.kv_cache_layout
    ):
        raise RuntimeError(
            f"KV cache layout mismatch: prefill server has kv_cache_layout={info.kv_cache_layout}, "
            f"but decode server has kv_cache_layout={self.kv_cache_layout}."
        )

    if (
        info.kv_cache_dtype is not None
        and info.kv_cache_dtype != self.kv_cache_dtype_str
    ):
        raise RuntimeError(
            f"KV cache dtype mismatch: prefill server has kv_cache_dtype={info.kv_cache_dtype}, "
            f"but decode server has kv_cache_dtype={self.kv_cache_dtype_str}. "
            f"Both servers must use the same --kv-cache-dtype value."
        )

    if (
        info.kv_layout is not None
        and self.kv_layout is not None
        and info.kv_layout != self.kv_layout
    ):
        raise RuntimeError(
            f"KV layout mismatch: prefill server has kv_layout={info.kv_layout}, "
            f"but decode server has kv_layout={self.kv_layout}. Both servers "
            f"must use the same --dsv4-attn-backend and SGLANG_DSV4_KV_LAYOUT."
        )

    local_layout = self.dsv41_spec_layout
    if local_layout is not None or info.dsv41_spec_layout is not None:
        if local_layout != info.dsv41_spec_layout:
            mismatched_fields = sorted(
                key
                for key in (local_layout or {}).keys()
                | (info.dsv41_spec_layout or {}).keys()
                if (local_layout or {}).get(key)
                != (info.dsv41_spec_layout or {}).get(key)
            )
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark PD layout mismatch "
                f"({', '.join(mismatched_fields)}): both servers must "
                "enable DSpark with the same block size and target/draft KV "
                "layout. Upgrade both servers together."
            )
        decode_tp_size_aligns_prefill_world_size = (
            info.attn_cp_size > 1
            and (self.is_mla_backend or self.is_hybrid_mla_backend)
            and self.attn_cp_size == 1
            and info.attn_tp_size * info.attn_cp_size == self.attn_tp_size
        )
        layer_split_mla_layout = (
            info.cp_cache_layer_split
            and info.attn_cp_size > 1
            and self.attn_cp_size == 1
            and info.attn_tp_size == self.attn_tp_size == 1
            and self.dcp_size == 1
            and (self.is_mla_backend or self.is_hybrid_mla_backend)
        )
        if (
            info.attn_tp_size != self.attn_tp_size
            and not decode_tp_size_aligns_prefill_world_size
        ):
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark PD requires the same TP size on both "
                "servers (including prefill CP ranks for MLA)"
            )
        if (
            info.attn_cp_size != 1 or self.attn_cp_size != 1
        ) and not layer_split_mla_layout:
            raise RuntimeError(
                "DeepSeek-V4.1 DSpark PD requires CP=1 on both servers "
                "or descriptor-matched LayerSplit prefill with attention TP=1 "
                "and decode CP=1, DCP=1"
            )

    # Every send path rejects head-sharded MXFP8 block scales across
    # attention TP sizes, so each request would fail at its first transfer.
    if (
        info.attn_tp_size != self.attn_tp_size
        and not self.is_mla_backend
        and not self.is_hybrid_mla_backend
    ):
        block_scale_states = [
            st
            for st in getattr(self.kv_args, "state_types", None) or []
            if st in (StateType.BLOCK_SCALE, StateType.BLOCK_SCALE_SWA)
        ]
        if block_scale_states:
            raise RuntimeError(
                "PD Disaggregation does NOT support different attention TP "
                "sizes with MXFP8 KV cache: prefill attn_tp_size="
                f"{info.attn_tp_size}, decode attn_tp_size={self.attn_tp_size}, "
                f"state types {[st.value for st in block_scale_states]}."
            )

    if self.dcp_size > 1:
        if not (self.is_mla_backend or self.is_hybrid_mla_backend):
            raise RuntimeError(
                "PD decode DCP requires an MLA or hybrid-MLA KV pool."
            )
        if info.attn_cp_size != 1:
            raise RuntimeError(
                "PD decode DCP currently requires prefill attention CP=1, "
                f"got {info.attn_cp_size}."
            )

    self._resolve_rank_mapping(info)
    self.prefill_info_table[bootstrap_addr] = info
    logger.debug(f"Prefill parallel info for [{bootstrap_addr}]: {info}")
    return True


def manager_resolve_rank_mapping(self, info: PrefillServerInfo) -> None:
    """Compute TP/CP/PP rank mapping and store on the PrefillServerInfo object.
    Deterministic for a given (bootstrap_addr, decode engine) pair."""
    # TP rank mapping
    if self.attn_tp_size == info.attn_tp_size:
        target_tp_rank = self.kv_args.engine_rank % self.attn_tp_size
        required_dst_info_num = 1
        required_prefill_response_num = 1
        target_tp_ranks = [target_tp_rank]
    elif self.attn_tp_size > info.attn_tp_size:
        if not self.is_mla_backend and not self.is_hybrid_mla_backend:
            logger.warning_once(
                "Performance is NOT guaranteed when using different TP sizes for non-MLA models. "
            )
        target_tp_rank = (self.kv_args.engine_rank % self.attn_tp_size) // (
            self.attn_tp_size // info.attn_tp_size
        )
        required_dst_info_num = self.attn_tp_size // info.attn_tp_size
        required_prefill_response_num = 1
        target_tp_ranks = [target_tp_rank]
    else:
        if not self.is_mla_backend and not self.is_hybrid_mla_backend:
            logger.warning_once(
                "Performance is NOT guaranteed when using different TP sizes for non-MLA models. "
            )
        # For non-MLA models, one decode rank needs to retrieve KVCache from multiple prefill ranks
        target_tp_ranks = list(
            range(
                (self.kv_args.engine_rank % self.attn_tp_size)
                * (info.attn_tp_size // self.attn_tp_size),
                (self.kv_args.engine_rank % self.attn_tp_size + 1)
                * (info.attn_tp_size // self.attn_tp_size),
            )
        )
        # For MLA models, we can retrieve KVCache from only one prefill rank, but we still need to maintain
        # multiple connections in the connection pool and have to send dummy requests to other prefill ranks,
        # or the KVPoll will never be set correctly
        target_tp_rank = target_tp_ranks[0]
        required_dst_info_num = 1
        if self.is_mla_backend:
            required_prefill_response_num = 1
        else:
            required_prefill_response_num = info.attn_tp_size // self.attn_tp_size

    # CP rank mapping — decode cp size should be equal to 1
    assert self.attn_cp_size == 1, (
        f"Decode cp size ({self.attn_cp_size}) should be equal to 1",
    )
    if self.attn_cp_size == info.attn_cp_size:
        assert info.attn_cp_size == 1, (
            f"When prefill cp size is 1, attn cp size should be 1, but got {self.attn_cp_size}",
        )
        target_cp_ranks = [self.attn_cp_rank]
    else:
        target_cp_ranks = list(range(info.attn_cp_size))
        pull_from_all_cp_ranks = (
            self.enable_all_cp_ranks_for_transfer
            or info.enable_dsa_cache_layer_split
            or info.cp_cache_layer_split
        )
        if not pull_from_all_cp_ranks:
            # Only retrieve from prefill CP rank 0 when not using all ranks
            target_cp_ranks = target_cp_ranks[:1]
            required_prefill_response_num *= 1
        else:
            required_prefill_response_num *= info.attn_cp_size // self.attn_cp_size

    # PP rank mapping — decode pp size should be equal to prefill pp size or 1
    assert self.pp_size == info.pp_size or self.pp_size == 1, (
        f"Decode pp size ({self.pp_size}) should be equal to prefill pp size ({info.pp_size}) or 1",
    )
    if info.pp_size == self.pp_size:
        target_pp_ranks = [self.pp_rank]
    else:
        target_pp_ranks = list(range(info.pp_size))
        required_prefill_response_num *= info.pp_size // self.pp_size

    info.target_tp_rank = target_tp_rank
    info.target_tp_ranks = target_tp_ranks
    info.target_cp_ranks = target_cp_ranks
    info.target_pp_ranks = target_pp_ranks
    info.required_dst_info_num = required_dst_info_num
    info.required_prefill_response_num = required_prefill_response_num


def manager_get_mla_kv_ptrs_with_pp(
    self,
    src_kv_ptrs: List[int],
    dst_kv_ptrs: List[int],
    state_type: Optional[StateType] = None,
) -> Tuple[List[int], List[int], int]:
    # Fast path: both sides use exactly the same PP layout
    if len(src_kv_ptrs) == len(dst_kv_ptrs):
        return src_kv_ptrs, dst_kv_ptrs, len(src_kv_ptrs)

    if state_type == StateType.DSA:
        hf_config = getattr(self.model_config, "hf_config", None)
        if hf_config is not None and is_deepseek_dsa(hf_config):
            end_layer = self.kv_args.prefill_end_layer
            if end_layer is None:
                raise ValueError(
                    "prefill_end_layer is required for compact DSA state transfer"
                )

            compact_indexer_layer_ids = get_dsa_full_indexer_layer_ids(hf_config)
            total_layers = hf_config.num_hidden_layers
            # HiCache/HiSparse deliberately retain the dense main layout.
            # Infer the decode registration layout from its pointer count;
            # compact GLM layouts are smaller than num_hidden_layers.
            indexer_layer_ids = (
                list(range(total_layers))
                if len(dst_kv_ptrs) >= total_layers
                else compact_indexer_layer_ids
            )
            start_index = sum(
                layer_id < self.kv_args.prefill_start_layer
                for layer_id in indexer_layer_ids
            )
            end_index = sum(layer_id < end_layer for layer_id in indexer_layer_ids)
            src_main_layers = end_index - start_index
            src_draft_layers = len(src_kv_ptrs) - src_main_layers
            if src_draft_layers < 0:
                raise ValueError(
                    "DSA PP state pointer count is smaller than the number of "
                    "full-indexer layers in this stage"
                )

            sliced_dst_kv_ptrs = list(dst_kv_ptrs[start_index:end_index])
            if src_draft_layers:
                draft_start = len(indexer_layer_ids)
                sliced_dst_kv_ptrs.extend(
                    dst_kv_ptrs[draft_start : draft_start + src_draft_layers]
                )
            if len(sliced_dst_kv_ptrs) != len(src_kv_ptrs):
                raise ValueError(
                    "DSA PP state pointer mismatch after compact indexer mapping: "
                    f"src={len(src_kv_ptrs)}, dst={len(sliced_dst_kv_ptrs)}"
                )
            return src_kv_ptrs, sliced_dst_kv_ptrs, len(src_kv_ptrs)

    mla_ratios = getattr(self.kv_args, "mla_compression_ratios", None)
    if mla_ratios:
        # Compressed-MLA pointers are grouped by buffer type;
        # each group needs its own PP-stage slice.
        sliced_src_kv_ptrs, sliced_dst_kv_ptrs = self._mla_slice_ptrs_for_pp(
            src_kv_ptrs, dst_kv_ptrs, mla_ratios, state_type
        )
        return (
            sliced_src_kv_ptrs,
            sliced_dst_kv_ptrs,
            len(sliced_src_kv_ptrs),
        )

    # Regular MLA PP slicing
    # Decode pp size should be equal to prefill pp size or 1
    num_draft = (
        getattr(self.kv_args, "num_draft_entries", 0) if state_type is None else 0
    )
    start_layer, end_layer = self._mla_kv_entry_span_with_pp(
        len(src_kv_ptrs) - num_draft
    )
    sliced_dst_kv_ptrs = dst_kv_ptrs[start_layer:end_layer]
    if num_draft:
        # Draft entries follow the target's global layer range on both peers.
        sliced_dst_kv_ptrs += dst_kv_ptrs[-num_draft:]
    return src_kv_ptrs, sliced_dst_kv_ptrs, len(src_kv_ptrs)


def receiver_get_bootstrap_info_from_server(
    self, prefill_dp_rank, prefill_cp_rank, target_tp_rank, target_pp_rank
):
    """Fetch the bootstrap info from the bootstrap server."""
    try:
        url = f"http://{self.bootstrap_addr}/route?prefill_dp_rank={prefill_dp_rank}&prefill_cp_rank={prefill_cp_rank}&target_tp_rank={target_tp_rank}&target_pp_rank={target_pp_rank}"
        response = _get_bootstrap_session(self.bootstrap_addr).get(url, timeout=5)
        if response.status_code == 200:
            bootstrap_info = response.json()
            bootstrap_info["pp_rank"] = int(target_pp_rank)
            # PD hidden-state transfer resolves the source rank from these.
            bootstrap_info["target_cp_rank"] = int(prefill_cp_rank)
            bootstrap_info["target_tp_rank"] = int(target_tp_rank)
            bootstrap_info["target_pp_rank"] = int(target_pp_rank)
            return bootstrap_info
        else:
            logger.error(
                f"Failed to get prefill server info: {response.status_code}, {response.text}"
            )
            return None
    except Exception as e:
        logger.error(f"Error fetching prefill info from bootstrap: {e}")
        return None


def receiver_connect(cls, endpoint: str, is_ipv6: bool = False):
    with cls._global_lock:
        if endpoint not in cls._socket_cache:
            sock = cls._ctx.socket(zmq.PUSH)
            if is_ipv6:
                sock.setsockopt(zmq.IPV6, 1)
            sock.setsockopt(zmq.LINGER, 0)
            # Bound send so a dead peer cannot block the scheduler forever.
            sock.setsockopt(
                zmq.SNDTIMEO,
                envs.SGLANG_DISAGGREGATION_ZMQ_SEND_TIMEOUT.get() * 1000,
            )
            # Bound the queued send backlog as well as individual sends.
            sock.setsockopt(zmq.SNDHWM, 1000)
            sock.connect(endpoint)
            cls._socket_cache[endpoint] = sock
            cls._socket_locks[endpoint] = threading.Lock()
        return cls._socket_cache[endpoint], cls._socket_locks[endpoint]


def bootstrap_init(self, host: str, port: int):
    self.host = host
    self.port = port
    self.app = web.Application()
    self.store = dict()
    self.lock = asyncio.Lock()
    # The event loop only keeps weak references to tasks, so a long-lived
    # task needs a strong reference to survive garbage collection.
    self._background_tasks: Set[asyncio.Task] = set()
    self._setup_routes()
    self.pp_size = None
    self.attn_tp_size = None
    self.attn_cp_size = None
    self.dp_size = None
    self.page_size = None
    self.kv_cache_dtype: Optional[str] = None
    self.kv_layout: Optional[str] = None
    self.dsv41_spec_layout: Optional[dict] = None
    self.kv_cache_layout: Optional[str] = None
    self.follow_bootstrap_room: Optional[bool] = None
    self.enable_dsa_cache_layer_split: Optional[bool] = None
    self.cp_cache_layer_split: Optional[bool] = None
    self.prefill_http_port: Optional[int] = None
    self.decode_allocation_policy = "early"
    self.prefill_port_table: Dict[
        int, Dict[int, Dict[int, Dict[int, PrefillRankInfo]]]
    ] = {}
    self.room_to_dp_rank: Dict[int, Dict[str, Union[int, float]]] = {}
    self._registered_count = 0
    self.entry_cleanup_interval = (
        envs.SGLANG_DISAGGREGATION_BOOTSTRAP_ENTRY_CLEANUP_INTERVAL.get()
    )

    # Start bootstrap server
    self.thread = threading.Thread(target=self._run_server, daemon=True)
    self.run()


async def bootstrap_handle_route_put(self, request: web.Request):
    data = await request.json()
    attn_tp_size = data["attn_tp_size"]
    attn_tp_rank = data["attn_tp_rank"]
    attn_cp_size = data["attn_cp_size"]
    attn_cp_rank = data["attn_cp_rank"]
    attn_dp_size = data["attn_dp_size"]
    attn_dp_rank = data["attn_dp_rank"]
    pp_size = data["pp_size"]
    pp_rank = data["pp_rank"]
    system_dp_size = data["system_dp_size"]
    system_dp_rank = data["system_dp_rank"]
    rank_ip = data["rank_ip"]
    rank_port = int(data["rank_port"])
    page_size = int(data["page_size"])
    kv_cache_dtype = data["kv_cache_dtype"]
    kv_cache_layout = data.get("kv_cache_layout")
    kv_layout = data.get("kv_layout")
    prefill_http_port = data.get("prefill_http_port")
    dsv41_spec_layout = data.get("dsv41_spec_layout")

    if self._registered_count and self.dsv41_spec_layout != dsv41_spec_layout:
        return web.Response(
            text="DeepSeek-V4.1 DSpark PD layout differs across prefill ranks",
            status=400,
        )
    policy = data.get("decode_allocation_policy", "early")
    if self._registered_count and self.decode_allocation_policy != policy:
        return web.Response(
            text="Inconsistent prefill allocation policies", status=400
        )
    self.dsv41_spec_layout = dsv41_spec_layout
    self.decode_allocation_policy = policy

    if self.attn_tp_size is None:
        self.attn_tp_size = attn_tp_size

    if self.attn_cp_size is None:
        self.attn_cp_size = attn_cp_size

    if self.dp_size is None:
        self.dp_size = attn_dp_size if system_dp_size == 1 else system_dp_size

    if self.pp_size is None:
        self.pp_size = pp_size

    if self.page_size is None and page_size is not None:
        self.page_size = page_size

    if self.kv_cache_dtype is None and kv_cache_dtype is not None:
        self.kv_cache_dtype = kv_cache_dtype

    if self.kv_cache_layout is None and kv_cache_layout is not None:
        self.kv_cache_layout = kv_cache_layout
    if self.kv_layout is None and kv_layout is not None:
        self.kv_layout = kv_layout

    if self.prefill_http_port is None and prefill_http_port is not None:
        self.prefill_http_port = int(prefill_http_port)

    if self.follow_bootstrap_room is None:
        load_balance_method = data.get(
            "load_balance_method", "follow_bootstrap_room"
        )
        self.follow_bootstrap_room = load_balance_method == "follow_bootstrap_room"

    if self.enable_dsa_cache_layer_split is None:
        self.enable_dsa_cache_layer_split = bool(
            data.get("enable_dsa_cache_layer_split", False)
        )
    if self.cp_cache_layer_split is None:
        self.cp_cache_layer_split = bool(data.get("cp_cache_layer_split", False))

    if system_dp_size == 1:
        dp_group = attn_dp_rank
    else:
        dp_group = system_dp_rank

    # Add lock to make sure thread-safe
    async with self.lock:
        dp_group_table = self.prefill_port_table.setdefault(dp_group, {})
        cp_group_table = dp_group_table.setdefault(attn_cp_rank, {})
        tp_group_table = cp_group_table.setdefault(attn_tp_rank, {})

        tp_group_table[pp_rank] = PrefillRankInfo(
            rank_ip=rank_ip,
            rank_port=rank_port,
        )

        self._registered_count += 1

    expected = self.dp_size * self.attn_cp_size * self.attn_tp_size * self.pp_size
    logger.debug(
        f"Register prefill bootstrap: DP{dp_group} CP{attn_cp_rank} TP{attn_tp_rank} PP{pp_rank} with rank_ip: {rank_ip} and rank_port: {rank_port}"
        f" ({self._registered_count}/{expected} registered)"
    )

    return web.Response(text="OK", status=200)


async def bootstrap_handle_route_get(self, request: web.Request):
    prefill_dp_rank = request.query.get("prefill_dp_rank")
    prefill_cp_rank = request.query.get("prefill_cp_rank")
    target_tp_rank = request.query.get("target_tp_rank")
    target_pp_rank = request.query.get("target_pp_rank")
    if (
        not prefill_dp_rank
        or not prefill_cp_rank
        or not target_tp_rank
        or not target_pp_rank
    ):
        return web.Response(text="Missing inputs for bootstrap server.", status=400)

    if (
        int(prefill_dp_rank) == -1
        and int(prefill_cp_rank) == -1
        and int(target_tp_rank) == -1
        and int(target_pp_rank) == -1
    ):
        if not self._is_ready():
            return web.Response(
                text=f"Prefill server not fully registered yet"
                f" ({self._registered_count} workers registered).",
                status=503,
            )
        info = PrefillServerInfo(
            attn_tp_size=self.attn_tp_size,
            attn_cp_size=self.attn_cp_size,
            dp_size=self.dp_size,
            pp_size=self.pp_size,
            page_size=self.page_size,
            kv_cache_dtype=self.kv_cache_dtype,
            kv_layout=self.kv_layout,
            dsv41_spec_layout=self.dsv41_spec_layout,
            kv_cache_layout=self.kv_cache_layout,
            follow_bootstrap_room=(
                self.follow_bootstrap_room
                if self.follow_bootstrap_room is not None
                else True
            ),
            enable_dsa_cache_layer_split=bool(self.enable_dsa_cache_layer_split),
            cp_cache_layer_split=bool(self.cp_cache_layer_split),
            prefill_http_port=self.prefill_http_port,
            decode_allocation_policy=self.decode_allocation_policy,
        )
        payload = dataclasses.asdict(info)
        if info.dsv41_spec_layout is None:
            payload.pop("dsv41_spec_layout")
        if info.kv_layout is None:
            payload.pop("kv_layout")
        return web.json_response(payload, status=200)

    if not self._is_ready():
        return web.Response(
            text=f"Prefill server not fully registered yet"
            f" ({self._registered_count} workers registered).",
            status=503,
        )

    # Find corresponding prefill info
    try:
        async with self.lock:
            bootstrap_info = self.prefill_port_table[int(prefill_dp_rank)][
                int(prefill_cp_rank)
            ][int(target_tp_rank)][int(target_pp_rank)]
    except KeyError:
        return web.Response(
            text=f"Bootstrap info not found for dp_rank={prefill_dp_rank} cp_rank={prefill_cp_rank} "
            f"tp_rank={target_tp_rank} pp_rank={target_pp_rank}",
            status=404,
        )

    return web.json_response(dataclasses.asdict(bootstrap_info), status=200)




def get_dsa_full_indexer_layer_ids(
    config: PretrainedConfig,
    start_layer: int = 0,
    end_layer: Optional[int] = None,
) -> List[int]:
    """Return DSA layers that run an indexer instead of reusing top-k."""
    if end_layer is None:
        end_layer = config.num_hidden_layers
    assert 0 <= start_layer <= end_layer
    if not is_deepseek_dsa(config):
        return list(range(start_layer, end_layer))
    return [
        layer_id
        for layer_id in range(start_layer, end_layer)
        if not dsa_layer_skips_topk(config, layer_id)
    ]

