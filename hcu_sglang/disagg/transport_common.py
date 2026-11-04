from __future__ import annotations

from abc import abstractmethod

from sglang.srt.disaggregation.common import conn as N
from sglang.srt.disaggregation.base import conn as B
from hcu_sglang.disagg.protocol import StateType, PrefillServerInfo
from hcu_sglang.utils.detection import is_hcu

globals().update({name: value for name, value in vars(N).items() if not name.startswith("__") and name not in ("StateType", "PrefillServerInfo", "is_hcu")})

def validate_dsa_state_transfer_abi(
    src_format: str,
    dst_format: str,
    src_item_lens: List[int],
    dst_item_lens: List[int],
) -> None:
    """Validate the page ABI before transferring a DSA index-K cache."""
    if src_format != dst_format:
        raise RuntimeError(
            "DSA index-K cache transfer ABI mismatch between prefill and decode: "
            f"prefill_format={src_format!r}, decode_format={dst_format!r}. "
            "Ensure P and D use the same SGLANG_DSA_HCU_INT8_INDEX_K_CACHE "
            "setting, page size, model, and SGLang revision."
        )

    src_sizes = {int(item_len) for item_len in src_item_lens}
    dst_sizes = {int(item_len) for item_len in dst_item_lens}
    if len(src_sizes) == 1 and src_sizes == dst_sizes:
        return

    raise RuntimeError(
        "DSA index-K cache page layout mismatch between prefill and decode: "
        f"prefill_item_lens={sorted(src_sizes)}, "
        f"decode_item_lens={sorted(dst_sizes)}. Ensure P and D use the same "
        "SGLANG_DSA_HCU_INT8_INDEX_K_CACHE setting, page size, model, and "
        "SGLang revision."
    )

class CommonKVManagerMethods:
    def supports_pd_hidden_streaming(self) -> bool:
        return False

    def mark_pd_hidden_request_done(
        self,
        bootstrap_room: int,
        state_indices: Optional[List] = None,
    ) -> None:
        """Mark the hidden-transfer portion of a request done.

        Backends that support streaming hidden transfer override this to release
        their source window independently from KV request completion.
        """
        del bootstrap_room, state_indices
        return None

    def pop_pd_hidden_request_done(self, bootstrap_room: int) -> bool:
        """Consume a hidden-request-done event for early source-window release."""
        del bootstrap_room
        return False

    def _wake_pd_hidden_ack_waiters(self, bootstrap_room: int) -> None:
        """Wake backend-specific PD hidden ACK waiters after request failure."""
        del bootstrap_room
        return None

    def mark_pd_hidden_done(
        self,
        bootstrap_room: int,
        state_indices: Optional[List] = None,
    ) -> None:
        self.mark_pd_hidden_request_done(bootstrap_room, state_indices)

    def pop_pd_hidden_done(self, bootstrap_room: int) -> bool:
        return self.pop_pd_hidden_request_done(bootstrap_room)

    def validate_remote_state_transfer_abis(
        self,
        dst_state_data_formats: List[str],
        dst_state_item_lens: List[List[int]],
    ) -> None:
        """Validate request-state page layouts advertised by a decode peer."""
        state_types = getattr(self.kv_args, "state_types", []) or []
        src_state_data_formats = getattr(self.kv_args, "state_data_formats", []) or []
        src_state_item_lens = getattr(self.kv_args, "state_item_lens", []) or []

        for i, state_type in enumerate(state_types):
            if state_type != StateType.DSA:
                continue

            src_format = (
                src_state_data_formats[i] if i < len(src_state_data_formats) else ""
            )
            dst_format = (
                dst_state_data_formats[i] if i < len(dst_state_data_formats) else ""
            )
            src_item_lens = (
                src_state_item_lens[i] if i < len(src_state_item_lens) else []
            )
            dst_item_lens = (
                dst_state_item_lens[i] if i < len(dst_state_item_lens) else []
            )
            validate_dsa_state_transfer_abi(
                src_format,
                dst_format,
                src_item_lens,
                dst_item_lens,
            )

    def get_mha_kv_ptrs_with_pp(
        self, src_kv_ptrs: List[int], dst_kv_ptrs: List[int]
    ) -> Tuple[List[int], List[int], List[int], List[int], int]:
        total_draft_layers = getattr(self.kv_args, "total_draft_kv_layers", None) or 0
        total_main_layers = getattr(self.kv_args, "total_main_kv_layers", None) or 0
        if total_draft_layers > 0 and total_main_layers > 0:
            src_num_layers = len(src_kv_ptrs) // 2
            dst_num_layers = len(dst_kv_ptrs) // 2
            src_k_ptrs = list(src_kv_ptrs[:src_num_layers])
            src_v_ptrs = list(src_kv_ptrs[src_num_layers:])
            dst_k_all = list(dst_kv_ptrs[:dst_num_layers])
            dst_v_all = list(dst_kv_ptrs[dst_num_layers:])

            # Equal PP partitions register the same local main/draft layers.
            if src_num_layers == dst_num_layers:
                return src_k_ptrs, src_v_ptrs, dst_k_all, dst_v_all, src_num_layers

            main_start = getattr(self.kv_args, "prefill_main_start_layer", None)
            main_end = getattr(self.kv_args, "prefill_main_end_layer", None)
            draft_start = getattr(self.kv_args, "prefill_draft_start_layer", None)
            draft_end = getattr(self.kv_args, "prefill_draft_end_layer", None)
            if None in (main_start, main_end, draft_start, draft_end):
                raise ValueError(
                    "MHA+MTP PP transfer requires prefill main/draft layer ranges "
                    "on KVArgs."
                )

            expected_dst_layers = total_main_layers + total_draft_layers
            if dst_num_layers != expected_dst_layers:
                raise ValueError(
                    f"Unexpected decode MHA+MTP KV pointer layout: "
                    f"dst_num_layers={dst_num_layers}, expected={expected_dst_layers} "
                    f"(main={total_main_layers}, draft={total_draft_layers})."
                )

            dst_k_ptrs = (
                dst_k_all[main_start:main_end]
                + dst_k_all[
                    total_main_layers + draft_start : total_main_layers + draft_end
                ]
            )
            dst_v_ptrs = (
                dst_v_all[main_start:main_end]
                + dst_v_all[
                    total_main_layers + draft_start : total_main_layers + draft_end
                ]
            )
            if len(dst_k_ptrs) != src_num_layers:
                raise ValueError(
                    f"MHA+MTP PP pointer mismatch: src_layers={src_num_layers}, "
                    f"dst_layers={len(dst_k_ptrs)}, main=[{main_start},{main_end}), "
                    f"draft=[{draft_start},{draft_end})."
                )
            return src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, src_num_layers

        start_layer = self.kv_args.prefill_start_layer
        num_kv_layers = len(src_kv_ptrs) // 2
        configured_end_layer = getattr(self.kv_args, "prefill_end_layer", None)
        end_layer = start_layer + num_kv_layers
        dst_num_total_layers = len(dst_kv_ptrs) // 2
        src_k_ptrs = src_kv_ptrs[:num_kv_layers]
        src_v_ptrs = src_kv_ptrs[num_kv_layers:]

        def is_hyv3_model() -> bool:
            hf_config = getattr(self.model_config, "hf_config", None)
            archs = (
                getattr(hf_config, "architectures", None)
                or getattr(self.model_config, "architectures", None)
                or []
            )
            return any(str(arch).startswith("HYV3ForCausalLM") for arch in archs)

        if is_hyv3_model():
            if num_kv_layers == dst_num_total_layers:
                dst_k_ptrs = dst_kv_ptrs[:dst_num_total_layers]
                dst_v_ptrs = dst_kv_ptrs[dst_num_total_layers:]
            elif (
                num_kv_layers < dst_num_total_layers
                and dst_num_total_layers % num_kv_layers != 0
                and getattr(self.model_config, "is_draft_model", False)
            ):
                multiplier_ratio = dst_num_total_layers // num_kv_layers
                dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
                v_ptr_offset = num_kv_layers * multiplier_ratio
                dst_v_ptrs = dst_kv_ptrs[
                    v_ptr_offset + start_layer : v_ptr_offset + end_layer
                ]
            else:
                dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
                dst_v_ptrs = dst_kv_ptrs[
                    dst_num_total_layers + start_layer : dst_num_total_layers
                    + end_layer
                ]
            return src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, len(src_k_ptrs)

        def is_mha_layer_type(layer_type) -> bool:
            layer_type = str(layer_type)
            return layer_type in ("attention", "full_attention", "sliding_attention")

        def get_global_full_attention_layer_ids():
            hf_text_config = getattr(self.model_config, "hf_text_config", None)
            layer_types = getattr(hf_text_config, "layer_types", None)
            if layer_types is None:
                layer_types = getattr(hf_text_config, "layers_block_type", None)
            if layer_types is None:
                hf_config = getattr(self.model_config, "hf_config", None)
                layer_types = getattr(hf_config, "layer_types", None)
                if layer_types is None:
                    layer_types = getattr(hf_config, "layers_block_type", None)
            if layer_types:
                return [
                    layer_id
                    for layer_id, layer_type in enumerate(layer_types)
                    if is_mha_layer_type(layer_type)
                ]
            full_attention_layer_ids = getattr(
                self.model_config, "full_attention_layer_ids", None
            )
            if not full_attention_layer_ids:
                return None
            full_attention_layer_ids = list(full_attention_layer_ids)
            num_hidden_layers = getattr(self.model_config, "num_hidden_layers", None)
            if (
                num_hidden_layers is not None
                and len(full_attention_layer_ids) < num_hidden_layers
                and start_layer > 0
                and all(
                    layer_id >= start_layer for layer_id in full_attention_layer_ids
                )
            ):
                return None
            return full_attention_layer_ids

        num_hidden_layers = getattr(self.model_config, "num_hidden_layers", None)
        full_attention_layer_ids = get_global_full_attention_layer_ids()
        is_compact_full_attention_layout = (
            full_attention_layer_ids is not None
            and num_hidden_layers is not None
            and len(full_attention_layer_ids) < num_hidden_layers
        )
        decode_has_appended_draft_kv = (
            num_hidden_layers is not None
            and dst_num_total_layers > num_hidden_layers
            and end_layer <= num_hidden_layers
        )
        main_total_layers = (
            len(full_attention_layer_ids)
            if is_compact_full_attention_layout
            else num_hidden_layers
        )
        if (
            num_kv_layers != dst_num_total_layers
            and main_total_layers is not None
            and len(dst_kv_ptrs) >= 2 * main_total_layers
            and (is_compact_full_attention_layout or decode_has_appended_draft_kv)
        ):
            # Decode normalizes main and draft/MTP KV as:
            # [K_main..., K_draft..., V_main..., V_draft...].  Hybrid
            # linear-attention models store only full-attention layers in the
            # KV pool, so PP stage model-layer ids must be converted to compact
            # full-attention KV indices before slicing the decode-side pointers.
            if is_compact_full_attention_layout:
                model_end_layer = (
                    configured_end_layer
                    if configured_end_layer is not None
                    else end_layer
                )
                compact_start_layer = sum(
                    1 for layer_id in full_attention_layer_ids if layer_id < start_layer
                )
                local_attention_layer_ids = [
                    layer_id
                    for layer_id in full_attention_layer_ids
                    if start_layer <= layer_id < model_end_layer
                ]
                compact_end_layer = compact_start_layer + len(local_attention_layer_ids)
                if len(local_attention_layer_ids) != num_kv_layers:
                    compact_end_layer = compact_start_layer + num_kv_layers
            else:
                compact_start_layer = start_layer
                compact_end_layer = compact_start_layer + num_kv_layers
            if compact_end_layer <= main_total_layers:
                dst_k_ptrs = dst_kv_ptrs[compact_start_layer:compact_end_layer]
                dst_v_ptrs = dst_kv_ptrs[
                    dst_num_total_layers + compact_start_layer : dst_num_total_layers
                    + compact_end_layer
                ]
                layers_current_pp_stage = len(src_k_ptrs)
                if not (
                    len(src_k_ptrs)
                    == len(src_v_ptrs)
                    == len(dst_k_ptrs)
                    == len(dst_v_ptrs)
                    == layers_current_pp_stage
                ):
                    logger.error(
                        "MHA PP ptr slicing mismatch in compact path: "
                        "start_layer=%s, num_kv_layers=%s, end_layer=%s, "
                        "dst_num_total_layers=%s, main_total_layers=%s, "
                        "compact_start_layer=%s, compact_end_layer=%s, "
                        "src_k=%s, src_v=%s, dst_k=%s, dst_v=%s, "
                        "len_src_kv_ptrs=%s, len_dst_kv_ptrs=%s, "
                        "full_attention_layer_ids_len=%s",
                        start_layer,
                        num_kv_layers,
                        end_layer,
                        dst_num_total_layers,
                        main_total_layers,
                        compact_start_layer,
                        compact_end_layer,
                        len(src_k_ptrs),
                        len(src_v_ptrs),
                        len(dst_k_ptrs),
                        len(dst_v_ptrs),
                        len(src_kv_ptrs),
                        len(dst_kv_ptrs),
                        (
                            len(full_attention_layer_ids)
                            if full_attention_layer_ids is not None
                            else None
                        ),
                    )
                return (
                    src_k_ptrs,
                    src_v_ptrs,
                    dst_k_ptrs,
                    dst_v_ptrs,
                    layers_current_pp_stage,
                )
        if num_kv_layers == dst_num_total_layers:
            dst_k_ptrs = dst_kv_ptrs[:dst_num_total_layers]
            dst_v_ptrs = dst_kv_ptrs[dst_num_total_layers:]
        elif (
            num_kv_layers < dst_num_total_layers
            and dst_num_total_layers % num_kv_layers != 0
            and self.model_config.is_draft_model
        ):
            # Case: Decode has draft model KV while Prefill is deployed without speculative decoding
            # dst_kv_ptrs layout: [K_main..., V_main..., draft_K..., draft_V...]
            multiplier_ratio = dst_num_total_layers // num_kv_layers
            dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
            v_ptr_offset = num_kv_layers * multiplier_ratio
            dst_v_ptrs = dst_kv_ptrs[
                v_ptr_offset + start_layer : v_ptr_offset + end_layer
            ]
        else:
            # Decode pp size should be equal to prefill pp size or 1
            dst_k_ptrs = dst_kv_ptrs[start_layer:end_layer]
            dst_v_ptrs = dst_kv_ptrs[
                dst_num_total_layers + start_layer : dst_num_total_layers + end_layer
            ]
        layers_current_pp_stage = len(src_k_ptrs)
        if not (
            len(src_k_ptrs)
            == len(src_v_ptrs)
            == len(dst_k_ptrs)
            == len(dst_v_ptrs)
            == layers_current_pp_stage
        ):
            logger.error(
                "MHA PP ptr slicing mismatch: start_layer=%s, num_kv_layers=%s, "
                "end_layer=%s, dst_num_total_layers=%s, main_total_layers=%s, "
                "src_k=%s, src_v=%s, dst_k=%s, dst_v=%s, "
                "len_src_kv_ptrs=%s, len_dst_kv_ptrs=%s, "
                "full_attention_layer_ids_len=%s, is_draft_model=%s",
                start_layer,
                num_kv_layers,
                end_layer,
                dst_num_total_layers,
                main_total_layers,
                len(src_k_ptrs),
                len(src_v_ptrs),
                len(dst_k_ptrs),
                len(dst_v_ptrs),
                len(src_kv_ptrs),
                len(dst_kv_ptrs),
                (
                    len(full_attention_layer_ids)
                    if full_attention_layer_ids is not None
                    else None
                ),
                getattr(self.model_config, "is_draft_model", None),
            )
        return src_k_ptrs, src_v_ptrs, dst_k_ptrs, dst_v_ptrs, layers_current_pp_stage

class CommonKVSenderMethods:
    def _prepare_send_indices(
        self,
        kv_indices: npt.NDArray[np.int32],
        state_indices: Optional[List] = None,
    ) -> Tuple[npt.NDArray[np.int32], slice, bool, bool]:
        """Common pre-processing for send(): index tracking and CP-rank handling.

        Returns:
            (kv_indices, index_slice, is_last_chunk, should_skip)
            If should_skip is True, the caller should return immediately.
        """
        index_slice = slice(self.curr_idx, self.curr_idx + len(kv_indices))
        self.curr_idx += len(kv_indices)
        is_last_chunk = self.curr_idx == self.num_kv_indices

        if (
            self.kv_mgr.enable_all_cp_ranks_for_transfer
            and not get_parallel().enable_dsa_cache_layer_split
            and not getattr(self.kv_mgr, "cp_cache_layer_split", False)
        ):
            kv_indices, index_slice = filter_kv_indices_for_cp_rank(
                self.kv_mgr,
                kv_indices,
                index_slice,
                total_pages=self.num_kv_indices,
            )
        elif self.kv_mgr.is_dummy_cp_rank:
            if not is_last_chunk:
                return kv_indices, index_slice, is_last_chunk, True
            else:
                self.kv_mgr.update_status(self.bootstrap_room, KVPoll.Success)
                return kv_indices, index_slice, is_last_chunk, True

        return kv_indices, index_slice, is_last_chunk, False

    def clear(self) -> None:
        notification = None
        if self.kv_mgr.deferred_bootstrap is not None:
            if not self._owns_bootstrap_room:
                return
            notification = self.kv_mgr.deferred_bootstrap.close(
                self.bootstrap_room,
                self,
                success=self.kv_mgr.request_status.get(self.bootstrap_room)
                == KVPoll.Success,
            )
            self._owns_bootstrap_room = False
        self.kv_mgr.request_status.pop(self.bootstrap_room, None)
        self.kv_mgr.req_to_decode_prefix_len.pop(self.bootstrap_room, None)
        if hasattr(self.kv_mgr, "req_to_pd_hidden_meta"):
            self.kv_mgr.req_to_pd_hidden_meta.pop(self.bootstrap_room, None)
        self.kv_mgr.transfer_infos.pop(self.bootstrap_room, None)
        # A sender can be cleared while a counted transfer is still writing:
        # keep the target for the worker, or ACK now if nothing is in flight.
        self.kv_mgr._maybe_ack_drained_abort(self.bootstrap_room)
        if notification is not None:
            endpoint, failed = notification
            self.kv_mgr.send_kv_status_message(
                targets=[endpoint],
                bootstrap_room=self.bootstrap_room,
                status=KVPoll.Failed if failed else KVPoll.Success,
                failure_reason="Prefill bootstrap failed" if failed else None,
            )

class CommonKVReceiverMethods:
    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List[int]] = None,
        decode_prefix_len: Optional[int] = None,
        destination: KVTransferDestination = KVTransferDestination.DEVICE,
        spec_metadata: Optional[dict] = None,
    ):
        raise NotImplementedError

class BaseKVManagerMethods:
    def should_verify_kv_checksum(self, bootstrap_addr: str) -> bool:
        """Whether the checksum sent by the prefill at bootstrap_addr is comparable
        with one over this rank's local KV. A backend returns False for a peer
        whose digest covers a different byte range, e.g. a TP width mismatch where
        each decode rank receives only a slice of what the sender digested."""
        return True

class BaseKVSenderMethods:
    def set_source_event(self, source_event) -> None:
        del source_event

    def set_pd_hidden_chunk_meta(
        self,
        hidden_start: int,
        row_len: int,
        is_last_hidden_chunk: bool,
        release_indices: Optional[List[int]] = None,
    ) -> None:
        del hidden_start, row_len, is_last_hidden_chunk, release_indices

class BaseKVReceiverMethods:
    @abstractmethod
    def send_metadata(
        self,
        kv_indices: npt.NDArray[np.int32],
        aux_index: Optional[int] = None,
        state_indices: Optional[List] = None,
        decode_prefix_len: Optional[int] = None,
        destination: KVTransferDestination = KVTransferDestination.DEVICE,
        spec_metadata: Optional[dict] = None,
    ):
        """
        Notify the prefill server about the kv indices, aux index, and state_indices.
        """
        ...

def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from sglang.srt.disaggregation.common import utils as U
    from hcu_sglang.disagg import common_utils as owned
    for name in ("TransferKVChunk", "PDHiddenChunk", "PDHiddenRequestState", "pack_string_list", "unpack_string_list"):
        setattr(U, name, getattr(owned, name))
    N.validate_dsa_state_transfer_abi = validate_dsa_state_transfer_abi
    if hasattr(N.CommonKVManager, 'supports_pd_hidden_streaming'):
        HookRegistry.register("sglang.srt.disaggregation.common.conn.CommonKVManager.supports_pd_hidden_streaming", CommonKVManagerMethods.__dict__['supports_pd_hidden_streaming'], HookType.REPLACE)
    else:
        setattr(N.CommonKVManager, 'supports_pd_hidden_streaming', CommonKVManagerMethods.__dict__['supports_pd_hidden_streaming'])
    if hasattr(N.CommonKVManager, 'mark_pd_hidden_request_done'):
        HookRegistry.register("sglang.srt.disaggregation.common.conn.CommonKVManager.mark_pd_hidden_request_done", CommonKVManagerMethods.__dict__['mark_pd_hidden_request_done'], HookType.REPLACE)
    else:
        setattr(N.CommonKVManager, 'mark_pd_hidden_request_done', CommonKVManagerMethods.__dict__['mark_pd_hidden_request_done'])
    if hasattr(N.CommonKVManager, 'pop_pd_hidden_request_done'):
        HookRegistry.register("sglang.srt.disaggregation.common.conn.CommonKVManager.pop_pd_hidden_request_done", CommonKVManagerMethods.__dict__['pop_pd_hidden_request_done'], HookType.REPLACE)
    else:
        setattr(N.CommonKVManager, 'pop_pd_hidden_request_done', CommonKVManagerMethods.__dict__['pop_pd_hidden_request_done'])
    if hasattr(N.CommonKVManager, '_wake_pd_hidden_ack_waiters'):
        HookRegistry.register("sglang.srt.disaggregation.common.conn.CommonKVManager._wake_pd_hidden_ack_waiters", CommonKVManagerMethods.__dict__['_wake_pd_hidden_ack_waiters'], HookType.REPLACE)
    else:
        setattr(N.CommonKVManager, '_wake_pd_hidden_ack_waiters', CommonKVManagerMethods.__dict__['_wake_pd_hidden_ack_waiters'])
    if hasattr(N.CommonKVManager, 'mark_pd_hidden_done'):
        HookRegistry.register("sglang.srt.disaggregation.common.conn.CommonKVManager.mark_pd_hidden_done", CommonKVManagerMethods.__dict__['mark_pd_hidden_done'], HookType.REPLACE)
    else:
        setattr(N.CommonKVManager, 'mark_pd_hidden_done', CommonKVManagerMethods.__dict__['mark_pd_hidden_done'])
    if hasattr(N.CommonKVManager, 'pop_pd_hidden_done'):
        HookRegistry.register("sglang.srt.disaggregation.common.conn.CommonKVManager.pop_pd_hidden_done", CommonKVManagerMethods.__dict__['pop_pd_hidden_done'], HookType.REPLACE)
    else:
        setattr(N.CommonKVManager, 'pop_pd_hidden_done', CommonKVManagerMethods.__dict__['pop_pd_hidden_done'])
    if hasattr(N.CommonKVManager, 'validate_remote_state_transfer_abis'):
        HookRegistry.register("sglang.srt.disaggregation.common.conn.CommonKVManager.validate_remote_state_transfer_abis", CommonKVManagerMethods.__dict__['validate_remote_state_transfer_abis'], HookType.REPLACE)
    else:
        setattr(N.CommonKVManager, 'validate_remote_state_transfer_abis', CommonKVManagerMethods.__dict__['validate_remote_state_transfer_abis'])
    if hasattr(N.CommonKVManager, 'get_mha_kv_ptrs_with_pp'):
        HookRegistry.register("sglang.srt.disaggregation.common.conn.CommonKVManager.get_mha_kv_ptrs_with_pp", CommonKVManagerMethods.__dict__['get_mha_kv_ptrs_with_pp'], HookType.REPLACE)
    else:
        setattr(N.CommonKVManager, 'get_mha_kv_ptrs_with_pp', CommonKVManagerMethods.__dict__['get_mha_kv_ptrs_with_pp'])
    if hasattr(N.CommonKVSender, '_prepare_send_indices'):
        HookRegistry.register("sglang.srt.disaggregation.common.conn.CommonKVSender._prepare_send_indices", CommonKVSenderMethods.__dict__['_prepare_send_indices'], HookType.REPLACE)
    else:
        setattr(N.CommonKVSender, '_prepare_send_indices', CommonKVSenderMethods.__dict__['_prepare_send_indices'])
    if hasattr(N.CommonKVSender, 'clear'):
        HookRegistry.register("sglang.srt.disaggregation.common.conn.CommonKVSender.clear", CommonKVSenderMethods.__dict__['clear'], HookType.REPLACE)
    else:
        setattr(N.CommonKVSender, 'clear', CommonKVSenderMethods.__dict__['clear'])
    if hasattr(N.CommonKVReceiver, 'send_metadata'):
        HookRegistry.register("sglang.srt.disaggregation.common.conn.CommonKVReceiver.send_metadata", CommonKVReceiverMethods.__dict__['send_metadata'], HookType.REPLACE)
    else:
        setattr(N.CommonKVReceiver, 'send_metadata', CommonKVReceiverMethods.__dict__['send_metadata'])
    if not hasattr(B.BaseKVManager, 'should_verify_kv_checksum'):
        setattr(B.BaseKVManager, 'should_verify_kv_checksum', BaseKVManagerMethods.__dict__['should_verify_kv_checksum'])
    if not hasattr(B.BaseKVSender, 'set_source_event'):
        setattr(B.BaseKVSender, 'set_source_event', BaseKVSenderMethods.__dict__['set_source_event'])
    if not hasattr(B.BaseKVSender, 'set_pd_hidden_chunk_meta'):
        setattr(B.BaseKVSender, 'set_pd_hidden_chunk_meta', BaseKVSenderMethods.__dict__['set_pd_hidden_chunk_meta'])
    HookRegistry.register("sglang.srt.disaggregation.base.conn.BaseKVReceiver.send_metadata", BaseKVReceiverMethods.__dict__["send_metadata"], HookType.REPLACE)
