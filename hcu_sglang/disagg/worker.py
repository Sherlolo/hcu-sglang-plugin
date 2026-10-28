"""Upstream transfer worker with the HCU registration-snapshot argument."""
from __future__ import annotations
def transfer_worker(
    self,
    queue: FastQueue,
    executor: concurrent.futures.ThreadPoolExecutor,
    staging_buffer=None,
    worker_index=0,
):
    staging_strategy = None
    if self.enable_trace:
        trace_set_thread_info(
            f"mooncake transfer worker {worker_index}",
            tp_rank=self.attn_tp_rank,
            dp_rank=self.attn_dp_rank,
        )

    while True:
        kv_chunk: Optional[TransferKVChunk] = None
        try:
            kv_chunk = queue.get()
            # teardown() pushes a None sentinel to unblock get() and stop
            # the worker: FastQueue.get() blocks indefinitely, so checking
            # _stopped alone can never wake a parked worker during a role switch.
            if kv_chunk is None:
                break
            if self.enable_trace:
                kv_chunk.trace_ctx.rebuild_thread_context()
                kv_chunk.trace_ctx.trace_slice_start(
                    MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                    MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                )

            # Counted at dequeue, before the status check, so
            # `outstanding == 0` means nothing is dequeued or in flight --
            # the predicate the abort ack relies on. The flag survives
            # re-enqueue on defer.
            if not kv_chunk.staging_counted:
                self._staging_outstanding[kv_chunk.room] += 1
                kv_chunk.staging_counted = True

            # Under overlap scheduling an early cached-prefix send can
            # precede completion of the previous prefill forward. Wait on
            # the worker, not the scheduler, before reading source KV.
            # A room that already failed skips the wait; the status check
            # below then drops the chunk, including on an abort during it.
            if kv_chunk.wait_event is not None:
                if (
                    kv_chunk.room in self.request_status
                    and self.check_status(kv_chunk.room) != KVPoll.Failed
                ):
                    kv_chunk.wait_event.synchronize()
                kv_chunk.wait_event = None

            if (
                kv_chunk.room not in self.request_status
                or self.check_status(kv_chunk.room) == KVPoll.Failed
            ):
                logger.debug(
                    f"Skipping chunk for room {kv_chunk.room} because it has already failed or been aborted"
                )
                if self.enable_trace:
                    kv_chunk.trace_ctx.trace_slice_end(
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                        thread_finish_flag=True,
                    )
                self._staging_outstanding.pop(kv_chunk.room, None)
                if self.enable_deferred_decode_kv_release:
                    # Skipped => nothing written for this aborted room; ack.
                    self._maybe_ack_drained_abort(kv_chunk.room)
                continue

            if (
                self.enable_staging
                and staging_strategy is None
                and staging_buffer is not None
            ):
                staging_strategy = self._try_create_staging_strategy(staging_buffer)
            reqs_to_be_processed = (
                self.transfer_infos[kv_chunk.room].values()
                if kv_chunk.room in self.transfer_infos
                else []
            )
            polls = []
            dst_ranks_infos = []
            # Unique id per prefill sender so decode's response set size matches expected_response_num.
            prefill_unique_rank = self._prefill_unique_rank()
            # When staging transfer is not yet ready (watermark/allocation pending),
            # the chunk is re-enqueued and we break out of the req loop to retry later.
            staging_deferred = False
            for req in reqs_to_be_processed:
                start_ts = time.perf_counter()
                if not req.is_dummy:
                    # Early exit if the request has failed
                    with self.session_lock:
                        layout_rejection = self.state_layout_rejections.get(
                            req.mooncake_session_id
                        )
                        if (
                            layout_rejection is not None
                            or req.mooncake_session_id in self.failed_sessions
                        ):
                            self.conclude_failure(
                                bootstrap_room=kv_chunk.room,
                                failure_reason=layout_rejection
                                or (
                                    "Decode instance could be dead, remote "
                                    f"mooncake session {req.mooncake_session_id} "
                                    "is not alive"
                                ),
                            )
                            break

                    target_rank_registration_info: KVArgsRegisterInfo = (
                        self.decode_kv_args_table[req.mooncake_session_id]
                    )
                    is_dcp_transfer = (
                        target_rank_registration_info.requires_dcp_relayout
                    )
                    chunked_dst_device_kv_indice = None
                    if is_dcp_transfer:
                        if req.dst_device_kv_indices is not None:
                            raise RuntimeError(
                                "HiSparse destination device indices are not "
                                "supported by PD DCP relayout"
                            )
                        chunked_dst_kv_indice = req.dst_kv_indices
                    else:
                        chunked_dst_kv_indice = req.dst_kv_indices[
                            kv_chunk.index_slice
                        ]
                        if req.dst_device_kv_indices is not None:
                            chunked_dst_device_kv_indice = (
                                req.dst_device_kv_indices[kv_chunk.index_slice]
                            )

                        # NOTE: This is temporarily a workaround to deal with the case where the prefill_kv_indices
                        # is mismatched with the dst_kv_indices when page size > 1, this should never happen.
                        if len(chunked_dst_kv_indice) < len(
                            kv_chunk.prefill_kv_indices
                        ):
                            logger.warning(
                                f"len(chunked_dst_kv_indice) = {len(chunked_dst_kv_indice)}, len(kv_chunk.prefill_kv_indices) = {len(kv_chunk.prefill_kv_indices)}"
                            )
                            kv_chunk.prefill_kv_indices = (
                                kv_chunk.prefill_kv_indices[
                                    : len(chunked_dst_kv_indice)
                                ]
                            )
                        if chunked_dst_device_kv_indice is not None:
                            chunked_dst_device_kv_indice = (
                                chunked_dst_device_kv_indice[
                                    : len(kv_chunk.prefill_kv_indices)
                                ]
                            )

                    skip_kv, skip_state = self._get_dsa_cache_transfer_skip_flags(
                        target_rank_registration_info
                    )
                    if (
                        len(kv_chunk.prefill_kv_indices) == 0
                        or not self.kv_args.kv_data_ptrs
                        or skip_kv
                    ):
                        ret = 0
                    elif is_dcp_transfer:
                        dcp_token_item_lens = (
                            target_rank_registration_info.dcp_token_item_lens
                        )
                        assert dcp_token_item_lens is not None
                        pack_buffer = (
                            self._dcp_pack_buffers[worker_index]
                            if self._dcp_pack_buffers
                            else None
                        )
                        ret = self.send_kvcache_dcp(
                            req.mooncake_session_id,
                            kv_chunk.prefill_kv_indices,
                            target_rank_registration_info.dst_kv_ptrs,
                            chunked_dst_kv_indice,
                            dcp_token_item_lens=dcp_token_item_lens,
                            dst_dcp_size=target_rank_registration_info.dst_dcp_size,
                            dst_dcp_rank=target_rank_registration_info.dst_dcp_rank,
                            src_page_offset=kv_chunk.index_slice.start or 0,
                            decode_prefix_len=req.decode_prefix_len or 0,
                            num_kv_tokens=kv_chunk.num_kv_tokens,
                            executor=executor,
                            dst_layer_ids=(
                                target_rank_registration_info.dst_kv_layer_ids
                            ),
                            pack_buffer=pack_buffer,
                            dst_kv_item_lens=target_rank_registration_info.dst_kv_item_lens,
                            dst_tp_rank=target_rank_registration_info.dst_tp_rank,
                            dst_attn_tp_size=target_rank_registration_info.dst_attn_tp_size,
                        )
                    elif (
                        self.is_mla_backend
                        or self.is_hybrid_mla_backend
                        or self.attn_tp_size
                        == target_rank_registration_info.dst_attn_tp_size
                    ):
                        ret = self.send_kvcache(
                            req.mooncake_session_id,
                            kv_chunk.prefill_kv_indices,
                            target_rank_registration_info.dst_kv_ptrs,
                            chunked_dst_kv_indice,
                            executor,
                            dst_layer_ids=target_rank_registration_info.dst_kv_layer_ids,
                            dst_device_kv_indices=chunked_dst_device_kv_indice,
                            dst_kv_item_len=target_rank_registration_info.dst_kv_item_len,
                            dst_attn_tp_size=target_rank_registration_info.dst_attn_tp_size,
                            v4_transfer_metadata=target_rank_registration_info.v4_transfer_metadata,
                        )
                    elif (
                        self.enable_staging
                        and staging_strategy is not None
                        and (
                            target_rank_registration_info.staging_base_ptr != 0
                            or target_rank_registration_info.staging_total_size != 0
                        )
                    ):
                        ret, deferred = self._do_staging_transfer(
                            staging_strategy,
                            kv_chunk,
                            req,
                            target_rank_registration_info,
                            chunked_dst_kv_indice,
                            executor,
                            queue,
                            prefill_unique_rank,
                        )
                        if deferred:
                            staging_deferred = True
                            # Chunk re-enqueued; stop processing remaining reqs for this chunk
                            break
                    else:
                        ret = self.send_kvcache_slice(
                            req.mooncake_session_id,
                            kv_chunk.prefill_kv_indices,
                            target_rank_registration_info.dst_kv_ptrs,
                            chunked_dst_kv_indice,
                            target_rank_registration_info.dst_tp_rank,
                            target_rank_registration_info.dst_attn_tp_size,
                            target_rank_registration_info.dst_kv_item_len,
                            executor,
                            target_rank_registration_info.dst_kv_layer_ids,
                        )
                    if ret != 0:
                        with self.session_lock:
                            self.session_failures[req.mooncake_session_id] += 1
                            # Failures should never happen if the session is not dead, if the session fails once, mark it as failed
                            if self.session_failures[req.mooncake_session_id] >= 1:
                                self.failed_sessions.add(req.mooncake_session_id)
                                logger.error(
                                    f"Session {req.mooncake_session_id} failed."
                                )
                        self._poison_unsettled_send(kv_chunk.room)
                        self.conclude_failure(
                            bootstrap_room=kv_chunk.room,
                            failure_reason=(
                                f"Failed to send kv chunk of {kv_chunk.room} to "
                                f"{NetworkAddress(req.endpoint, req.dst_port).to_host_port_str()}"
                            ),
                        )
                        break

                    if kv_chunk.is_last_chunk:
                        if kv_chunk.state_indices and not skip_state:
                            state_rc = self.maybe_send_extra(
                                req,
                                kv_chunk.state_indices,
                                executor,
                                target_rank_registration_info,
                            )
                            # Before the Failed check: an ABORT during the
                            # send also marks the room Failed.
                            if state_rc != 0:
                                self._poison_unsettled_send(kv_chunk.room)
                            if (
                                self.request_status.get(
                                    kv_chunk.room, KVPoll.Failed
                                )
                                == KVPoll.Failed
                            ):
                                break
                            if state_rc != 0:
                                with self.session_lock:
                                    self.session_failures[
                                        req.mooncake_session_id
                                    ] += 1
                                    self.failed_sessions.add(
                                        req.mooncake_session_id
                                    )
                                self.conclude_failure(
                                    bootstrap_room=kv_chunk.room,
                                    failure_reason=(
                                        "Failed to send state components of "
                                        f"{kv_chunk.room} to "
                                        f"{NetworkAddress(req.endpoint, req.dst_port).to_host_port_str()}"
                                    ),
                                )
                                break

                        # Only the last chunk we need to send the aux data
                        ret = self.send_aux(
                            req,
                            kv_chunk.prefill_aux_index,
                            target_rank_registration_info.dst_aux_ptrs,
                        )
                        if ret != 0:
                            # The aux write targets decode's metadata
                            # slot, which the deferred release also frees.
                            self._poison_unsettled_send(req.room)
                        polls.append(True if ret == 0 else False)
                        dst_ranks_infos.append((req.endpoint, req.dst_port))

                        # Only sync status when all the dst ranks have received the kvcache
                        if len(polls) == req.required_dst_info_num:
                            status = KVPoll.Success if all(polls) else KVPoll.Failed
                            self.conclude_transfer(
                                bootstrap_room=req.room,
                                status=status,
                                targets=dst_ranks_infos,
                                failure_reason=(
                                    None
                                    if status == KVPoll.Success
                                    else f"Failed to send aux data of {req.room}"
                                ),
                            )
                else:
                    # Dummy request means the decode instance is not used, so its status can be marked as success directly
                    # Dummy request does not need to sync status to decode endpoint
                    if kv_chunk.is_last_chunk and req.room in self.request_status:
                        self.update_status(req.room, KVPoll.Success)

                if self.enable_trace:
                    mooncake_trace_slice(
                        kv_chunk.trace_ctx,
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND_SESSION,
                        start_ts,
                    )

            if self.enable_trace:
                kv_chunk.trace_ctx.trace_slice_end(
                    MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                    MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                    thread_finish_flag=True,
                )

            if staging_deferred:
                continue

            self._staging_outstanding[kv_chunk.room] -= 1
            if self.enable_deferred_decode_kv_release:
                # In-flight write finished; if aborted and nothing outstanding,
                # the pages are idle -> release the held ack.
                self._maybe_ack_drained_abort(kv_chunk.room)
            # Tear down only when no chunk is still outstanding and the room
            # has concluded: already cleared, Success, or a Failed *last*
            # chunk. A non-last Failed chunk keeps the room (more chunks may
            # follow), not on the last chunk alone since an earlier deferred
            # chunk may still need to transfer.
            if self._staging_outstanding.get(kv_chunk.room, 0) <= 0 and (
                kv_chunk.room not in self.request_status
                or self.check_status(kv_chunk.room) == KVPoll.Success
                or (
                    kv_chunk.is_last_chunk
                    and self.check_status(kv_chunk.room) == KVPoll.Failed
                )
            ):
                if kv_chunk.room in self.transfer_infos:
                    self.transfer_infos.pop(kv_chunk.room)
                self.req_to_decode_prefix_len.pop(kv_chunk.room, None)
                if self.enable_staging:
                    # Purge prefetch bookkeeping for the finished room.
                    # Snapshot first: the scheduler thread adds concurrently.
                    for key in list(self._staging_ctx.prefetch_requested):
                        if key[0] == kv_chunk.room:
                            self._staging_ctx.prefetch_requested.discard(key)
                    self._staging_ctx.prefetched_rooms.discard(kv_chunk.room)
                # Publish worker cleanup last: deferred allocation must not
                # acknowledge Success while this worker can still erase
                # bookkeeping for a subsequent transfer using the same room.
                self._staging_outstanding.pop(kv_chunk.room, None)

        except Exception as e:
            # NOTE(shangming): Remove this when we make sure the transfer thread is bug-free
            if kv_chunk is not None:
                self.poison_deferred_ack_room(kv_chunk.room)
            raise RuntimeError(
                f"Transfer thread failed because of {e}. Prefill instance with bootstrap_port={self.bootstrap_port} is dead."
            )
