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

            if kv_chunk.source_event is not None:
                kv_chunk.source_event.synchronize()
                kv_chunk.source_event = None
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
                # Only a recorded Failed status has hidden-ack waiters to
                # release; an unknown room never registered any.
                if kv_chunk.room in self.request_status:
                    self._wake_pd_hidden_ack_waiters(kv_chunk.room)
                logger.debug(
                    f"Skipping chunk for room {kv_chunk.room} because it has already failed or been aborted"
                )
                if kv_chunk.pd_hidden_start is not None:
                    with self.pd_hidden_events.inflight_lock:
                        self.pd_hidden_events.inflight_chunks.pop(
                            kv_chunk.room, None
                        )
                if not kv_chunk.pd_hidden_sent and self._has_pd_hidden_state(
                    kv_chunk.state_indices
                ):
                    self._release_or_mark_pd_hidden_done(kv_chunk)
                if self.enable_trace:
                    kv_chunk.trace_ctx.trace_slice_end(
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                        thread_finish_flag=True,
                    )
                self._staging_outstanding.pop(kv_chunk.room, None)
                self._notify_transfer_waiters()
                if self.enable_deferred_decode_kv_release:
                    # Skipped => nothing written for this aborted room; ack.
                    self._maybe_ack_drained_abort(kv_chunk.room)
                self._source_drain.complete(kv_chunk)
                continue

            reqs_to_be_processed = list(
                self.transfer_infos[kv_chunk.room].values()
                if kv_chunk.room in self.transfer_infos
                else []
            )
            # Unique id per prefill sender so decode's response set size matches expected_response_num.
            prefill_unique_rank = self._prefill_unique_rank()

            registration_error = None
            for req in reqs_to_be_processed:
                if req.is_dummy:
                    continue
                registration_info = self.decode_kv_args_table.get(
                    req.mooncake_session_id
                )
                if registration_info is None:
                    registration_error = (
                        "Decode peer registration is missing for Mooncake "
                        f"session {req.mooncake_session_id!r}"
                    )
                    break
                if registration_info.registration_error is not None:
                    registration_error = (
                        f"Decode peer {req.mooncake_session_id!r} is "
                        f"incompatible: {registration_info.registration_error}"
                    )
                    break

            if registration_error is not None:
                logger.error(
                    "Rejecting PD transfer for room %s: %s",
                    kv_chunk.room,
                    registration_error,
                )
                self.record_failure(kv_chunk.room, registration_error)
                self.update_status(kv_chunk.room, KVPoll.Failed)
                for req in reqs_to_be_processed:
                    if not req.is_dummy:
                        self.sync_status_to_decode_endpoint(
                            req.endpoint,
                            req.dst_port,
                            req.room,
                            KVPoll.Failed,
                            prefill_unique_rank,
                        )
                if self.enable_trace:
                    kv_chunk.trace_ctx.trace_slice_end(
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.stage_name,
                        MooncakeRequestStage.MOONCAKE_WORKER_SEND.level,
                        thread_finish_flag=True,
                    )
                self._staging_outstanding[kv_chunk.room] -= 1
                if self._staging_outstanding[kv_chunk.room] <= 0:
                    self._staging_outstanding.pop(kv_chunk.room, None)
                self._notify_transfer_waiters()
                if self.enable_deferred_decode_kv_release:
                    self._maybe_ack_drained_abort(kv_chunk.room)
                self._source_drain.complete(kv_chunk)
                continue
            if (
                self.enable_staging
                and staging_strategy is None
                and staging_buffer is not None
            ):
                staging_strategy = self._try_create_staging_strategy(staging_buffer)
            polls = []
            dst_ranks_infos = []
            pd_hidden_expected = 0
            pd_hidden_done_count = 0
            # When staging transfer is not yet ready (watermark/allocation pending),
            # the chunk is re-enqueued and we break out of the req loop to retry later.
            staging_deferred = False
            pd_hidden_deferred = False
            pd_hidden_failed = False
            if (
                not kv_chunk.pd_hidden_sent
                and kv_chunk.state_indices
                and self._has_pd_hidden_state(kv_chunk.state_indices)
            ):
                state_idx = self._pd_hidden_state_index()
                for req in reqs_to_be_processed:
                    if req.is_dummy:
                        continue
                    target_rank_registration_info = self.decode_kv_args_table[
                        req.mooncake_session_id
                    ]
                    _, skip_state = self._get_dsa_cache_transfer_skip_flags(
                        target_rank_registration_info
                    )
                    if not skip_state:
                        pd_hidden_expected += 1

                waiting_for_ack = (
                    kv_chunk.pd_hidden_start is not None
                    and kv_chunk.pd_hidden_ready_sent
                )
                hidden_inflight_key = (
                    (prefill_unique_rank, int(kv_chunk.pd_hidden_start))
                    if kv_chunk.pd_hidden_start is not None
                    else None
                )
                if (
                    hidden_inflight_key is not None
                    and not kv_chunk.pd_hidden_ready_sent
                ):
                    with self.pd_hidden_events.inflight_lock:
                        inflight_key = self.pd_hidden_events.inflight_chunks.get(
                            kv_chunk.room
                        )
                    if (
                        inflight_key is not None
                        and inflight_key != hidden_inflight_key
                    ):
                        self._park_pd_hidden_chunk_behind_room(queue, kv_chunk)
                        continue
                ack_ready = False
                if waiting_for_ack:
                    ack_ready = kv_chunk.pd_hidden_ack_ready
                    if ack_ready:
                        pd_hidden_done_count = pd_hidden_expected
                        self._finish_pd_hidden_streaming_chunk(
                            kv_chunk, hidden_inflight_key
                        )
                    elif kv_chunk.pd_hidden_ack_timed_out:
                        pd_hidden_failed = True

                if not ack_ready and not waiting_for_ack:
                    for req in reqs_to_be_processed:
                        if req.is_dummy:
                            continue
                        with self.session_lock:
                            session_failed = (
                                req.mooncake_session_id in self.failed_sessions
                            )
                        if session_failed:
                            continue
                        target_rank_registration_info = self.decode_kv_args_table[
                            req.mooncake_session_id
                        ]
                        _, skip_state = self._get_dsa_cache_transfer_skip_flags(
                            target_rank_registration_info
                        )
                        if skip_state:
                            continue
                        self._begin_pd_hidden_transfer(kv_chunk.room)
                        try:
                            ret, pd_hidden_done = self._send_pd_hidden_packet(
                                req,
                                kv_chunk.state_indices,
                                kv_chunk.pd_hidden_packet_idx,
                                executor,
                                target_rank_registration_info=target_rank_registration_info,
                            )
                        finally:
                            self._end_pd_hidden_transfer(kv_chunk.room)
                        if ret != 0:
                            self._mark_session_failed_and_sync(
                                kv_chunk=kv_chunk,
                                req=req,
                                prefill_unique_rank=prefill_unique_rank,
                                failure_reason=(
                                    "Failed to send PD hidden packet "
                                    f"{kv_chunk.pd_hidden_packet_idx} of "
                                    f"{kv_chunk.room} to "
                                    f"{NetworkAddress(req.endpoint, req.dst_port).to_host_port_str()}"
                                ),
                            )
                            if kv_chunk.pd_hidden_start is not None:
                                with self.pd_hidden_events.inflight_lock:
                                    self.pd_hidden_events.inflight_chunks.pop(
                                        kv_chunk.room, None
                                    )
                            self._release_or_mark_pd_hidden_done(kv_chunk)
                            pd_hidden_failed = True
                            break
                        if not pd_hidden_done:
                            pd_hidden_deferred = True
                            continue
                        if (
                            kv_chunk.pd_hidden_start is not None
                            and state_idx is not None
                            and state_idx < len(kv_chunk.state_indices)
                        ):
                            dst_state_idx = resolve_state_component_dst_index(self.kv_args.state_types, target_rank_registration_info.dst_state_types, state_idx)
                            dst_indices = (
                                req.dst_state_indices[dst_state_idx]
                                if dst_state_idx < len(req.dst_state_indices)
                                else []
                            )
                            row_len = int(kv_chunk.pd_hidden_row_len)
                            chunk_dst_indices = [
                                int(x) for x in dst_indices[:row_len]
                            ]
                            self.notify_pd_hidden_chunk_ready(
                                remote=req.endpoint,
                                dst_port=req.dst_port,
                                room=req.room,
                                prefill_rank=prefill_unique_rank,
                                hidden_start=int(kv_chunk.pd_hidden_start),
                                row_len=row_len,
                                is_last_hidden_chunk=bool(
                                    kv_chunk.pd_hidden_is_last_chunk
                                ),
                                dst_indices=chunk_dst_indices,
                            )
                            continue
                        pd_hidden_done_count += 1

                if pd_hidden_failed:
                    self._source_drain.complete(kv_chunk)
                    continue
                if waiting_for_ack and not ack_ready:
                    # A parked chunk is only re-enqueued by ACK/abort/timeout.
                    # Reaching this branch without a terminal status means a
                    # duplicate wakeup; avoid busy requeueing it.
                    continue
                if (
                    kv_chunk.pd_hidden_start is not None
                    and not kv_chunk.pd_hidden_ready_sent
                ):
                    if hidden_inflight_key is not None:
                        with self.pd_hidden_events.inflight_lock:
                            self.pd_hidden_events.inflight_chunks[kv_chunk.room] = (
                                hidden_inflight_key
                            )
                    kv_chunk.pd_hidden_ready_sent = True
                    if self.park_pd_hidden_chunk_for_ack(
                        transfer_queue=queue,
                        kv_chunk=kv_chunk,
                        prefill_rank=prefill_unique_rank,
                        expected_count=pd_hidden_expected,
                    ):
                        continue
                    ack_ready = True
                    pd_hidden_done_count = pd_hidden_expected
                    self._finish_pd_hidden_streaming_chunk(
                        kv_chunk, hidden_inflight_key
                    )
                current_status = self.request_status.get(kv_chunk.room)
                if (
                    pd_hidden_expected > 0
                    and pd_hidden_done_count == pd_hidden_expected
                    and current_status is not None
                    and current_status != KVPoll.Failed
                ):
                    kv_chunk.pd_hidden_sent = True
                    if kv_chunk.pd_hidden_start is not None:
                        if kv_chunk.pd_hidden_is_last_chunk:
                            self.mark_pd_hidden_request_done(
                                kv_chunk.room,
                                None,
                            )
                    else:
                        self.mark_pd_hidden_request_done(
                            kv_chunk.room,
                            self._pd_hidden_release_state_indices(kv_chunk),
                        )
                if pd_hidden_deferred:
                    kv_chunk.pd_hidden_packet_idx += 1
                    queue.put(kv_chunk)
                    continue

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
                            if (
                                kv_chunk.is_last_chunk
                                and not kv_chunk.pd_hidden_sent
                                and self._has_pd_hidden_state(
                                    kv_chunk.state_indices
                                )
                            ):
                                self._release_or_mark_pd_hidden_done(kv_chunk)
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

                        # A source/destination page-count mismatch means the
                        # decode allocation no longer describes this prefill
                        # chunk. Truncating either side silently accepts a
                        # partial KV cache and can produce plausible-looking
                        # but corrupted tokens. Fail the request before RDMA
                        # instead; PP consensus will release both stages.
                        src_page_count = len(kv_chunk.prefill_kv_indices)
                        dst_page_count = len(chunked_dst_kv_indice)
                        if src_page_count != dst_page_count:
                            failure_reason = (
                                "KV page-count mismatch before Mooncake transfer: "
                                f"room={kv_chunk.room}, src_pages={src_page_count}, "
                                f"dst_pages={dst_page_count}, slice={kv_chunk.index_slice}"
                            )
                            logger.error(failure_reason)
                            self.record_failure(kv_chunk.room, failure_reason)
                            self.update_status(kv_chunk.room, KVPoll.Failed)
                            self.sync_status_to_decode_endpoint(
                                req.endpoint,
                                req.dst_port,
                                req.room,
                                KVPoll.Failed,
                                prefill_unique_rank,
                            )
                            break

                    skip_kv, skip_state = self._get_dsa_cache_transfer_skip_flags(
                        target_rank_registration_info
                    )
                    if kv_chunk.kv_sent:
                        ret = 0
                    elif (
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
                        # Asymmetric attn TP (prefill != decode) without staging:
                        # fall back to the per-slice path.  This is the chain's
                        # catch-all -- without it `ret` stays unbound below.
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
                        self._poison_unsettled_send(kv_chunk.room)
                        self._mark_session_failed_and_sync(
                            kv_chunk=kv_chunk,
                            req=req,
                            prefill_unique_rank=prefill_unique_rank,
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
                                self._without_pd_hidden_state(
                                    kv_chunk.state_indices
                                ),
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

            current_status = self.request_status.get(kv_chunk.room)
            if current_status is not None and current_status != KVPoll.Failed:
                kv_chunk.kv_sent = True
            if (
                kv_chunk.is_last_chunk
                and not kv_chunk.pd_hidden_sent
                and pd_hidden_expected > 0
                and pd_hidden_done_count == pd_hidden_expected
                and current_status is not None
                and current_status != KVPoll.Failed
            ):
                if kv_chunk.pd_hidden_start is not None:
                    if kv_chunk.pd_hidden_is_last_chunk:
                        self.mark_pd_hidden_request_done(
                            kv_chunk.room,
                            self._pd_hidden_release_state_indices(kv_chunk),
                        )
                else:
                    self.mark_pd_hidden_request_done(
                        kv_chunk.room,
                        self._pd_hidden_release_state_indices(kv_chunk),
                    )
            if pd_hidden_deferred:
                kv_chunk.pd_hidden_packet_idx += 1
                queue.put(kv_chunk)
                continue

            self._staging_outstanding[kv_chunk.room] -= 1
            if self.enable_deferred_decode_kv_release:
                # In-flight write finished; if aborted and nothing outstanding,
                # the pages are idle -> release the held ack.
                self._maybe_ack_drained_abort(kv_chunk.room)
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

            self._source_drain.complete(kv_chunk)
            self._notify_transfer_waiters()

        except Exception as e:
            # Submission/event/allocator failure can have ambiguous side
            # effects. Quarantine the room instead of asserting a drain.
            if kv_chunk is not None:
                self._source_drain.poison(kv_chunk.room)
                self.poison_deferred_ack_room(kv_chunk.room)
                logger.error(
                    "Source drain is unknown for room %s; retaining resources",
                    kv_chunk.room,
                )
                self.record_failure(kv_chunk.room, str(e))
                self.update_status(kv_chunk.room, KVPoll.Failed)
            # The failed chunk's request will be surfaced as failed by
            # the normal transfer-failure path (update_status +
            # sync_status_to_decode_endpoint above); do not let a
            # single chunk kill the whole transfer thread, which
            # would silently stall every other in-flight KV transfer
            # on this prefill.
            logger.exception(
                "transfer_worker chunk failed (bootstrap_port=%s): %s; "
                "continuing to next chunk",
                self.bootstrap_port,
                e,
            )
            continue
