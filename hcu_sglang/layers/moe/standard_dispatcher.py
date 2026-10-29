from __future__ import annotations

import torch

from sglang.srt.layers.moe.token_dispatcher import standard as S
from hcu_sglang.utils.detection import is_dcu


def _map_topk_ids_to_local_experts(
    topk_ids: torch.Tensor, local_expert_mapping: torch.Tensor
) -> torch.Tensor:
    """Map global expert ids while preserving the ``-1`` drop sentinel."""
    return torch.where(
        (topk_ids < 0) | (topk_ids >= local_expert_mapping.numel()),
        -1,
        local_expert_mapping[topk_ids.clamp(0, local_expert_mapping.numel() - 1)],
    )


def _dispatch(
    self, hidden_states: torch.Tensor, topk_output: S.TopKOutput
) -> S.StandardDispatchOutput:

    if S.should_use_flashinfer_cutlass_moe_fp4_allgather():
        # all-gather fp4 hidden states
        if (
            S.fp4_quantize_flashinfer is None
            or S.nvfp4_block_scale_interleave_flashinfer is None
        ):
            raise RuntimeError(
                "FlashInfer fp4_quantize and nvfp4_block_scale_interleave "
                "are required for the flashinfer_cutlass FP4 all-gather "
                "path."
            )
        global_scale = self.quant_config.get("input_global_scale", None)
        assert global_scale is not None, "input_global_scale is not set"
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids

        # Quantize before comm, swizzle after.
        with S.use_symmetric_memory(
            S.get_parallel().tp_group, disabled=not S.is_allocation_symmetric()
        ):
            if hidden_states.shape[0] > 0:
                x, x_sf = S.fp4_quantize_flashinfer(
                    hidden_states, global_scale, is_sf_swizzled_layout=False
                )
            else:
                x_col = hidden_states.shape[1]
                x = torch.zeros(
                    0, x_col // 2, dtype=torch.uint8, device=hidden_states.device
                )
                x_sf = torch.zeros(
                    0, x_col // 16, dtype=torch.uint8, device=hidden_states.device
                )
        topk_weights, topk_ids, x, x_sf = S.get_parallel().tp_group.all_gatherv(
            [topk_weights, topk_ids, x, x_sf], sizes=S.get_dp_global_num_tokens()
        )
        # TODO: fuse into cutlass moe
        x_sf = S.nvfp4_block_scale_interleave_flashinfer(x_sf)

        hidden_states = x
        hidden_states_scale = x_sf
        topk_output = S.StandardTopKOutput(
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            router_logits=topk_output.router_logits,  # never tested
        )
    else:
        hidden_states = hidden_states
        hidden_states_scale = None

    if (
        self.moe_ep_size > 1
        and not self.skip_local_expert_mapping
        and S.TopKOutputChecker.format_is_standard(topk_output)
    ):
        if self.local_expert_mapping is None:
            device = S.get_device()
            self.local_expert_mapping = torch.full(
                (self.num_experts,), -1, dtype=torch.int32, device=device
            )
            self.local_expert_mapping[
                self.moe_ep_rank * self.num_local_routed_experts : (
                    self.moe_ep_rank + 1
                )
                * self.num_local_routed_experts
            ] = torch.arange(
                0, self.num_local_routed_experts, dtype=torch.int32, device=device
            )

            if self.num_local_shared_experts > 0:
                self.local_expert_mapping[-self.num_local_shared_experts :] = (
                    torch.arange(
                        self.num_local_routed_experts,
                        self.num_local_routed_experts
                        + self.num_local_shared_experts,
                        dtype=torch.int32,
                        device="cpu",
                    )
                )

    if self.local_expert_mapping is not None and not self.skip_local_expert_mapping:
        if self.use_aiter_moe_runner and self.expert_mask_gpu is None:
            self.expert_mask_gpu = (
                (
                    (self.local_expert_mapping >= 0)
                    & (self.local_expert_mapping < self.num_local_experts)
                )
                .to(torch.int32)
                .to(device="cuda")
            )
        elif not self.use_aiter_moe_runner:
            if S.TopKOutputChecker.format_is_standard(topk_output):
                topk_ids_local = _map_topk_ids_to_local_experts(
                    topk_output.topk_ids, self.local_expert_mapping
                )
                # Drop dp-attention MAX_LEN pad rows from the dispatch:
                # pad rows carry stale hidden through the router and
                # their expert outputs are discarded downstream — pure
                # wasted compute (and a masked-grouped-GEMM workspace
                # blow-up when they collide on the same top-k). -1 is the
                # drop sentinel both the triton and deep_gemm runners honor.
                if S._MASK_DP_PAD_MOE and S.is_dp_max_padding():
                    S.mask_dp_pad_moe_topk_ids(topk_ids_local)
                topk_output = topk_output._replace(topk_ids=topk_ids_local)
            elif S.TopKOutputChecker.format_is_triton_kernels(topk_output):
                raise NotImplementedError()

    return S.StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        topk_output=topk_output,
    )


def around_dispatch(original, self, hidden_states, topk_output):
    if not is_dcu():
        return original(self, hidden_states, topk_output)
    return _dispatch(self, hidden_states, topk_output)


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.moe.token_dispatcher.standard.StandardDispatcher.dispatch",
        around_dispatch,
        HookType.AROUND,
    )
