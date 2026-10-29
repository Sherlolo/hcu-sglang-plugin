"""HCU disaggregation pointer-layout compatibility."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Tuple

from hcu_sglang.utils.detection import is_dcu

logger = logging.getLogger(__name__)


def after_prefill_bootstrap_queue_init(result: Any, queue: Any, *args, **kwargs) -> Any:
    del args, kwargs
    if not is_dcu():
        return result

    tp_worker = queue.scheduler.tp_worker
    if not getattr(tp_worker, "is_hybrid_swa", False):
        return result

    model_runner = tp_worker.model_runner
    from sglang.srt import environ as envs

    use_full_token_pool = (
        any(
            cls.__name__ == "DeepSeekV4TokenToKVPool"
            for cls in type(queue.token_to_kv_pool).__mro__
        )
        and envs.SGLANG_DSV4_PD_PREFILL_USE_FULL_TOKEN_POOL.get()
    )
    if use_full_token_pool:
        queue.max_total_num_tokens = model_runner.max_token_pool_size
        logger.info(
            "DeepSeek-V4 PD prefill admission uses full token pool capacity: %d",
            queue.max_total_num_tokens,
        )
        return result

    swa_limit = getattr(model_runner, "swa_max_total_num_tokens", None)
    if swa_limit is not None:
        queue.max_total_num_tokens = min(queue.max_total_num_tokens, swa_limit)
    return result


def hcu_get_mha_kv_ptrs_with_pp(
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
            + dst_k_all[total_main_layers + draft_start : total_main_layers + draft_end]
        )
        dst_v_ptrs = (
            dst_v_all[main_start:main_end]
            + dst_v_all[total_main_layers + draft_start : total_main_layers + draft_end]
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
                dst_num_total_layers + start_layer : dst_num_total_layers + end_layer
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
            and all(layer_id >= start_layer for layer_id in full_attention_layer_ids)
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
                configured_end_layer if configured_end_layer is not None else end_layer
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
                dst_num_total_layers
                + compact_start_layer : dst_num_total_layers
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
        dst_v_ptrs = dst_kv_ptrs[v_ptr_offset + start_layer : v_ptr_offset + end_layer]
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


def around_get_mha_kv_ptrs_with_pp(
    original: Callable[..., Any],
    self: Any,
    src_kv_ptrs: List[int],
    dst_kv_ptrs: List[int],
):
    if not is_dcu():
        return original(self, src_kv_ptrs, dst_kv_ptrs)
    if not hasattr(self, "model_config"):
        from sglang.srt.configs.model_config import ModelConfig

        self.model_config = ModelConfig.from_server_args(self.server_args)
    return hcu_get_mha_kv_ptrs_with_pp(self, src_kv_ptrs, dst_kv_ptrs)


def register_hooks() -> None:
    from hcu_sglang.disagg.integration import register_hooks as register_pd
    register_pd()
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.disaggregation.prefill.PrefillBootstrapQueue.__init__",
        after_prefill_bootstrap_queue_init,
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.disaggregation.common.conn."
        "CommonKVManager.get_mha_kv_ptrs_with_pp",
        around_get_mha_kv_ptrs_with_pp,
        HookType.AROUND,
    )
