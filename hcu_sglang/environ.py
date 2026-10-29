"""Install HCU/DAS environment fields into official SGLang."""

from __future__ import annotations

from typing import Any

from sglang.srt.environ import EnvBool, EnvBoolWithAlias, EnvFloat, EnvInt, EnvStr, Envs, _default_hip
from hcu_sglang.utils.detection import is_dcu


_ENV_SPECS: dict[str, tuple[type, Any]] = {

    'SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL': (EnvBool, True),
    'SGLANG_USE_FUSED_SILU_MUL_FP8_QUANT': (EnvBool, False),
    'SGLANG_AITER_INTREE_GEMM_TUNE': (EnvBool, True),
    'SGLANG_AITER_INTREE_MHC_TILE_M16': (EnvBool, True),
    'SGLANG_DSV4_DECODER_TAIL_GRAPH_MAX_ROWS': (EnvInt, 2048),
    'SGLANG_DSV4_DENSE_MXSCALE': (EnvBool, False),
    'SGLANG_DSV4_FUSE_MHC_REPEAT_CP_SPLIT': (EnvBool, False),
    "SGLANG_W4A8_TPMOE_BACKEND": (EnvStr, "auto"),
    "SGLANG_FORCE_AITER_MOE_C": (EnvBool, False),
    "SGLANG_DSPARK_FORCE_W4A8_TPMOE_BACKEND": (EnvStr, None),
    "SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB": (EnvFloat, None),
    "SGLANG_HCU_TOPK_V2_PAGED": (EnvBool, False),
    'SGLANG_DSV4_COMPRESS_RLC': (EnvBool, False),
    'SGLANG_DSV4_FP4_LOGITS_BUDGET_MB': (EnvInt, 2048),
    'SGLANG_DSV4_PAGED_MQA_LOGITS_FREE_MEM_FRACTION': (EnvFloat, None),
    'SGLANG_HIP_FP4_INDEXER_ROWGROUP': (EnvBool, _default_hip),
    'SGLANG_MQA_PERSISTENT': (EnvBool, False),
    'SGLANG_NSA_INDEX_Q_INT8': (EnvBool, False),
    'SGLANG_OPT_DSV41_BF16_INDEX_LOGITS': (EnvBool, False),
    'SGLANG_PREFILL_CP_MIN_TOKENS_PER_SEQUENCE': (EnvInt, 0),

    "SGLANG_DSV4_HCU_USE_BF16_FLASH_MLA": (EnvBool, False),
    "SGLANG_DSV4_HCU_USE_LIGHTOP_BF16_GATHER": (EnvBool, False),
    "SGLANG_LIGHTOP_DEQUANTIZE_K_CACHE_PAGED": (EnvBool, False),
    "SGLANG_DSV4_HCU_INT8_INDEX_K_CACHE": (EnvBool, False),
    "SGLANG_HIP_DSV41_INDEX_K_PAGE8": (EnvBool, False),
    "SGLANG_HCU_HC_MIX_STATS_OPT": (EnvBool, True),
    "SGLANG_USE_LIGHTOP_PAGED_MQA_LOGITS_FP4": (EnvBool, True),
    "SGLANG_USE_LIGHTOP_DSV41_PAGED_INDEXER": (EnvBool, True),
    "SGLANG_USE_LIGHTOP_DSV41_CANDIDATE_BLOCKS": (EnvBool, True),
    "SGLANG_HCU_PREFILL_PAGED_SPARSE": (EnvBool, True),
    "SGLANG_HCU_PREFILL_SPARSE_FUSION": (EnvBool, True),
    "SGLANG_DSV41_SKIP_MID_CHUNK_DECODER": (EnvBool, True),
    "SGLANG_HCU_ENABLE_DSV41_W4A8_ENGRAM_LOAD_OPTIMIZATION": (EnvBool, False),
    "SGLANG_DSV41_PAGED_MQA_LOGITS_FREE_MEM_FRACTION": (EnvFloat, None),
    "SGLANG_CUTLASS_MOE": (EnvBool, False),
    "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": (EnvBool, False),
    "SGLANG_DSV4_FIX_TP_ATTN_A2A_SCATTER": (EnvBool, True),
    "SGLANG_DSV4_MHC_PREWARM": (EnvBool, True),
    "SGLANG_DSV4_PD_PREFILL_USE_FULL_TOKEN_POOL": (EnvBool, True),
    "SGLANG_DSV4_SPLIT_PREFILL_DECODE_MLA": (EnvBool, False),
    "SGLANG_ENABLE_CUSTOM_ALL_REDUCE_V2_MULTINODE": (EnvBool, False),
    "SGLANG_ENABLE_DP_ATTENTION_LOCAL_CONTROL_BROADCAST": (EnvBool, False),
    "SGLANG_ENABLE_EPLB_BALANCEDNESS_METRIC": (EnvBool, False),
    "SGLANG_ENABLE_NVFP4_GEMM_SWIGLU_FUSION": (EnvBool, True),
    "SGLANG_FIX_MTP_HC_HIDDEN": (EnvBool, False),
    "SGLANG_HACK_SKIP_FP4_FP8_GEMM": (EnvBool, False),
    "SGLANG_HCU_MEGA_MOE_RUNTIME": (EnvStr, "deep_gemm"),
    "SGLANG_K3_DENSE_MLP_ATTN_TP": (EnvBool, False),
    "SGLANG_K3_SHARED_EXPERTS_ATTN_TP": (EnvBool, False),
    "SGLANG_KV_LAYOUT_HCU_FA": (EnvBool, True),
    "SGLANG_LIGHTOP_TOPK": (EnvBool, False),
    "SGLANG_NPU_FUSED_MOE_MODE": (EnvInt, 1),
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS": (EnvBool, False),
    "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND": (EnvBool, False),
    "SGLANG_OPT_DPSK_V4_RADIX": (EnvBool, True),
    "SGLANG_OPT_FIX_MEGA_MOE_MEMORY": (EnvBool, False),
    "SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN": (EnvBool, False),
    "SGLANG_OPT_SWIGLU_CLAMP_FUSION": (EnvBool, True),
    "SGLANG_OPT_USE_AITER_MHC_POST": (EnvBool, True),
    "SGLANG_OPT_USE_AITER_MHC_PRE": (EnvBool, True),
    "SGLANG_OPT_USE_COMPRESSOR_V2": (EnvBool, True),
    "SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE": (EnvBool, False),
    "SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL": (EnvBool, True),
    "SGLANG_OPT_USE_FUSED_COMPRESS": (EnvBool, False),
    "SGLANG_OPT_USE_FUSED_COMPRESS_TRITON": (EnvBool, False),
    "SGLANG_OPT_USE_FUSED_STORE_CACHE": (EnvBool, True),
    "SGLANG_OPT_USE_JIT_EP_ACTIVATION": (EnvBool, True),
    "SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK": (EnvBool, True),
    "SGLANG_OPT_USE_JIT_NORM": (EnvBool, True),
    "SGLANG_OPT_USE_OLD_COMPRESSOR": (EnvBool, False),
    "SGLANG_OPT_USE_TRITON_FUSED_MHC": (EnvBool, True),
    "SGLANG_OPT_USE_TRITON_SWA_PREPARE": (EnvBool, True),
    "SGLANG_PREP_IN_CUDA_GRAPH": (EnvBool, True),
    "SGLANG_ROCM_USE_AITER_TILELANG_MHC": (EnvBool, False),
    "SGLANG_TOPK_TRANSFORM_512_TORCH": (EnvBool, False),
    "SGLANG_TRTLLM_GEN_MOE_CUBIN_POOL": (EnvStr, None),
    "SGLANG_USE_AITER_LINEAR_ATTN": (EnvBool, False),
    "SGLANG_USE_DPSKV4_LIGHTOP_QUANT_K_CACHE": (EnvBool, False),
    "SGLANG_USE_DPSKV4_LIGHTOP_RMSNORM": (EnvBool, False),
    "SGLANG_USE_FAST_HADAMARD_TRANSFORM": (EnvBool, False),
    "SGLANG_USE_FUSED_BAILING_RMS_QUANT": (EnvBool, False),
    "SGLANG_USE_FUSED_BAILING_SILU_MUL_FP8_QUANT": (EnvBool, False),
    "SGLANG_USE_FUSED_DPSKV4_QNORM_ROPE_KV_ROPE_QUANT": (EnvBool, False),
    "SGLANG_USE_FUSED_DPSKV4_SILU_MUL_FP8_QUANT": (EnvBool, False),
    "SGLANG_USE_FUSED_MLA_CAT": (EnvBool, False),
    "SGLANG_USE_FUSED_RESHAPE_TO_FLOAT": (EnvBool, False),
    "SGLANG_USE_FUSED_RMS_QUANT": (EnvBool, False),
    "SGLANG_USE_FUSED_RMS_ROTARY": (EnvBool, False),
    "SGLANG_USE_FUSED_SILU_MUL_QUANT": (EnvBool, False),
    "SGLANG_USE_FUSED_TOPK_SOFTMAX": (EnvBool, False),
    "SGLANG_USE_LIGHTOP": (EnvBool, False),
    "SGLANG_USE_LIGHTOP_GROUP_FP8_QUANT": (EnvBool, False),
    "SGLANG_USE_LIGHTOP_MOE_SUM_MUL_ADD": (EnvBool, False),
    "SGLANG_USE_LINEAR_BF16_FP32_USE_BLASLT": (EnvBool, False),
    "SGLANG_USE_OPT_CAT": (EnvBool, False),
}

_HCU_DEFAULT_OVERRIDES: dict[str, tuple[type, Any]] = {
    "SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD": (EnvInt, 0),
    "SGLANG_HACK_FLASHMLA_BACKEND": (EnvStr, "kernel"),
    "SGLANG_OPT_FP8_WO_A_GEMM": (EnvBool, False),
    "SGLANG_OPT_FUSE_MHC_POST_PRE": (EnvBool, False),
}


def _install_field(name: str, field_type: type, default: Any, *, replace: bool) -> None:
    if not replace and hasattr(Envs, name):
        return
    field = field_type(default)
    field.name = name
    setattr(Envs, name, field)


def register_envs() -> None:
    """Install fork-only fields and HCU-specific default overrides."""
    for name, alias in {
        "SGLANG_DSA_HCU_LIGHTOP_MASK_TOPK": "SGLANG_USE_LIGHTOP_MASK_TOPK",
        "SGLANG_DSA_HCU_REUSE_SORTED_TOPK": "SGLANG_NSA_HCU_REUSE_SORTED_TOPK",
    }.items():
        if not hasattr(Envs, name):
            field = EnvBoolWithAlias(False, deprecated_name=alias)
            field.name = name
            setattr(Envs, name, field)

    for name, (field_type, default) in _ENV_SPECS.items():
        _install_field(name, field_type, default, replace=False)

    if is_dcu():
        for name, (field_type, default) in _HCU_DEFAULT_OVERRIDES.items():
            _install_field(name, field_type, default, replace=True)
