"""SGLang DCU general plugin - model and layer adaptations."""

import logging

logger = logging.getLogger(__name__)


def register() -> None:
    """Entry point for sglang.srt.plugins (called by load_plugins)."""
    from hcu_sglang.compat.aiter import install_aiter_compat

    install_aiter_compat()

    from hcu_sglang.environ import register_envs

    register_envs()

    from hcu_sglang import (
        batch_overlap,
        configs,
        disaggregation,
        distributed,
        kernel_compat,
        multimodal_processor,
        quick_all_reduce,
        server_args,
        server_args_hooks,
        speculative_compat,
    )
    from hcu_sglang import jit_overlay_loader
    from hcu_sglang.layers.attention import (
        base_backend,
        debug_flash_mla_adapter,
        dsa_kernels,
        dsa_topk,
        dsv4_compressor,
        flashattention_backend,
        triton_hcu,
    )
    from hcu_sglang.layers.attention import hcu_mla_registry
    from hcu_sglang.layers.attention.fla import (
        layernorm_gated,
        linear_attention,
    )
    from hcu_sglang.layers.attention.linear import gdn_backend
    from hcu_sglang.layers import norm_activation, quantization, linear_fusion, layer_boundary, rotary
    from hcu_sglang.layers.moe import aiter_w4a8, lightop_runner, mega_moe, topk
    from hcu_sglang.mem_cache import allocation as mem_cache_allocation
    from hcu_sglang.mem_cache import (
        dsa,
        dsv4,
        hcu_host_pool,
        hcu_mha_pool,
        kv_cache_dtype,
        mla,
        swa,
    )
    from hcu_sglang.models import bailing_moe
    from hcu_sglang.models import dspark
    from hcu_sglang.models import hunyuan
    from hcu_sglang.models import glm4_moe
    from hcu_sglang.models import minimax_m2
    from hcu_sglang.models import utils as model_utils
    from hcu_sglang.multimodal_gen import attention as multimodal_attention
    from hcu_sglang.multimodal_gen import config_compat as multimodal_config
    from hcu_sglang.multimodal_gen import minimax_h3
    from hcu_sglang.multimodal_gen import platform as multimodal_platform
    from hcu_sglang.multimodal_gen import profiling
    from hcu_sglang.multimodal_gen import qknorm_rope
    from hcu_sglang.multimodal_gen import wan_sla
    from hcu_sglang.models.deepseek import register_hooks as register_deepseek
    from hcu_sglang.models.qwen import qwen2, qwen3, qwen3_5, qwen3_moe

    jit_overlay_loader.register_hooks()
    configs.register_hooks()
    server_args.register()
    server_args_hooks.register()
    speculative_compat.register_hooks()
    multimodal_processor.register_hooks()
    batch_overlap.register_hooks()
    quick_all_reduce.register_hooks()
    distributed.register_hooks()
    disaggregation.register_hooks()
    kernel_compat.register_hooks()
    hcu_mla_registry.register()
    base_backend.register_hooks()
    flashattention_backend.register()
    debug_flash_mla_adapter.register_hooks()
    dsa_kernels.register_hooks()
    dsa_topk.register_hooks()
    dsv4_compressor.register_hooks()
    triton_hcu.register_hooks()
    gdn_backend.register()
    linear_attention.register()
    layernorm_gated.register()
    topk.register()
    aiter_w4a8.register_hooks()
    lightop_runner.register_hooks()
    mega_moe.register_hooks()
    quantization.register_hooks()
    norm_activation.register_hooks()
    linear_fusion.register_hooks()
    layer_boundary.register_hooks()
    rotary.register_hooks()
    mem_cache_allocation.register_hooks()
    kv_cache_dtype.register_hooks()
    swa.register_hooks()
    hcu_host_pool.register_hooks()
    hcu_mha_pool.register_hooks()
    mla.register_hooks()
    dsa.register_hooks()
    dsv4.register_hooks()
    bailing_moe.register()
    dspark.register_hooks()
    hunyuan.register_hooks()
    glm4_moe.register_hooks()
    minimax_m2.register_hooks()
    multimodal_platform.register_hooks()
    multimodal_config.register_hooks()
    profiling.register_hooks()
    multimodal_attention.register_hooks()
    minimax_h3.register_hooks()
    qknorm_rope.register_hooks()
    wan_sla.register_hooks()
    register_deepseek()
    model_utils.register_hooks()
    qwen2.register()
    qwen3.register()
    qwen3_moe.register()
    qwen3_5.register()
    logger.info("[hcu_sglang] optimization hooks registered")
