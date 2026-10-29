"""Current HCU CP validation through the official resolution contract."""
from __future__ import annotations
import sglang.srt.arg_groups.deepseek_v4_hook as upstream
globals().update({name:value for name,value in vars(upstream).items() if not name.startswith('__')})
from hcu_sglang.utils.detection import is_dcu
_is_hcu=is_dcu()

def validate_deepseek_v4_cp(server_args: ServerArgs) -> None:
    """Validate DeepSeek V4 context-parallel configuration."""
    cfg = resolving_view(server_args)
    if not cfg.enable_prefill_cp:
        return
    if cfg.cp_strategy not in ('interleave', 'zigzag'):
        raise ValueError(f'DeepSeekV4 only supports interleave/zigzag CP strategy, got {cfg.cp_strategy}')
    if cfg.cp_strategy == 'zigzag' and (not is_npu()):
        raise ValueError('DeepSeekV4 zigzag CP requires the NPU backend; CUDA/HIP backends reindex with interleave order.')
    if get_platform().is_hip and (not _is_hcu):
        (prefill_backend, decode_backend) = attention_backends_of(resolved_view(server_args))
        if (prefill_backend, decode_backend) != ('dsv4', 'dsv4'):
            raise ValueError(f'DeepSeekV4 prefill CP on HIP requires the dsv4 attention backend for both phases, got prefill={prefill_backend!r}, decode={decode_backend!r}.')
        unsupported = (('multiple nodes', cfg.nnodes > 1), ('DeepSeek-V4.1', model_config_of(server_args).hf_config.model_type == 'deepseek_v41'), ('decoder SWA bounded replay', cfg.enable_decoder_swa_bounded_replay), ('--enable-two-batch-overlap', cfg.enable_two_batch_overlap))
        for (feature, enabled) in unsupported:
            if enabled:
                raise ValueError(f'DeepSeekV4 prefill CP on HIP does not support {feature} yet.')
    attn_dp_size = context_parallel_attn_dp_size(cfg, 'DeepSeek-V4 context parallelism')
    declare_resolution(server_args, 'validate_deepseek_v4_cp', attn_dp_size=attn_dp_size, dp_size=1)
    declare_resolution(server_args, 'validate_deepseek_v4_cp', moe_dense_tp_size=1)
    declare_resolution(server_args, 'validate_deepseek_v4_cp', attn_cp_size=cfg.tp_size // attn_dp_size)
    if not is_npu():
        assert attn_dp_size == 1, 'For round-robin split mode, dp attention is not supported.'
        assert cfg.tp_size <= 8, 'Context parallel only supports single machine (tp_size <= 8). Cross-machine CP has precision issues.'
    supported_a2a_backends = ('none', 'deepep', 'megamoe', 'mori')
    if cfg.moe_a2a_backend not in supported_a2a_backends:
        raise ValueError(f'DeepSeekV4 CP supports moe_a2a_backend in {supported_a2a_backends}, got {cfg.moe_a2a_backend!r}.')
    explicit_hcu_sparse_cp = _is_hcu and envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.is_set() and envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()
    if explicit_hcu_sparse_cp:
        logger.warning('Keeping explicitly enabled HCU sparse prefill with DeepSeekV4 CP.')
    elif model_config_of(server_args).hf_config.model_type != 'deepseek_v41':
        logger.warning('Disabling SGLANG_OPT_FLASHMLA_SPARSE_PREFILL because DeepSeekV4 context parallelism is enabled.')
        envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.set(False)
    logger.warning(f'Enable Context Parallel for DeepSeekV4, strategy={cfg.cp_strategy}, attn_dp_size={attn_dp_size}, moe_dense_tp_size={cfg.moe_dense_tp_size}, attn_cp_size={cfg.attn_cp_size}, ep_size={cfg.ep_size}, tp_size={cfg.tp_size}')
