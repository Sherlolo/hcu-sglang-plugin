"""HCU DeepSeek V4.1 checkpoint normalization for the pinned official API."""
from __future__ import annotations
from hcu_sglang.utils.detection import is_dcu

def _per_channel_fp8_as_compressed_tensors(quant_config):
    """FP8-Channel checkpoints declare {"quant_method": "fp8", "is_per_channel": true}
    and store [out, 1] weight_scale_inv tensors, which the block/tensor Fp8Config
    cannot load. Serve them through the compressed-tensors W8A8-FP8 channel scheme
    (dynamic per-token activations) that the V4 FP8-Channel checkpoints use; the
    loader already maps weight_scale_inv onto its weight_scale parameters."""
    if not (
        isinstance(quant_config, dict)
        and quant_config.get("quant_method") == "fp8"
        and quant_config.get("is_per_channel")
        and quant_config.get("weight_block_size") is None
        and quant_config.get("activation_scheme", "dynamic") == "dynamic"
    ):
        return quant_config
    return {
        "quant_method": "compressed-tensors",
        "format": "float-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "dynamic": False,
                    "strategy": "channel",
                    "group_size": None,
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "float",
                    "symmetric": True,
                    "dynamic": True,
                    "strategy": "token",
                    "group_size": None,
                },
            }
        },
        # wo_a ships in bf16 without a scale.
        "ignore": ["re:.*attn.wo_a.*"],
    }

def after_normalize_deepseek_v41_config(result, values):
    if is_dcu() and "quantization_config" in result:
        result = dict(result)
        result["quantization_config"] = _per_channel_fp8_as_compressed_tensors(
            result["quantization_config"]
        )
    return result


def register_hooks():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register(
        "sglang.srt.configs.deepseek_v41.normalize_deepseek_v41_config",
        after_normalize_deepseek_v41_config,
        HookType.AFTER,
    )
