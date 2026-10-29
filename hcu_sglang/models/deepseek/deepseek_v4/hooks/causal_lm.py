from hcu_sglang.utils.detection import is_dcu
from hcu_sglang.models.deepseek.hook_utils import register_method_replaces_from_path


def register_hooks():
    if not is_dcu():
        return 0
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
    from hcu_sglang.models.deepseek.current_v4.execution import core
    if not hasattr(DeepseekV4ForCausalLM, "logits_from_body_output"):
        DeepseekV4ForCausalLM.logits_from_body_output = core.DeepseekV4ForCausalLM.logits_from_body_output
    return register_method_replaces_from_path(
        "sglang.srt.models.deepseek_v4", "DeepseekV4ForCausalLM", ('__init__', 'pad_input_ids', 'set_dspark_layers_to_capture', 'prepare_language_model_inputs', 'forward', 'post_load_weights', '_prewarm_mhc_kernels', 'load_weights'),
        "hcu_sglang.models.deepseek.deepseek_v4.methods.causal_lm",
    )
