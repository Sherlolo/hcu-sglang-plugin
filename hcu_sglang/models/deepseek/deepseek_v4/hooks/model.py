from hcu_sglang.utils.detection import is_dcu
from hcu_sglang.models.deepseek.hook_utils import register_method_replaces_from_path


def register_hooks():
    if not is_dcu():
        return 0
    return register_method_replaces_from_path(
        "sglang.srt.models.deepseek_v4", "DeepseekV4Model", ('__init__', '_check_late_layer_tail_readers', '_forward_layers_hc_pre_from_prev', '_forward_layers_tbo', 'forward'),
        "hcu_sglang.models.deepseek.deepseek_v4.methods.model",
    )
