"""Register complete current MQA classes and their DSpark subclass."""
from hcu_sglang.utils.detection import is_dcu


def register_hooks():
    if not is_dcu():
        from hcu_sglang.models.deepseek.hook_utils import register_method_replaces_from_path
        return register_method_replaces_from_path('sglang.srt.models.deepseek_v4','MQALayer',
            ('__init__','_compute_q_b','_forward_prepare','forward'),
            'hcu_sglang.models.deepseek.deepseek_v4.methods.mqa')
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from hcu_sglang.models.deepseek.current_v4 import attention, dspark
    from hcu_sglang.models.deepseek.current_v4.hcu_backend import integration
    for name in ['MqaAttentionBase','MQALayer']:
        HookRegistry.register('sglang.srt.models.deepseek_v4.'+name,getattr(attention,name),HookType.REPLACE)
    HookRegistry.register('sglang.srt.models.deepseek_v4_dspark.DSparkAttention',dspark.DSparkAttention,HookType.REPLACE)
    return 3+integration.register_hooks()
