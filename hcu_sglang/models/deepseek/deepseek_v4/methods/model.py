from hcu_sglang.models.deepseek.current_v4.execution import core as C

__init__ = C.DeepseekV4Model.__init__
_check_late_layer_tail_readers = C.DeepseekV4Model._check_late_layer_tail_readers
_forward_layers_hc_pre_from_prev = C.DeepseekV4Model._forward_layers_hc_pre_from_prev
_forward_layers_tbo = C.DeepseekV4Model._forward_layers_tbo
forward = C.DeepseekV4Model.forward
