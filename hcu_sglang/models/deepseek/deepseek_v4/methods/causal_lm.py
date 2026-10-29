from hcu_sglang.models.deepseek.current_v4.execution import core as C

__init__ = C.DeepseekV4ForCausalLM.__init__
pad_input_ids = C.DeepseekV4ForCausalLM.pad_input_ids
set_dspark_layers_to_capture = C.DeepseekV4ForCausalLM.set_dspark_layers_to_capture
prepare_language_model_inputs = C.DeepseekV4ForCausalLM.prepare_language_model_inputs
forward = C.DeepseekV4ForCausalLM.forward
post_load_weights = C.DeepseekV4ForCausalLM.post_load_weights
_prewarm_mhc_kernels = C.DeepseekV4ForCausalLM._prewarm_mhc_kernels
load_weights = C.DeepseekV4ForCausalLM.load_weights
