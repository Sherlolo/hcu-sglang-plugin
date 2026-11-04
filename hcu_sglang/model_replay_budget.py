# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
from sglang.srt.arg_groups import model_hook as N
from hcu_sglang.utils.detection import is_hcu

_REGISTERED = False


def capability_replay_budget(server_args, previous):
    result = previous(server_args)
    if not is_hcu():
        return result
    cfg = N.resolving_view(server_args)
    if N.parse_connector_type(cfg.model_path) == N.ConnectorType.INSTANCE:
        return result
    hf_config = N.model_config_of(server_args).hf_config
    if getattr(hf_config, "model_type", None) == "deepseek_v41":
        from hcu_sglang.replay_validation import (
            validate_deepseek_v41_encoder_replay_budget,
        )

        validate_deepseek_v41_encoder_replay_budget(server_args)
    elif cfg.enable_encoder_swa_bounded_replay:
        raise ValueError("--enable-encoder-swa-bounded-replay requires DeepSeek-V4.1")
    return result


def register_hooks():
    global _REGISTERED
    if not is_hcu() or _REGISTERED:
        return
    from sglang.srt.arg_groups.resolution_hooks import register_resolution_hook
    register_resolution_hook("handle_model_capability_adjustments")(capability_replay_budget)
    _REGISTERED = True
