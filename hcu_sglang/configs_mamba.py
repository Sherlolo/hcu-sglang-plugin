from __future__ import annotations

import torch
from sglang.srt.configs import mamba_utils as N
from hcu_sglang.utils.detection import is_hcu

Mamba2StateDType = N.Mamba2StateDType
Mamba2StateShape = N.Mamba2StateShape
Mamba2CacheParams = N.Mamba2CacheParams
KimiLinearStateShape = N.KimiLinearStateShape
KimiLinearCacheParams = N.KimiLinearCacheParams
BaseLinearStateParams = N.BaseLinearStateParams
mamba2_state_dtype = N.mamba2_state_dtype


class BaseLinearStateMethods:
    @property
    def use_hcu_kda(self) -> bool:
        from hcu_sglang.utils.detection import is_hcu

        return self.is_kda and is_hcu()

    def replayssm_ring_bytes_per_req(self, record_len: int) -> int:
        """ReplaySSM spec-verify scratch bytes across all layers.

        GDN keeps compact d/k/g plus low parts for the activation-dtype d/k
        rings. KDA keeps its raw-input fold window and d/k rings.
        """
        hv, v_dim, k_dim = self.shape.temporal
        h_k = self.shape.num_k_heads_per_tp
        conv_b = self.dtype.conv.itemsize
        fp32_b = 4
        if self.is_kda:
            per_layer = (
                hv * record_len * v_dim * conv_b  # rawv
                + h_k * record_len * k_dim * conv_b  # rawk
                + hv * record_len * fp32_b  # beta
                + hv * record_len * k_dim * fp32_b  # vector g
            )
            if not self.use_hcu_kda:
                per_layer += (
                    hv * record_len * v_dim * conv_b + h_k * record_len * k_dim * conv_b
                )
        else:
            per_layer = (
                hv * record_len * v_dim * conv_b  # d
                + h_k * record_len * k_dim * conv_b  # normalized k
                + hv * record_len * fp32_b  # scalar g
            )
            if self.dtype.conv != torch.float32:
                per_layer += (
                    hv * record_len * v_dim * conv_b  # d low part
                    + h_k * record_len * k_dim * conv_b  # normalized-k low part
                )
        return per_layer * len(self.layers)

def around_replayssm_ring_bytes_per_req(original, self, record_len):
    if not is_hcu():
        return original(self, record_len)
    return BaseLinearStateMethods.replayssm_ring_bytes_per_req(self, record_len)


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    N.BaseLinearStateParams.use_hcu_kda = BaseLinearStateMethods.__dict__["use_hcu_kda"]
    HookRegistry.register("sglang.srt.configs.mamba_utils.BaseLinearStateParams.replayssm_ring_bytes_per_req", around_replayssm_ring_bytes_per_req, HookType.AROUND)
    from hcu_sglang import configs_mamba_glm
    configs_mamba_glm.register_hooks()
