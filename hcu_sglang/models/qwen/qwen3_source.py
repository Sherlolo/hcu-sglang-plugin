# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Hygon modifications to this file are licensed under the Apache License,
# Version 2.0 (the "License"); you may not use these modifications except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from qwen2.py
from __future__ import annotations
import types
import torch
import sglang.srt.models.qwen3 as upstream
from sglang.srt.runtime_context import get_server_args
from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from hcu_sglang.utils.detection import is_hcu

globals().update({name: value for name, value in vars(upstream).items()
                  if not name.startswith("__")})
_is_hcu = is_hcu()
_use_fused_qwen_bailing_rotary = get_bool_env_var("SGLANG_USE_FUSED_RMS_ROTARY")
_has_fused_qk_norm_mrope = False
_has_fused_qk_norm_rope = False
if _use_aiter:
    try:
        from aiter import fused_qk_norm_mrope_3d_cache_pts_quant_shuffle

        _has_fused_qk_norm_mrope = True
        logger.info("aiter fused_qk_norm_mrope_3d kernel available")
    except ImportError:
        pass
    try:
        from aiter import fused_qk_norm_rope_cache_pts_quant_shuffle

        _has_fused_qk_norm_rope = True
        logger.info("aiter fused_qk_norm_rope kernel available")
    except ImportError:
        pass

if _is_hcu and _use_fused_qwen_bailing_rotary:
    from lightop import rms_rotary_embedding_fuse_with_kv_store

def aiter_fused_qk_norm_rope_set_kv(
    qkv: torch.Tensor,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    rotary_emb: RotaryEmbedding,
    positions: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> torch.Tensor:
    """QK-norm + RoPE (1D, or 3D mRoPE for MRotaryEmbedding) + KV cache write in
    one aiter HIP kernel.

    ``qkv`` is the (num_tokens, (num_heads + 2 * num_kv_heads) * head_dim) QKV
    projection; ``k_cache`` / ``v_cache`` are the NHD (slots, num_kv_heads,
    head_dim) pool buffers. ``k_scale`` / ``v_scale`` must be CPU tensors: the
    kernel reads them on the host. Returns q as (num_tokens, num_heads * head_dim).
    """
    num_tokens = qkv.shape[0]
    q_out = torch.empty(
        num_tokens, num_heads, head_dim, dtype=qkv.dtype, device=qkv.device
    )
    kernel_kwargs = dict(
        qkv=qkv.view(num_tokens, -1, head_dim),
        qw=q_norm.weight,
        kw=k_norm.weight,
        cos_sin=rotary_emb.cos_sin_cache_as(qkv.dtype),
        positions=positions,
        num_tokens=num_tokens,
        num_heads_q=num_heads,
        num_heads_k=num_kv_heads,
        num_heads_v=num_kv_heads,
        head_size=head_dim,
        is_neox_style=rotary_emb.is_neox_style,
        eps=q_norm.variance_epsilon,
        q_out=q_out,
        # The kernel reads cache strides as (block, token, head); an NHD buffer
        # is one-slot blocks addressed with block_size=0.
        k_cache=k_cache.unsqueeze(1),
        v_cache=v_cache.unsqueeze(1),
        slot_mapping=out_cache_loc,
        per_tensor_k_scale=k_scale,
        per_tensor_v_scale=v_scale,
        k_out=None,
        v_out=None,
        return_kv=False,
        use_shuffle_layout=False,
        block_size=0,
        x=0,
    )
    if isinstance(rotary_emb, MRotaryEmbedding):
        fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(
            **kernel_kwargs,
            mrope_section_=rotary_emb.mrope_section,
            is_interleaved=rotary_emb.mrope_interleaved,
        )
    else:
        fused_qk_norm_rope_cache_pts_quant_shuffle(**kernel_kwargs)
    return q_out.view(num_tokens, -1)

class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        start_layer: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = None,
        attention_bias: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        torch.nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.start_layer = start_layer
        self.total_num_heads = num_heads
        attn_tp_size = get_parallel().attn_tp_size

        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        norm_kwargs = (
            dict(
                weight_dtype=torch.float32,
                cast_x_before_out_mul=True,
            )
            if get_exec().deterministic.rl_on_policy_target is not None
            else {}
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps, **norm_kwargs)

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            parallel_group="attn_tp",
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            parallel_group="attn_tp",
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )
        self.alt_stream = alt_stream
        self.page_size = 64
        self.layer_id = layer_id
        if get_server_args().kv_cache_dtype == "fp8_e4m3":
            self.kv_cache_dtype = torch.float8_e4m3fn
        elif get_server_args().kv_cache_dtype == "fp8_e5m2":
            self.kv_cache_dtype = torch.float8_e5m2
        elif get_server_args().kv_cache_dtype in ("bf16", "bfloat16"):
            self.kv_cache_dtype = torch.bfloat16

        self.use_fused_qk_norm_mrope = (
            _has_fused_qk_norm_mrope
            and isinstance(self.rotary_emb, MRotaryEmbedding)
            and getattr(self.rotary_emb, "mrope_section", None) is not None
        )
        # The 1D kernel is instantiated for these head sizes, indexes
        # cos_sin_cache as [pos, rotary_dim] and takes one eps for both norms;
        # scaled-RoPE subclasses stay on the unfused path.
        self.use_fused_qk_norm_rope = (
            _has_fused_qk_norm_rope
            and type(self.rotary_emb) is RotaryEmbedding
            and self.rotary_emb.rotary_dim == self.head_dim
            and self.head_dim in (64, 128, 256, 512)
            and self.q_norm.variance_epsilon == self.k_norm.variance_epsilon
        )
        if self.use_fused_qk_norm_mrope or self.use_fused_qk_norm_rope:
            # Scale tensors MUST stay on CPU: the C++ kernel uses .item<float>()
            # which triggers hipMemcpy D2H + sync on CUDA tensors, breaking graph capture.
            # Explicit device='cpu' is required because SGLang constructs models inside
            # a `with torch.device('cuda'):` context that changes the default device.
            self._fused_k_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")
            self._fused_v_scale = torch.tensor(1.0, dtype=torch.float32, device="cpu")

    def forward_prepare_native(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if _is_hcu and _use_fused_qwen_bailing_rotary:
            # Fused RMSNorm + RoPE + kv_store path through custom op.
            cos_sin_cache = self.rotary_emb.cos_sin_cache
            if cos_sin_cache.device != q.device or cos_sin_cache.dtype != q.dtype:
                cos_sin_cache = cos_sin_cache.to(
                    q.device, dtype=q.dtype, non_blocking=True
                )
                # Persist the converted cache so we don't re-copy/re-allocate
                # on every forward when the original buffer starts on CPU.
                self.rotary_emb.cos_sin_cache = cos_sin_cache
            k_buffer, v_buffer = get_token_to_kv_pool().get_kv_buffer(self.layer_id)

            q, k, v = rms_rotary_embedding_fuse_with_kv_store(
                positions,
                q,
                k,
                v,
                cos_sin_cache,
                self.head_dim,
                self.page_size,
                k_buffer,
                v_buffer,
                forward_batch.out_cache_loc,
                is_neox=True,
                weight_q=self.q_norm.weight,
                weight_k=self.k_norm.weight,
                output_dtype=self.kv_cache_dtype,
                residual_q=None,
                residual_k=None,
                k_scale=None,
                v_scale=None,
                epsilon=self.q_norm.variance_epsilon,
            )
        else:
            q, k = apply_qk_norm(
                q=q,
                k=k,
                q_norm=self.q_norm,
                k_norm=self.k_norm,
                head_dim=self.head_dim,
                alt_stream=self.alt_stream,
            )
            q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn.layer_id == self.start_layer:
            self.rotary_emb.get_cos_sin_with_position(positions)
        q, k, v = split_qkv_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            eps=self.q_norm.variance_epsilon,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            q_bias=getattr(self.q_norm, "bias", None),
            k_bias=getattr(self.k_norm, "bias", None),
        )
        return q, k, v

    def forward_prepare_aiter_fused(
        self, positions, hidden_states, forward_batch, k_cache, v_cache
    ):
        """Decode-only fused path; KV is already in the paged cache when this
        returns, so the caller must pass save_kv_cache=False to attn."""
        qkv, _ = self.qkv_proj(hidden_states)
        q = aiter_fused_qk_norm_rope_set_kv(
            qkv,
            self.q_norm,
            self.k_norm,
            self.rotary_emb,
            positions,
            k_cache,
            v_cache,
            forward_batch.out_cache_loc,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self._fused_k_scale,
            self._fused_v_scale,
        )
        return q, None, None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if get_exec().deterministic.rl_on_policy_target is not None:
            hidden_states = hidden_states.bfloat16()

        save_kv_cache = True
        use_aiter_fused = False
        can_try_aiter_fused = (
            (self.use_fused_qk_norm_mrope or self.use_fused_qk_norm_rope)
            and forward_batch.forward_mode.is_decode()
            and get_exec().deterministic.rl_on_policy_target is None
        )
        if can_try_aiter_fused:
            k_cache, v_cache = get_token_to_kv_pool().get_kv_buffer(self.attn.layer_id)
            # The fused write targets the NHD layout in the activation dtype and
            # does not apply per-layer KV scales.
            use_aiter_fused = (
                k_cache.dim() == 3 and k_cache.dtype == hidden_states.dtype
            )

        if use_aiter_fused:
            q, k, v = self.forward_prepare_aiter_fused(
                positions, hidden_states, forward_batch, k_cache, v_cache
            )
            save_kv_cache = False
        elif not _is_npu:
            q, k, v = self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )
        else:
            q, k, v = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        if get_exec().deterministic.rl_on_policy_target is not None:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)

        attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=save_kv_cache)
        output, _ = self.o_proj(attn_output)
        return output
