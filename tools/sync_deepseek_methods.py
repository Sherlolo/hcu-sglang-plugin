#!/usr/bin/env python3
"""Re-extract precise DeepSeek method bodies from the current SGLang fork."""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path


V2_CONSTANT_MAP = {
    "_is_hcu": "C._is_dcu",
    "_is_dcu": "C._is_dcu",
    **{
        name: f"C.{name}"
        for name in (
            "_is_hip",
            "_is_cuda",
            "_is_npu",
            "_is_cpu",
            "_is_cpu_amx_available",
            "_device_sm",
            "_is_gfx95_supported",
            "_is_sbo_enabled",
            "_use_aiter",
            "_use_aiter_gfx95",
            "_use_lightop_moe_sum_mul_add",
            "_use_fused_silu_mul_quant",
            "_use_opt_cat_decode",
            "_use_fused_mla_cat",
            "_use_fused_rmsnorm_rope",
            "_use_fused_rms_quant",
            "_rms_quant_path",
            "FORWARD_ABSORB_CORE_ATTENTION_BACKENDS",
            "dtype_mapping",
            "fused_rms_norm_rope_contiguous",
        )
    },
}

V4_CONSTANT_MAP = {
    "_is_hcu": "C._is_dcu",
    "_is_dcu": "C._is_dcu",
    **{
        name: f"C.{name}"
        for name in (
            "_use_dpskv4_lightop_rmsnorm",
            "_use_fused_qnorm_rope_kv_rope_quant",
            "_use_aiter_tilelang_mhc",
            "op",
        )
    },
}

V2_HEADER = '''\
"""DCU DeepSeek V2 fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from sglang_dcu_plugin.models.deepseek.deepseek_v2 import constants as C
from sglang_dcu_plugin.models.deepseek.deepseek_v2.methods._runtime import *  # noqa: F403

'''

V4_HEADER = '''\
"""DCU DeepSeek V4 fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from sglang_dcu_plugin.models.deepseek.deepseek_v4 import constants as C
from sglang_dcu_plugin.models.deepseek.deepseek_v4.methods._runtime import *  # noqa: F403

'''

COMMON_HEADER = '''\
"""DCU deepseek_common fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from sglang_dcu_plugin.models.deepseek.deepseek_common import constants as C
from sglang_dcu_plugin.models.deepseek.deepseek_common.methods._runtime import *  # noqa: F403

'''

NEXTN_HEADER = '''\
"""DCU deepseek_nextn fork methods (auto-extracted for plugin hooks)."""

from __future__ import annotations

from sglang_dcu_plugin.models.deepseek.deepseek_nextn.methods._runtime import *  # noqa: F403

'''


def extract_method(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method_node = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    segment = ast.get_source_segment(source, method_node)
    if segment is None:
        raise RuntimeError(f"Unable to extract {class_name}.{method_name} from {path}")
    return segment


def transform(source: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        source = re.sub(rf"\b{re.escape(old)}\b", new, source)
    return source.rstrip()


def write_methods(
    destination: Path,
    header: str,
    specs: list[tuple[Path, str, str]],
    replacements: dict[str, str],
    check: bool,
) -> bool:
    content = header + "\n\n".join(
        transform(extract_method(path, class_name, method_name), replacements)
        for path, class_name, method_name in specs
    ) + "\n"
    previous = destination.read_text(encoding="utf-8") if destination.exists() else None
    changed = previous != content
    if changed and not check:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
    print(f"{'would update' if check and changed else 'updated' if changed else 'unchanged'} {destination}")
    return changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fork-python", type=Path, required=True)
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    models = args.fork_python / "sglang/srt/models"
    plugin_models = args.plugin_root / "sglang_dcu_plugin/models"
    v2_source = models / "deepseek_v2.py"
    v4_source = models / "deepseek_v4.py"
    forward_methods = models / "deepseek_common/attention_forward_methods"

    jobs = [
        (
            plugin_models / "deepseek/deepseek_v2/methods/mlp.py",
            V2_HEADER,
            [(v2_source, "DeepseekV2MLP", "forward")],
            V2_CONSTANT_MAP,
        ),
        (
            plugin_models / "deepseek/deepseek_v2/methods/moe.py",
            V2_HEADER,
            [
                (v2_source, "DeepseekV2MoE", name)
                for name in (
                    "__init__",
                    "forward",
                    "forward_normal",
                    "forward_deepep",
                    "forward_normal_dual_stream",
                    "_forward_shared_experts",
                )
            ],
            V2_CONSTANT_MAP,
        ),
        (
            plugin_models / "deepseek/deepseek_v2/methods/decoder.py",
            V2_HEADER,
            [
                (v2_source, "DeepseekV2DecoderLayer", name)
                for name in ("__init__", "forward")
            ],
            V2_CONSTANT_MAP,
        ),
        (
            plugin_models / "deepseek/deepseek_v2/methods/attention.py",
            V2_HEADER,
            [
                *[
                    (v2_source, "DeepseekV2AttentionMLA", name)
                    for name in ("__init__", "prepare_qkv_latent")
                ],
                *[
                    (forward_methods / "forward_mla.py", "DeepseekMLAForwardMixin", name)
                    for name in ("forward_absorb_prepare", "forward_absorb_core")
                ],
                *[
                    (forward_methods / "forward_mha.py", "DeepseekMHAForwardMixin", name)
                    for name in (
                        "_set_mla_kv_buffer",
                        "_get_mla_kv_buffer",
                        "_get_mla_kv_buffer_from_fp8_for_dsa",
                        "_concat_and_cast_mha_k",
                        "_chunked_prefix_attn_mha",
                        "forward_normal_chunked_kv_prepare",
                        "forward_normal_chunked_kv_core",
                        "forward_normal_one_shot_prepare",
                        "forward_normal_one_shot_core",
                    )
                ],
                *[
                    (
                        forward_methods / "forward_mla_fused_rope_rocm.py",
                        "DeepseekMLAFusedRopeRocmForwardMixin",
                        name,
                    )
                    for name in (
                        "forward_absorb_fused_mla_rope_prepare",
                        "forward_absorb_fused_mla_rope_core",
                    )
                ],
                *[
                    (
                        forward_methods / "forward_mla_fused_rope_cpu.py",
                        "DeepseekMLACpuForwardMixin",
                        name,
                    )
                    for name in (
                        "forward_absorb_fused_mla_rope_cpu_prepare",
                        "forward_absorb_fused_mla_rope_cpu_core",
                    )
                ],
            ],
            V2_CONSTANT_MAP,
        ),
        *[
            (
                plugin_models / f"deepseek/deepseek_v4/methods/{filename}",
                V4_HEADER,
                [(v4_source, class_name, name) for name in methods],
                V4_CONSTANT_MAP,
            )
            for filename, class_name, methods in (
                ("mqa.py", "MQALayer", ("__init__", "_compute_q_b", "_forward_prepare", "forward")),
                ("decoder.py", "DeepseekV4DecoderLayer", ("hc_pre", "hc_post", "forward")),
                ("model.py", "DeepseekV4Model", ("__init__", "forward")),
                ("causal_lm.py", "DeepseekV4ForCausalLM", ("__init__", "forward", "load_weights", "post_load_weights")),
            )
        ],
        (
            plugin_models / "deepseek/deepseek_common/methods/weight_loader.py",
            COMMON_HEADER,
            [
                (
                    models / "deepseek_common/deepseek_weight_loader.py",
                    "DeepseekV2WeightLoaderMixin",
                    name,
                )
                for name in ("do_load_weights", "post_load_weights")
            ],
            {"_is_hcu": "C._is_dcu", "_is_dcu": "C._is_dcu"},
        ),
        (
            plugin_models / "deepseek/deepseek_nextn/methods/model.py",
            NEXTN_HEADER,
            [
                (models / "deepseek_nextn.py", "DeepseekModelNextN", name)
                for name in ("__init__", "forward")
            ],
            {},
        ),
    ]

    results = [
        write_methods(destination, header, specs, replacements, args.check)
        for destination, header, specs, replacements in jobs
    ]
    changed = any(results)
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
