#!/usr/bin/env python3
"""Re-extract precise DeepSeek method bodies from the current SGLang fork."""

from __future__ import annotations

import argparse
import ast
import sys
import tempfile
import textwrap
import os
from pathlib import Path

if __package__:
    from .plugin_layout import resolve_plugin_package
else:
    from plugin_layout import resolve_plugin_package


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
from __future__ import annotations

from sglang_dcu_plugin.models.deepseek.deepseek_v2 import constants as C
from sglang_dcu_plugin.models.deepseek.deepseek_v2.methods._runtime import *

'''

V4_HEADER = '''\
from __future__ import annotations

from sglang_dcu_plugin.models.deepseek.deepseek_v4 import constants as C
from sglang_dcu_plugin.models.deepseek.deepseek_v4.methods._runtime import *

'''

COMMON_HEADER = '''\
from __future__ import annotations

from sglang_dcu_plugin.models.deepseek.deepseek_common import constants as C
from sglang_dcu_plugin.models.deepseek.deepseek_common.methods._runtime import *

'''

NEXTN_HEADER = '''\
from __future__ import annotations

from sglang_dcu_plugin.models.deepseek.deepseek_nextn.methods._runtime import *

'''


class ExtractionError(ValueError):
    """Pinned source no longer provides the declared extraction contract."""


def extract_method(path: Path, class_name: str, method_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    classes = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    if class_name not in classes:
        raise ExtractionError(
            f"{path}: class {class_name} is absent; available classes: "
            + ", ".join(classes)
            + ". Check class splits/aliases and the source profile; do not guess an implementation."
        )
    node = next((n for n in classes[class_name].body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                 and n.name == method_name), None)
    if node is None:
        raise ExtractionError(f"{path}: method {class_name}.{method_name} is absent")
    for decorator in node.decorator_list:
        if (isinstance(decorator, ast.Name)
                and decorator.id in {"classmethod", "staticmethod", "property"}):
            raise ExtractionError(
                f"{path}: {class_name}.{method_name} uses descriptor "
                f"{decorator.id}; declare descriptor binding before extraction"
            )
    first_line = min([node.lineno] + [n.lineno for n in node.decorator_list])
    return textwrap.dedent("\n".join(
        source.splitlines()[first_line - 1:node.end_lineno]
    ))


def transform(source: str, replacements: dict[str, str],
              owner_class: str | None = None) -> str:
    """Rewrite identifier references and preserve the extracted method's owner.

    Offsets use UTF-8 bytes, as AST column offsets do. Attribute names,
    strings and comments are never treated as global identifier references.
    Repeated calls preserve already-qualified constants.
    """
    tree = ast.parse(source)
    method = next((n for n in tree.body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    if method is None:
        raise ExtractionError("Expected one extracted method")
    first_arg = (method.args.posonlyargs + method.args.args)
    edits = []
    lines = source.encode("utf-8").splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def edit(node, replacement):
        edits.append((starts[node.lineno - 1] + node.col_offset,
                      starts[node.end_lineno - 1] + node.end_col_offset,
                      replacement.encode("utf-8")))

    # Substituting a name shadowed by a parameter/local would change Python
    # scope semantics. Refuse that profile instead of guessing a global.
    local_names = {n.arg for n in ast.walk(method) if isinstance(n, ast.arg)}
    local_names.update(n.id for n in ast.walk(method)
                       if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store))
    conflicts = local_names & replacements.keys()
    if conflicts:
        raise ExtractionError("Constant names are locally bound: " + ", ".join(sorted(conflicts)))

    class References(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            # A nested class has its own __class__/super closure.
            return

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in replacements:
                edit(node, replacements[node.id])
            elif isinstance(node.ctx, ast.Load) and node.id == "__class__":
                if owner_class is None:
                    raise ExtractionError("Extracted __class__ requires an explicit owner")
                edit(node, owner_class)

        def visit_Call(self, node):
            if (isinstance(node.func, ast.Name) and node.func.id == "super"
                    and not node.args and not node.keywords):
                if owner_class is None or not first_arg:
                    raise ExtractionError("Extracted super() requires an explicit owner and receiver")
                edit(node, f"super({owner_class}, {first_arg[0].arg})")
                return
            self.generic_visit(node)

    References().visit(method)
    data = source.encode("utf-8")
    for start, end, replacement in sorted(edits, reverse=True):
        data = data[:start] + replacement + data[end:]
    return data.decode("utf-8").rstrip()


def render_methods(header, specs, replacements):
    content = header + "\n\n".join(
        transform(extract_method(path, class_name, method_name), replacements, class_name)
        for path, class_name, method_name in specs
    ) + "\n"
    ast.parse(content, filename="<generated methods>")
    return content


def plan_updates(jobs):
    # Resolve every source symbol and generated syntax before touching outputs.
    return [(destination, render_methods(header, specs, replacements))
            for destination, header, specs, replacements in jobs]


def apply_update(destination, content, check):
    previous = destination.read_text(encoding="utf-8") if destination.exists() else None
    changed = previous != content
    if changed and not check:
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".method-sync-", dir=destination.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(content)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    print(f"{'would update' if check and changed else 'updated' if changed else 'unchanged'} {destination}")
    return changed


def write_methods(destination, header, specs, replacements, check):
    return apply_update(destination, render_methods(header, specs, replacements), check)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fork-python", type=Path, required=True)
    parser.add_argument("--plugin-root", type=Path, required=True)
    parser.add_argument("--plugin-package", choices=("hcu_sglang", "sglang_dcu_plugin"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    models = args.fork_python / "sglang/srt/models"
    try:
        package = resolve_plugin_package(args.plugin_root, args.plugin_package)
    except ValueError as error:
        parser.error(str(error))
    plugin_models = args.plugin_root / package / "models"
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

    try:
        jobs = [(dest, header.replace("sglang_dcu_plugin", package), specs, mapping)
                for dest, header, specs, mapping in jobs]
        updates = plan_updates(jobs)
    except (ExtractionError, OSError, SyntaxError) as error:
        print(f"error: {error}\nSource preflight failed; no output files were written.", file=sys.stderr)
        return 2
    results = [apply_update(destination, content, args.check)
               for destination, content in updates]
    changed = any(results)
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())
