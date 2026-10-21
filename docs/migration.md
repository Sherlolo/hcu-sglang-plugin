# 三方审计、迁移与升级

命令均从本库根目录执行；先将变量替换成真实绝对路径。`PLUGIN` 必须指向外部实现仓，不能指向当前方法库。审计允许脏树并记录 WORKTREE；不得先 reset/clean。回溯对干净树的要求见单独手册。

```bash
OFFICIAL=/path/to/official-sglang
FORK=/path/to/sglang-das
PLUGIN=/path/to/sglang-plugin-das
ARTIFACTS="$PWD/artifacts/migration"
mkdir -p "$ARTIFACTS"
python3 tools/audit_repositories.py \
  --repo "official=$OFFICIAL" --repo "fork=$FORK" --repo "plugin=$PLUGIN" \
  --compare fork=official --output "$ARTIFACTS/repo_audit.json"
```

## 固定基线与生成清单

记录各树 branch、HEAD、dirty 文件、解释器/容器和依赖版本。使用 fork 中已有且与 official SHA 一致的 ref；若没有，可导入本地对象到**新的**专用 ref，不改变工作树：

```bash
# 如该 ref 已存在先核对，不覆盖别人的 ref；不要假设上游分支叫 main。
OFFICIAL_SHA=$(git -C "$OFFICIAL" rev-parse HEAD)
AUDIT_REF="refs/audit/hcu-plugin/$OFFICIAL_SHA"
git -C "$FORK" fetch "$OFFICIAL" "$OFFICIAL_SHA:$AUDIT_REF"
git -C "$FORK" merge-base HEAD "$AUDIT_REF"
python3 tools/build_migration_manifest.py \
  --official-repo "$OFFICIAL" --fork-repo "$FORK" --plugin-repo "$PLUGIN" \
  --official-ref "$AUDIT_REF" --output-dir "$ARTIFACTS/manifest"
```

fetch 写入 fork 的 Git ref/对象，但不改源码。如果需要严格不动来源 Git 元数据，先在独立临时 clone 中执行。工具对 merge-base 之后的 fork/upstream 变化分组，同时追加 `direct_hardware_drift`，避免漏掉继承自旧基线、被当前 upstream 删除的硬件行为。硬件关键词只帮助找候选，不能据此排除无关键词的 cache/scheduler/模型改动。

## 运行时目标与处置台账

在指定 official/plugin 运行环境中，从顶层入口导出目标（会执行 register；先确认来源导入路径）：

```bash
PYTHONPATH="$OFFICIAL/python:$PLUGIN" python3 - <<'TARGETS'
from pathlib import Path
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang_dcu_plugin import register
HookRegistry.reset()
register()
Path('artifacts/migration/registered_hook_targets.txt').write_text(
    '\n'.join(sorted(HookRegistry._hooks)) + '\n')
TARGETS
cp templates/disposition_overrides.json "$ARTIFACTS/overrides.json"
python3 tools/build_disposition_ledger.py \
  --manifest "$ARTIFACTS/manifest/migration_manifest.json" \
  --official-repo "$OFFICIAL" --fork-repo "$FORK" --plugin-repo "$PLUGIN" \
  --hook-targets-file "$ARTIFACTS/registered_hook_targets.txt" \
  --overrides "$ARTIFACTS/overrides.json" \
  --output-json "$ARTIFACTS/disposition_ledger.json" \
  --output-markdown "$ARTIFACTS/disposition_ledger.md"
```

没有运行环境时可省略 targets 文件获得 AST 候选台账。报告用 `hook_target_source` 区分 `ast-candidates` 与 `runtime-export`，分别填写 `candidate_hook_targets` 或 `registered_hook_targets`，另一项为 null。模块路径匹配的条目保持 pending，避免误报 implemented/active。即使来自 register，也尚未证明目标属性能 apply。逐项补充来源与行为、裁决 adapter/plugin、查历史 commit，人工审核后才更新 overrides；不要套用旧任务的验收结论。

## 方法抽取与升级验证

下面两个工具保留历史 `sglang_dcu_plugin/models/deepseek` 布局，不能泛化到任意 plugin 名称。

```bash
# --check 只读；不加 --check 会改外部 PLUGIN 方法文件，需在计划开发的树内执行。
python3 tools/sync_deepseek_methods.py \
  --fork-python "$FORK/python" --plugin-root "$PLUGIN" --check
PYTHONPATH="$OFFICIAL/python:$PLUGIN" python3 tools/validate_hook_signatures.py \
  --official-python "$OFFICIAL/python" --plugin-root "$PLUGIN" \
  --output "$ARTIFACTS/hook_signatures.json"
```

签名检查只覆盖已注册的 `sglang_dcu_plugin.models` REPLACE Hook，比较参数结构、默认值数量等；不比较默认值语义，不覆盖所有 AROUND/BEFORE/AFTER、动态别名或全部类契约。必须另测完整预期 targets、重复/组合顺序、apply 解析、fallback 与模型语义。空集合不能当作完整通过。

输出 `history/model-pr-history-notes.md` 记录来源 commit、符号、影响的选型和回归风险；模板见 [history-notes](../templates/history-notes.md)。验收按 [validation](validation.md) 的适用矩阵，缺 GPU 的 lane 记 pending。
