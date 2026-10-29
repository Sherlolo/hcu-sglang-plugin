# 内嵌源码生成与核验指南

> **本文用途**：说明如何将固定官方 SGLang 与外部 DCU plugin 组合成独立源码树，以及输入保护、清单核验与运行验证边界。它是源码内嵌 Hook 回溯，**不是历史 DAS fork 的逐文件或逐 commit 复刻**。Skill 入口见 [hcu-plugin-backtrack](../skills/hcu-plugin-backtrack/SKILL.md)。

## 使用 Skill 生成与核验内嵌源码

对应 Skill：[`hcu-plugin-backtrack`](../skills/hcu-plugin-backtrack/SKILL.md)。该 Skill 调用本指南中的生成与核验工具；它不自动内嵌 adapter，也不将哈希通过视为运行正确。

**输入要求**：干净 official/plugin 输入、明确 revision、新输出目录、产物目录及要求的运行验收范围。

**调用示例**：将占位符替换为实际绝对路径与任务信息，提交给 Agent。

```text
读取 <方法库绝对路径>/skills/hcu-plugin-backtrack/SKILL.md，按该 Skill 执行。
官方输入：<路径及 revision/SHA>。
插件输入：<路径及 SHA，包含 hcu_sglang 或旧 sglang_dcu_plugin>。
新输出目录：<与两输入不重叠的路径>；证据目录：<本次任务目录>。
运行验收范围：<仅生成与核验，或指定模型/配置>。
检查输入干净和目录边界，生成内嵌源码树并核验清单。
保留已有输出和用户编辑；若需更新旧生成树，先执行手册规定的完整保护检查。
运行验证与生成/哈希结果分别记录。
```

**预期交付**：独立 Git 源码树、来源与 SHA256 清单、核验结果、实际模块来源及运行验证状态。

## 1. 输入与适用范围

| 输入 | 要求 |
|---|---|
| 解释器与 Git | Python 3.10+；Git 可执行 |
| official 来源 | 干净 worktree；有 loader、package initializer 和 `python/pyproject.toml` |
| 外部 plugin | 干净 worktree；受 Git 跟踪的 `hcu_sglang/__init__.py`（或旧包）和包内 `compat/aiter.py` |
| revision | 明确的官方 commit-ish，交付时记录解析后的 SHA |
| 新输出路径 | 不与任一来源相同或父子重叠；默认不存在 |

生成器 v4 自动识别唯一的 `hcu_sglang` 或旧 `sglang_dcu_plugin` 包；两者并存时必须用 `--plugin-package` 显式选择，缺失包会拒绝生成。当前 HCU 包使用内嵌 `hcu` 注册，并打包 JIT overlay 和 DeepSeek V4 backend csrc；旧输入保留 `dcu`。来源清单增加实际包名，清单文件名为兼容已有核验工具保持不变。源码生成与真实运行分别验收。

## 2. 工具执行流程

```mermaid
flowchart LR
    V[检查干净输入 / 路径 / anchor] --> C[同级临时目录 clone 官方 revision]
    C --> P[复制跟踪 plugin 文件与许可依赖]
    P --> I[注入 bootstrap / loader / package-data]
    I --> S[语法检查与 SHA256 清单]
    S --> O[重命名为新输出]
    O --> H[独立核验清单]
```

生成器来自 `plugin_work/reconstruct_sglang_das.py`。它把受 Git 跟踪的 DCU plugin 放进新树，在 SGLang bootstrap 提前安装 AITER 兼容，在 loader 增加对应包的内嵌注册，并补齐 JIT package data、依赖与第三方许可。

生成器版本 3 起不再向 bootstrap/loader 插入注释或 BEGIN/END 标记；通过导入符号和历史标记检测拒绝重复集成。复制的来源注释和许可声明保持原样，来源信息仍写入 `.sglang-dcu-embedded.json` 与 `DCU_EMBEDDED.md`。统一要求见[生成代码约定](contributing.md#生成代码约定)。

## 3. 源码生成与清单核验

从本库根目录执行，先按 [路径配置](configuration.md#1-当前工具的路径配置) 设置输入和新输出目录，并核对本次官方 SHA：

```bash
: "${OFFICIAL:?请设置官方源码路径}"
: "${PLUGIN:?请设置外部插件路径}"
: "${OUTPUT:?请设置与输入不重叠的新输出目录}"
OFFICIAL_SHA=$(git -C "$OFFICIAL" rev-parse HEAD)
python3 tools/reconstruct_sglang_das.py \
  --upstream "$OFFICIAL" \
  --plugin "$PLUGIN" \
  --output "$OUTPUT" \
  --revision "$OFFICIAL_SHA" --branch embedded-dcu
python3 tools/verify_reconstruction.py \
  --output "$OUTPUT"
```

| 产物 | 内容 | 应检查 |
|---|---|---|
| 新 Git 仓库 | 保留官方历史，内嵌和注入内容待提交 | `git status`、diff 与实际 import 来源 |
| `.sglang-dcu-embedded.json` | 输入路径/SHA、模式、分支、版本、关键文件 SHA256 | 与指定输入/输出相符 |
| verifier JSON | 检查文件数、错误和 status | `passed` 且 errors 为空；失败退出码非零 |

清单证明列出文件的完整性；不证明未列出文件、来源签名、模型正确性或性能。不要在发现损坏后重写清单来掩盖差异。

## 4. 拒绝条件与覆盖语义

| 情况 | 工具行为与处理 |
|---|---|
| 来源 dirty（含未跟踪文件） | 拒绝；保留用户工作，另选干净基线 |
| 输出与来源父子重叠 | 拒绝；改用独立新路径 |
| anchor 与指定上游格式不符 | 失败并保留来源；针对新 SHA 修复并测试生成器，不猜注入点 |
| 输出已存在 | 默认拒绝 |
| 显式 `--force` | 仅接受有合法 kind 清单的旧生成结果，但会删除该输出的后续编辑；先确认可替换 |
| 清单越界、文件缺失/修改 | verifier 返回非零，定位实际差异 |

显式覆盖前，先运行 verifier 核验旧清单全部哈希，再将 `git diff --name-only`、暂存变更和 `git ls-files --others --exclude-standard` 的完整集合与旧生成清单及 manifest 文件对照。额外路径或旧哈希不符说明有生成范围外工作或修改，保留它并选择新输出/备份后再处置；合法 kind 本身不证明可丢弃。

临时树验证成功后才安装输出；此处的保护不等于 `--force` 具备编辑合并或历史备份能力。首次复现优先使用新目录。

## 5. 生成后的运行验证

1. 审阅新增文件和注入点，检查依赖/许可、package-data 与源码语法。
2. 证明 `sglang` 和 `sglang_dcu_plugin` 均从输出的 `python` 导入；隔离同名外部 dcu entry point。
3. 内嵌 dcu 不依赖外部 `SGLANG_PLUGINS` allowlist；loader 的 `setdefault` 不会覆盖已发现的同名外部入口，因此不能混用后只凭“注册成功”判断来自内嵌版本。
4. 若部署还使用 adapter，另记录 adapter SHA、有效目标和加载链；**生成器不会自动内嵌 adapter**，也不会证明两仓组合已兼容。
5. 按 [验证手册](validation.md) 完成适用 prefill/decode、状态、Graph、精度与性能。生成/哈希成功和模型通过分开记录。

生成器不自动 commit 或 push。确认差异与结果后，按实际交付流程管理新输出仓库。

## 6. Native 资源与运行来源

生成器 v2 的 package-data 包含 `jit_overlay/**/*`，覆盖 `.cu/.cpp/.hpp` 等真实资源。不能只确认源码树存在文件，还要检查打包规则与 wheel 内容；这次问题与回归见 [V3 实践](v3-runtime-validation.md)。

核验哈希后，用 [严格验证工具](../tools/validate_plugin_runtime.py) 的 `--disable-entry-points` 屏蔽外部发现，强制两模块从输出 python 导入。外部测试源码配合 `--import-mode=importlib`，避免测试路径把外部实现插回 sys.path。

## 7. 历史证据的边界

2026-09-07 原总结记录官方 `c3947ee...`、plugin `ce870e...`，173 个 plugin 跟踪文件、179 个关键哈希、5 项生成器测试、214 项 plugin 测试和 MiniMax TP4 请求通过。这些是历史环境结果，不能迁入新输出的验收结论。

当前工具库的版本验证见 [交付记录](delivery-validation.md)，来源与改写记录见 [抽取地图](extraction-map.md)。

## 8. HCU 来源逻辑对照

当需求另提供 HCU fork 并要求逻辑一致性核验时，在两个生成输入之外固定该 fork SHA，独立输出对照记录：

| 比较层次 | 核验内容 | 结论边界 |
|---|---|---|
| 生成保真 | 实际插件文件到 embedded 的逐文件哈希、清单及 bootstrap/loader 映射 | 证明本次生成保留了哪些实现 |
| 来源保真 | 当前 HCU 的对应符号、native/package-data 与实际输出 | 发现迁移遗漏或错误来源，不能只核对输出自写 manifest |
| 行为一致性 | 构造/装饰器/receiver、全局别名、状态、collective owner、公开调用与实际生成 | 对适配差异给出证据；无法证明的范围明确保留未验证 |

整个 fork 与官方基础结构可不同。按 caller/consumer 比较行为，保留有证据的接口适配差异；不能为文本相等引入重复 norm、KV store 或归约。源码哈希、selected AST 等价和 loader apply 不足以证明整个 fork 逻辑等价。本轮实例及未覆盖项见 [2026-10-26 实施记录](v3-20261026-development.md#4-源码内嵌与-hcu-逻辑对照)。


## 9. 来源索引保护

生成器对子进程设置 `GIT_OPTIONAL_LOCKS=0`，避免 `git status` 在来源校验时刷新输入仓库的 stat cache。它仍允许输出 checkout 的必要 Git 写操作，不是只读文件系统保护。

需要保留输入索引的任务，生成前后分别记录 HEAD、工作区/暂存状态与索引文件 SHA256；人工检查来源时也可使用 `GIT_OPTIONAL_LOCKS=0 git -C <来源> status --porcelain=v1`。发现索引哈希变化时先核实 staged 内容与缓存差异，保留证据，不通过 reset 或复制重建索引掩盖变化。案例与实际边界见 [索引缓存案例](development-methodology.md#17-只读-git-检查与索引缓存)。
