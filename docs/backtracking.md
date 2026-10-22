# 源码回溯手册：生成与核验内嵌 Hook 仓库

> **本文用途**：说明如何将固定官方 SGLang 与外部 DCU plugin 组合成独立源码树，以及输入保护、清单核验与运行验证边界。它是源码内嵌 Hook 回溯，**不是历史 DAS fork 的逐文件或逐 commit 复刻**。skill 入口见 [hcu-plugin-backtrack](../skills/hcu-plugin-backtrack/SKILL.md)。

## 1. 输入与适用范围

| 输入 | 要求 |
|---|---|
| 解释器与 Git | Python 3.10+；Git 可执行 |
| official 来源 | 干净 worktree；有 loader、package initializer 和 `python/pyproject.toml` |
| 外部 plugin | 干净 worktree；受 Git 跟踪的 `sglang_dcu_plugin/__init__.py`、`compat/aiter.py` |
| revision | 明确的官方 commit-ish，交付时记录解析后的 SHA |
| 新输出路径 | 不与任一来源相同或父子重叠；默认不存在 |

当前方法库没有上述 plugin 业务包，不能作为 `--plugin`。历史工具不自动支持未来 `hcu_sglang_plugin` 包名；若新实现改变布局，要同步修改生成器和测试。

## 2. 工具执行链

```mermaid
flowchart LR
    V[检查干净输入 / 路径 / anchor] --> C[同级临时目录 clone 官方 revision]
    C --> P[复制跟踪 plugin 文件与许可依赖]
    P --> I[注入 bootstrap / loader / package-data]
    I --> S[语法检查与 SHA256 清单]
    S --> O[重命名为新输出]
    O --> H[独立核验清单]
```

生成器来自 `plugin_work/reconstruct_sglang_das.py`。它把受 Git 跟踪的 DCU plugin 放进新树，在 SGLang bootstrap 提前安装 AITER 兼容，在 loader 增加内嵌 dcu 注册，并补齐 JIT package data、依赖与第三方许可。

## 3. 生成并核验

从本库根目录执行，替换真实路径与官方 SHA：

```bash
python3 tools/reconstruct_sglang_das.py \
  --upstream /path/to/official-sglang \
  --plugin /path/to/sglang-plugin-das \
  --output /path/to/new-sglang-das-embedded \
  --revision <official-sha> --branch embedded-dcu
python3 tools/verify_reconstruction.py \
  --output /path/to/new-sglang-das-embedded
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

临时树验证成功后才安装输出；此处的保护不等于 `--force` 具备编辑合并或历史备份能力。首次复现优先使用新目录。

## 5. 生成后的运行验证

1. 审阅新增文件和注入点，检查依赖/许可、package-data 与源码语法。
2. 证明 `sglang` 和 `sglang_dcu_plugin` 均从输出的 `python` 导入；隔离同名外部 dcu entry point。
3. 内嵌 dcu 不依赖外部 `SGLANG_PLUGINS` allowlist；loader 的 `setdefault` 不会覆盖已发现的同名外部入口，因此不能混用后只凭“注册成功”判断来自内嵌版本。
4. 若部署还使用 adapter，另记录 adapter SHA、有效目标和加载链；**生成器不会自动内嵌 adapter**，也不会证明两仓组合已兼容。
5. 按 [验证手册](validation.md) 完成适用 prefill/decode、状态、graph、精度与性能。生成/哈希成功和模型通过分开记录。

生成器不自动 commit 或 push。确认差异与结果后，按实际交付流程管理新输出仓库。

## 6. 历史证据的边界

2026-09-07 原总结记录官方 `c3947ee...`、plugin `ce870e...`，173 个 plugin 跟踪文件、179 个关键哈希、5 项生成器测试、214 项 plugin 测试和 MiniMax TP4 请求通过。这些是历史环境结果，不能迁入新输出的验收结论。

当前工具库的版本验证见 [交付记录](delivery-validation.md)，来源与改写记录见 [抽取地图](extraction-map.md)。
