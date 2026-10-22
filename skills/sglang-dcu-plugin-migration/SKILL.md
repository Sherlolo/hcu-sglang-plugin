---
name: sglang-dcu-plugin-migration
description: 审计 SGLang official、DAS fork 与外部 DCU plugin，生成差异清单和处置台账并检查版本漂移。用于 fork 同步、插件升级或覆盖审计，同时拆分 adapter/plugin 实现职责并记录所需运行组合。
---

# SGLang DCU 三方迁移与兼容审计


## 用途与输入

用于固定基线后的三方差异审计、处置和升级。输入 official/fork/外部 plugin 路径、产物目录；组合使用 adapter 时补充版本与目标交互。当前方法库不能作为包含业务代码的 plugin 输入。

详细命令按 [迁移手册](../../docs/migration.md) 执行；先记录 SHA、merge-base、dirty 和环境。审计保留脏文件，不 reset/clean。adapter 与 plugin 可共同部署，实现归属与运行组合分别记录。

## 执行流程

1. 用 [audit_repositories.py](../../tools/audit_repositories.py) 记录文件/AST/哈希，再用 [build_migration_manifest.py](../../tools/build_migration_manifest.py) 生成清单；同时审查 merge-base delta、WORKTREE 和 direct_hardware_drift，不能只筛硬件关键词。
2. 查询相关模型/运行时的实际 commit 与源码 diff，填写 [历史模板](../../templates/history-notes.md)。
3. 读 [迁移契约](references/migration-contract.md)，逐项补行为、处置、证据和缺口。`build_disposition_ledger.py` 只生成审计起点，overrides 不能直接继承旧 verified。
4. 从顶层 register 导出 runtime targets，核对解析/别名/签名/顺序与语义。AST、注册、apply 和有效行为分开计数；组合契约见 [plugin-architecture](references/plugin-architecture.md)。
5. DeepSeek 同步先用 `sync_deepseek_methods.py --check`，它依赖历史外部布局；签名工具只覆盖 models REPLACE 子集，空结果或子集全绿不是完整兼容证明。
6. 按 [验证手册](../../docs/validation.md) 执行适用 lane。使用 adapter 时单独验证组合加载、配置与 fallback 到有效基线，不能用两仓各自通过替代组合。

## 输出与完成条件

提供固定基线、清单、历史证据、人工审核台账、各处置/状态计数、目标与运行结果、native/package/integration gap。跨仓任务引用前置接口和最低兼容版本，见 [协作](../../docs/contributing.md)。没有完整台账和适用运行证据，不宣称 fork 等价；缺依赖/硬件记 pending。
