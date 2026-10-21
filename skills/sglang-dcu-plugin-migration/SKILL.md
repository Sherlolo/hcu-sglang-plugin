---
name: sglang-dcu-plugin-migration
description: 审计 SGLang official、DAS fork 与外部 DCU plugin 的三方差异，抽取迁移清单、处置台账并验证版本漂移。用于 fork 同步、插件 rebase 或等价性审计；先区分 adapter 行为补齐与 plugin 新增能力。
---

# SGLang DCU 三方迁移


按 [迁移手册](../../docs/migration.md) 操作，输入 official/fork/外部 plugin 路径和产物目录；记录 SHA、merge-base、dirty 和运行环境。审计保留脏文件，不 reset 来源树。

1. 执行共享 [审计工具](../../tools/audit_repositories.py) 与 [三方清单工具](../../tools/build_migration_manifest.py)。同时审查 merge-base delta、WORKTREE 和 `direct_hardware_drift`，不能只筛硬件关键词。
2. 查受影响模型/运行时的 commit 与源码 diff，填写 [历史模板](../../templates/history-notes.md)。
3. 读取 [迁移契约](references/migration-contract.md)，逐项分类并按治理边界交付；`build_disposition_ledger.py` 只生成审计起点，自动覆盖/人工 overrides 都须核验来源证据。
4. 从顶层注册导出 runtime targets；AST/源码 f-string 候选不能宣称 active。按 [架构契约](references/plugin-architecture.md) 验证入口、解析、签名与 fallback。
5. 需要 DeepSeek 方法同步时用外部实现仓先跑 `sync_deepseek_methods.py --check`；该工具依赖旧 DCU 布局。签名工具只检查 models REPLACE 子集，不能替代完整 Hook 合约。
6. 用 [验证矩阵](../../docs/validation.md) 记录适用 lane。缺 GPU 记 pending；没有完整台账和适用运行证据不宣称 fork 等价。
