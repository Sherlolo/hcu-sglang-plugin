---
name: sglang-dcu-plugin-migration
description: 审计 SGLang official、DAS fork 与外部 DCU plugin，生成差异清单和处置台账并检查版本漂移。用于 fork 同步、插件升级或覆盖审计，同时拆分 adapter/plugin 实现职责并记录所需运行组合。
---

# SGLang DCU 三方差异审计与迁移


## 适用场景与任务衔接

[阅读指南](../../docs/reading-guide.md) 给出本 Skill 的输入、产出和可复制请求。各项 Skill 按任务选用，不要求顺序执行；安装现有插件使用 [安装 Skill](../hcu-plugin-install/SKILL.md) 和 [安装指南](../../docs/installation.md)。

## 用途与输入

用于固定基线后的三方差异审计、处置和升级。输入 official/fork/外部 plugin 路径、产物目录；组合使用 adapter 时补充版本与目标交互。当前方法库不能作为包含业务代码的 plugin 输入。

详细命令按 [迁移手册](../../docs/migration.md) 执行；先记录 SHA、merge-base、dirty 和环境。审计保留脏文件，不 reset/clean。adapter 与 plugin 可共同部署，实现归属与运行组合分别记录。

生成或修改源码遵守[生成代码约定](../../docs/contributing.md#生成代码约定)：不新增说明性注释、生成标记、来源 docstring 或 lint 抑制注释；保留来源已有内容与许可声明，说明写入独立文档和清单。

## 执行流程

1. 用 [audit_repositories.py](../../tools/audit_repositories.py) 记录文件/AST/哈希，再用 [build_migration_manifest.py](../../tools/build_migration_manifest.py) 生成清单；同时审查 merge-base delta、WORKTREE 和 direct_hardware_drift，不能只筛硬件关键词。
2. 查询相关模型/运行时的实际 commit 与源码 diff，填写 [历史模板](../../templates/history-notes.md)。
3. 读 [迁移契约](references/migration-contract.md)，逐项补行为、处置、证据和缺口。`build_disposition_ledger.py` 只生成审计起点，overrides 不能直接继承旧 verified。
4. 从顶层 register 导出 runtime targets，核对解析/别名/签名/顺序与语义。沿 [caller 与 owner 检查](../../docs/development-methodology.md#4-moe-归约所有权与重复通信) 审查状态/归约是否已移到框架边界；旧方法中留下的 all-reduce、norm、KV store 或默认参数可能成为重复操作。AST、注册、apply 和有效行为分开计数；组合契约见 [plugin-architecture](references/plugin-architecture.md)。
5. DeepSeek 同步先用 `sync_deepseek_methods.py --check`，它依赖历史外部布局；签名工具只覆盖 models REPLACE 子集，空结果或子集全绿不是完整兼容证明。
6. 按 [验证手册](../../docs/validation.md) 执行适用 lane。使用 adapter 时单独验证组合加载、配置与 fallback 到有效基线，不能用两仓各自通过替代组合。

## 输出与完成条件

提供固定基线、清单、历史证据、人工审核台账、各处置/状态计数、目标与运行结果、native/package/integration gap。跨仓任务引用前置接口和最低兼容版本，见 [协作](../../docs/contributing.md)。没有完整台账和适用运行证据，不宣称 fork 等价；缺依赖/硬件记 pending。

## 实践补充

V3 中 merge-base 清单包含大量上游历史，结合当前树等价性与具体符号审阅；不要把原始条目数当迁移完成数。方法同步先全部预检，类拆分/别名不自动猜测。真实 loader 需要 [严格运行时检查](../../tools/validate_plugin_runtime.py)，捕获框架只记日志的软失败和空注册。见 [V3 实践](../../docs/v3-runtime-validation.md)。

## 指定目标升级与发版

若任务包含从当前基线升级到指定 official ref，并要求 wheel/image/embedded 交付验收，读取 [升级与发版 Skill](../upgrading-sglang-hcu-upstream-first/SKILL.md)。先做上游接管判定，将本 Skill 的处置台账关联本次验收矩阵；仅盘点差异时仍按本入口完成，无需加载全部发版参考。

## 当前源码与迁移逻辑核验

来源 SHA 变化后重新检查当前树，不继承旧输出/测试或旧 verified。输出缺失时，从有来源哈希的快照恢复并明确新 Git 历史，再迁移当前 delta。抽取方法比较包括类拥有者、装饰器、模块别名及 collective owner；参考 [本轮案例](../../docs/v3-20261026-development.md#4-源码内嵌与-hcu-逻辑对照)。资源哈希一致、Hook 全量 apply 与整体 fork 逻辑等价分别给结论，未覆盖的新功能显式保留。


## 协议与消费者的逻辑对照

差异审阅包含 async handler，并沿状态生产、wire codec、发布、调用方和下一消费者检查实际数据。来源 AST 一致只证明声明的节点；源实现也可能保留与新描述符语义冲突的旧预检。用修复前反例和实际字节效果解释有意适配差异，保留正常基线，参见 [PD 描述符案例](../../docs/development-methodology.md#13-描述符身份与位置式校验冲突) 与 [本轮记录](../../docs/v3-20261026-development.md#73-pd-描述符生产协商与复制)。

## 团队模块盘点与研发计划

准备将 DAS 按模块转为插件时，先读 [迁移规划指南](../../docs/sglang-das-migration-plan.md) 和 [研发网页](../../migration-plan.html)。可用 [build_migration_plan.py](../../tools/build_migration_plan.py) 对固定三方 HEAD 采集全量路径、顶层 Python 定义、合同与来源线索，按工作域组织建议任务。它不执行运行时加载，不做完整 merge-base 归因，也不把静态线索变成完成率。来源或计划定义变化后重新采集；随后仍按本 Skill 做人工处置、符号/消费者审阅及实际验收。
