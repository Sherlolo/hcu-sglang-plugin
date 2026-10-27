# 任务与 Skill 使用指南

> **本文用途**：提供任务、操作文档与 Skill 的对应关系。选择与当前目标匹配的入口，按操作文档中的调用示例执行。

## 1. 任务分类与文档入口

| 任务类别 | 推荐顺序 |
|---|---|
| 运行时使用 | [安装与启用](installation.md) → 安装与入口核验 → 设备及加载验证 → 要求的服务验收 |
| 插件开发 | [工程初始化](first-plugin.md)（新工程）或 [能力开发](plugins.md)（已有工程）→ 实现与针对性验证 |
| 故障诊断 | [故障诊断](troubleshooting.md) → 按症状选择参考 → 按需查询 [历史经验](development-methodology.md) |
| 迁移与交付 | [三方迁移](migration.md)、[版本升级](upgrading.md) 或 [源码回溯](backtracking.md)，按目标选择 |

[架构](architecture.md) 用于接口与组件设计，[治理](governance.md) 用于实现归属和 adapter 交互，[验证](validation.md) 用于确定验收范围。完整文档职责见 [文档中心](README.md)。

## 2. Skill 适用范围与交付要求

| 任务 | Skill | 执行说明与调用示例 | 主要交付 |
|---|---|---|---|
| 创建独立插件工程 | [hcu-plugin-scaffold](../skills/hcu-plugin-scaffold/SKILL.md) | [工程初始化](first-plugin.md#使用-skill-创建新插件) | 外部工程、入口映射、实际调用与启停结果 |
| 安装已有插件源码或 wheel | [hcu-plugin-install](../skills/hcu-plugin-install/SKILL.md) | [安装与验证](installation.md#使用-skill-安装与验证插件) | 安装与环境记录、入口及加载报告 |
| 新增或修改已有插件能力 | [hcu-plugin-dev](../skills/hcu-plugin-dev/SKILL.md) | [能力开发](plugins.md#使用-skill-开发插件能力) | 实现、注册/回退契约、相关测试及证据 |
| 定位加载后的运行回归 | [hcu-plugin-triage](../skills/hcu-plugin-triage/SKILL.md) | [故障诊断](troubleshooting.md#使用-skill-执行故障诊断) | 最小复现、首次分叉、修复及回归结果 |
| official / fork / plugin 差异审计 | [sglang-dcu-plugin-migration](../skills/sglang-dcu-plugin-migration/SKILL.md) | [三方迁移](migration.md#使用-skill-执行三方迁移) | 固定基线、差异清单、处置台账 |
| 指定官方目标升级与发版 | [upgrading-sglang-hcu-upstream-first](../skills/upgrading-sglang-hcu-upstream-first/SKILL.md) | [版本升级](upgrading.md#使用-skill-执行版本升级) | 补丁处置、验收矩阵、产物来源与 last-good |
| 生成内嵌源码产物 | [hcu-plugin-backtrack](../skills/hcu-plugin-backtrack/SKILL.md) | [源码回溯](backtracking.md#使用-skill-生成与核验内嵌源码) | 新源码树、来源清单、哈希及运行验证记录 |

## 3. 调用规则与任务衔接

向 Agent 提供方法库绝对路径，并使用对应文档中的请求模板。路径方式不依赖自动发现；`$skill-name` 仅适用于客户端已发现的 Skill。保留完整仓库，避免共享相对路径失效。

工程初始化 Skill 负责包与接入链；真实 HCU 功能由开发 Skill 负责。安装 Skill 区分安装、入口、设备和加载；加载后出现数值、状态、Graph 或性能问题时转诊断 Skill。三方迁移负责差异处置，指定官方升级与发版由升级 Skill 组织，要求 embedded 交付时再调用回溯 Skill。

普通安装不要求迁移或回溯；已有插件新增能力不要求重新创建工程。Skill 的任务输入和验收标准分别定义在对应操作文档与 `SKILL.md` 中，缺少本次必需输入时先检查已有上下文，再补充必要信息。

## 4. 参考资料与历史证据

- 方法工具：[环境准备](bootstrap.md)、[三方迁移](migration.md)、[源码回溯](backtracking.md)。
- 工程规范：[配置管理](configuration.md)、[验证规范](validation.md)、[协作流程](contributing.md)。
- 历史故障：[开发经验与故障案例索引](development-methodology.md)、[V3 实践记录](v3-runtime-validation.md)。
- 方法来源与交付：[vLLM Skill 迁移分析](vllm-skill-migration.md)、[交付记录](delivery-validation.md)。

当前需求的实现范围、验证结果和 HCU 逻辑差异见 [V3 2026-10-26 实施记录](v3-20261026-development.md)。历史经验用于定位相似问题，当前支持范围以对应固定版本的实施记录为准。
