---
name: hcu-plugin-dev
description: 设计或开发 SGLang 在 HCU 上的新增融合、后端、专项内核或调度能力，规划 seam、注册、依赖与回退，并验证独立或 adapter 组合运行。用于插件能力开发与设计评审，已有行为适配部分按职责拆分给 adapter。
---

# HCU Plugin 能力开发


## 用途与输入

用于一项新能力从设计到验证的执行。输入：需求与支持范围、official/plugin SHA 和路径；使用 adapter 时加其 SHA/能力依赖；可选 DAS 来源、产物目录和执行资源。当前仓库仅提供方法和工具，不含可安装运行时包。

先读 [架构](../../docs/architecture.md) 与 [治理](../../docs/governance.md)：adapter 和 plugin 独立开发、可共同部署。判别“上游是否已有行为”用于实现拆分，不把部署画成二选一。

## 执行流程

1. 固定基线与 dirty，定义新增行为、shape/dtype/mode、模型/并行和接受条件；保留用户已有工作。
2. 按 [开发指南](../../docs/plugins.md) 选择正式 seam 或窄 Hook。说明目标所有权、与 adapter 的交互、加载/包装顺序、依赖与关闭后的有效基线。
3. 将映射/门控与实现分开，顶层入口沿唯一聚合链到预期目标集合。有意多 AROUND 组合时测试顺序；AST 候选不代表已注册或已应用。
4. `original_fn` 可能已经过 adapter 或合法前序包装；回退应保留当前有效基线，不重新 import “原版”绕过适配。可选依赖懒加载，不吞 kernel/状态错误。
5. 编写支持/回退与缓存/状态测试，再做 graph、服务和适用组合。正式部署不从 DAS fork 动态 import 实现。

## 按需参考

| 任务 | 读取 |
|---|---|
| 包结构、阶段或首次运行时落地 | [架构规划](../../docs/architecture.md#4-未来运行时目录规划)、[CI](../../docs/ci.md) |
| 环境变量、配置、全局/per-forward 状态 | [runtime-and-env](references/runtime-and-env.md)、[配置手册](../../docs/configuration.md) |
| 方法搬移、数值/graph 测试或 bench | [test-contract](references/test-contract.md)、[验证矩阵](../../docs/validation.md) |
| 跨仓协作或评审 | [协作流程](../../docs/contributing.md) |

## 输出与完成条件

输出能力方案、涉及仓库/版本/目标表、实现和文档变更、[验证记录](../../templates/validation-record.md)。区分单仓/组合与 CPU/graph/服务/指标结论；未执行 lane 保持 pending 或有理由的 not-applicable。不把未来包名/entry point 当作当前已实现接口。
