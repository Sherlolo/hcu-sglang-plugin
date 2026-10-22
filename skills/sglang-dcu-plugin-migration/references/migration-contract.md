# 迁移处置契约与证据要求

> **用途**：定义每个来源差异的台账字段、处置类别和状态门槛，供三方审计与人工复核使用。工具字段保持原名；命令见 [迁移手册](../../../docs/migration.md)，组合规则见 [治理](../../../docs/governance.md)。

## 1. 台账字段

每个影响 runtime、build、packaging、test 或 invocation 的差异保留一行。

| 字段 | 应记录的内容 |
|---|---|
| `id` | 当前台账内可引用的条目 ID；清单重排后不能只凭序号追溯 |
| `fork_commit` / `fork_path` / `symbols` | 来源 commit 或 WORKTREE、路径、函数/类/常量 |
| `behavior` | 用户可见或运行时行为，不能只写文件名 |
| `official_status` | 上游缺失、部分、等价或已替代的实际证据 |
| `plugin_status` | 候选/registered/部分/active 等阶段，不能把 AST 候选标为生效 |
| `disposition` | 经审核的处置；默认 pending 分类只作起点 |
| `plugin_target` / `hook_type` | Hook 或自有模块，BEFORE/AFTER/AROUND/REPLACE/module patch/N/A |
| `dependencies` | adapter 能力、vendor、wheel、环境和最低版本 |
| `static_evidence` | 可达、目标解析、签名、测试、方法对照 |
| `runtime_evidence` | 组合、模型、命令、环境、日志和指标 |
| `status` / `risk` | 证据状态、兼容风险、未解决条件和跨仓任务引用 |

实现所有者与部署组合分别记录：同一条目可以引用 adapter 前置任务及 plugin 扩展，各自拥有实现和 commit。版本、装配顺序、fallback 基线可写入证据/风险字段或关联 [验证记录](../../../templates/validation-record.md)，不声称当前工具已实现额外 schema。

## 2. 九类语义处置

| 类别 | 判断依据 |
|---|---|
| Already upstream | pinned 上游已有等价行为，附源码和适用测试，不只看名称 |
| Already covered by plugin | 顶层入口能到达且覆盖的是该条行为，不只匹配模块 |
| New plugin Hook | 可通过支持的窄 Hook 增加行为 |
| Module patch | 常量、registry、alias 需要指定阶段更新 |
| Plugin-owned implementation | 新 helper/backend/model 片段由 plugin 维护 |
| Native/package work | 内核、wheel、编译、依赖工作仍需独立跟踪 |
| Official integration gap | 当前框架没有安全 seam，记录所需最小接口 |
| Test/CI/docs only | 迁移验证或文档，不宣称运行时功能 |
| Obsolete/intentionally excluded | 有来源证明和接受条件的过时/排除项 |

这些是人工审计的语义类别，不是当前脚本强制输出的严格枚举。adapter 行为补齐作为自身职责/依赖跟踪，不为关闭 plugin 台账而把其代码强塞入 plugin。WORKTREE 和继承硬件漂移都要审查，不以缺少硬件关键词排除通用运行时变化。

## 3. 状态证据门槛

| 状态 | 最低证据 |
|---|---|
| pending | 差异与待确认项可追溯；候选/模块映射仍在这一阶段 |
| implemented | 实现存在、顶层可达、target 解析通过，支持/回退单元契约具备 |
| verified | implemented 全部证据，加适用组合/集成、正确性和分支命中日志；性能宣称另需 A/B |
| blocked | 缺接口、依赖或资源，说明解除条件 |
| excluded | 过时、上游等价或主动排除的来源证明及接受条件 |

导出 target 只证明该次注册集合，不证明 apply 或行为。`original_fn` 可能已经由 adapter 包装，回退应保留当前有效基线。升级任一相关 SHA 后重新核对适用组合，不能继承旧 verified。

## 4. 交付统计

分别报告条目总数、各处置/状态分布、AST 候选数、runtime-export 注册数、成功 apply 数、未解析目标、静态 pass/fail、运行 lane pass/pending 及 native/package 缺口。

当前台账工具只区分 AST 和 runtime-export 计数，没有自动产生完整 apply/语义统计；后两者需要相应集成验证另留证据。不从默认自动分类直接得出“全量 fork 等价”。
