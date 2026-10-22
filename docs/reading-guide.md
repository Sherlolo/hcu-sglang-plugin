# 阅读指南：从任务到文档与 skill

> **本文用途**：帮助新人、维护者和 agent 用最短路径找到需要的设计、操作和验收规则。全量文档职责见 [文档中心](README.md)，项目快速使用见 [根 README](../README.md)。

## 1. 第一次接手

按顺序阅读：

1. [根 README](../README.md)：当前能做什么、怎样运行、哪些仍在规划。
2. [架构设计](architecture.md) §1–§3：两仓独立且可组合、运行时关系和目录。
3. [职责治理](governance.md)：实现归属、目标所有权与兼容证据。
4. [环境准备](bootstrap.md)：确认 Python、路径和执行环境。
5. 选择下表的一条任务路线，不必一次加载所有参考资料。

## 2. 按任务阅读

| 任务 | 推荐阅读顺序 | 完成时应留下 |
|---|---|---|
| 新能力设计/开发 | [开发 skill](../skills/hcu-plugin-dev/SKILL.md) → [开发指南](plugins.md) → [配置](configuration.md) → [验证](validation.md) | 目标、门控、基线回退、适用组合和证据 |
| 跨仓改动/评审 | [治理](governance.md) → [协作](contributing.md) → [CI](ci.md) | 两仓任务依赖、兼容版本和组合测试 |
| DAS fork 同步或官方升级 | [迁移 skill](../skills/sglang-dcu-plugin-migration/SKILL.md) → [迁移手册](migration.md) | 固定基线、差异清单、处置台账、未验证项 |
| 源码内嵌回溯 | [回溯 skill](../skills/hcu-plugin-backtrack/SKILL.md) → [回溯手册](backtracking.md) | 新输出树、来源清单、核验与运行边界 |
| 精度 / graph / hang / 性能问题 | [排障 skill](../skills/hcu-plugin-triage/SKILL.md) → 对应 references → [验证](validation.md) | 最小复现、首个分叉、修复与回归证据 |
| 新增配置或改默认值 | [配置](configuration.md) → 开发 skill 的 runtime-and-env 参考 | 所有者、类型、默认、覆盖顺序、命中日志 |
| 改文档/skill | [文档约定](README.md#4-文档编写约定) → [来源地图](extraction-map.md) → [协作](contributing.md) | 一致的导航、目的说明、链接与状态标注 |
| 核查交付或找下一项任务 | [交付记录](delivery-validation.md) → [任务看板](board.md) | 已验证/待验证边界与接受条件 |

## 3. 接手时先核对的事实

| 核对项 | 原因 |
|---|---|
| 当前仓库没有可安装的运行时 plugin 包 | 避免对方法库执行 `pip install -e .` 或作为回溯 `--plugin` |
| adapter 与 plugin 可以共同部署 | 避免把职责分类解释成系统二选一 |
| 来源/目标是哪些 SHA，是否有脏文件 | 避免混合基线、覆盖用户工作或继承旧结论 |
| `original_fn` 指向哪一层有效基线 | 避免 fallback 绕过 adapter 或另一合法包装 |
| 当前结果是 AST 候选、注册、apply，还是行为验证 | 避免用接线成功代替功能等价 |
| 当前 gate 是否真正执行过 | `.workflow` 配方不等于已上线的远端 CI |

## 4. 需要重新检查的失效信号

上游目标路径/签名变化、顶层注册调用链变化、adapter 调整加载顺序、同一目标出现新包装层、依赖版本或平台判定改变、配置来源冲突、graph-only 分叉和 rank 间 collective 不一致，都应触发对应契约重新验证。

执行结果写 [验证记录模板](../templates/validation-record.md)；改完文档更新 [文档中心](README.md)，提交前按 [协作检查](contributing.md) 自查。
