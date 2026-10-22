---
name: hcu-plugin-triage
description: 定位 SGLang HCU plugin 的精度、缓存/状态、graph、分布式 hang 或性能回归，并检查 adapter/plugin 组合的目标和配置交互。用于最小复现、启停 A/B 和首次分叉定位。
---

# HCU Plugin 回归定位与组合诊断


## 用途与输入

输入能复现的模型/请求、official/plugin SHA、adapter SHA（使用时）、依赖与硬件环境、启动参数和日志。先确认实际 import 来源、包装顺序、有效基线和能力启停是否真的命中；无探针命中不能推断路径正常。

## 按症状选择路线

| 症状 | 读取与行动 |
|---|---|
| 输出、cache、状态或 graph 分叉 | [precision-and-graph](references/precision-and-graph.md)：先 token 对齐，再查首次输入同而输出不同的边界 |
| 多 rank 卡死、crash、性能 | [distributed-and-performance](references/distributed-and-performance.md)：排除 rank 已退出，比较 collective/状态，分开冷编译与热测量 |
| 仅在组合使用 adapter 时发生 | [治理](../../docs/governance.md)、[配置](../../docs/configuration.md)：检查导入期状态、相同目标、旧别名和 fallback 是否绕开适配 |
| 决定阈值、状态断言和运行组合 | [验证手册](../../docs/validation.md)：按模型/TP/环境标定，保留适用性解释 |

adapter 可能向 SGLang 暴露 CUDA 语义，不能仅凭某个 `is_cuda()` 推断真实硬件或 vendor 能力。不要用短回答、宽松阈值或 broad fallback 把错误隐藏成通过。

## 输出与完成条件

将根因缩成算子、状态、Hook 或组合加载的最小复现，增加针对性回归，并按 [验证模板](../../templates/validation-record.md) 写命令、环境、首次分叉、结果和未覆盖 lane。共享节点只操作本任务资源，不自动重启其他服务。
