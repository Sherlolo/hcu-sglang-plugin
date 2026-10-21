---
name: hcu-plugin-triage
description: 定位 SGLang HCU plugin 的精度分叉、缓存/状态漂移、graph replay、分布式 hang 或性能回归。用于能力启停 A/B、逐层对拍和 rank 分叉定位；不负责通用服务运维或自动重启共享节点进程。
---

# HCU Plugin 回归定位


先固定能复现的模型、请求、各仓 SHA、依赖与硬件环境，记录插件启用/停用是否实际命中；无日志命中不能推断路径无问题。

- 输出/缓存/graph 差异：读取 [precision-and-graph](references/precision-and-graph.md)，先对齐 token/输入，再定位输入相同而输出首次分叉的边界。
- 多 rank 卡死、crash、性能：读取 [distributed-and-performance](references/distributed-and-performance.md)，区分 rank 已退出、collective 分叉、冷编译与热测量。
- 测试阈值和状态副作用：读取 [验证矩阵](../../docs/validation.md)，不要用短回答或宽松容差代替数值证据。

将根因缩成独立算子/状态/Hook 的最小复现，加针对性回归，并按 [记录模板](../../templates/validation-record.md) 写出命令、结果及未覆盖 lane。保留真实错误；不为绿测静默 fallback。
