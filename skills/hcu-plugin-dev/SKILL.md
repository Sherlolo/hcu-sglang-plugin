---
name: hcu-plugin-dev
description: 开发 SGLang 在 HCU 上的新融合、专项内核或调度能力，选择 plugin seam、Hook 类型、注册链、回退和验证方案。用于新增能力设计或实现；已有上游行为的 HCU 补齐先按 adapter 边界分类。
---

# HCU Plugin 开发


先读 [双仓治理](../../docs/governance.md)，按“上游参照平台是否已有行为”决定归属，不能以性能为由直接选 plugin。

1. 固定 official、adapter、外部 plugin 和来源 fork 的 SHA/dirty；本方法库不包含运行时包。
2. 按 [开发流程](../../docs/plugins.md) 选择最窄 seam，写明 shape/dtype/mode、缓存/状态和 collective 契约；Hook 映射与方法体分离。
3. 验证 entry point 沿唯一聚合链到完整预期目标集合。动态 targets 从运行时取，不能用 grep 数量证明 active；刻意多 AROUND 组合需要顺序测试。
4. 用 `original_fn` 保留受支持的回退；可选依赖懒加载，不吞 kernel/语义错误。正式运行不依赖 fork import。
5. 配置或全局/每步状态改动，读取 [runtime-and-env](references/runtime-and-env.md)；编写测试/基准时读取 [test-contract](references/test-contract.md)。
6. 交付 [验证记录](../../templates/validation-record.md)，区分 CPU、graph、服务、精度与性能证据；未运行的 lane 保持 pending。
