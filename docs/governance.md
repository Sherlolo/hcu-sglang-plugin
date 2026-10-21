# 双仓边界与证据治理

先问：**上游在参照平台上有没有这个行为？** 有：HCU 行为补齐归 adapter；没有：新能力归 plugin。性能优化不是单独的分仓标准。历史 DAS plugin 混有平台判定、库兼容、kernel 路由和新融合路径，不能整包照搬。

| 例子 | 去向与理由 |
|---|---|
| 补齐已存在算子 ABI/平台语义/缺失库符号 | adapter，还原既有行为 |
| 上游没有的融合、新调度或专项内核 | plugin，通过窄 seam 接入 |
| 某项 fork 行为已被当前上游覆盖 | 台账标 already upstream，附源码/测试证据 |
| 缺 wheel/编译符号/框架 seam | native/package 或 integration gap，不能假装 Hook 可解决 |

每项迁移记录来源 commit/path/symbol、行为、official/plugin 状态、处置、目标与 Hook 类型、依赖、静态和运行证据、状态与风险。九类处置详见 [迁移契约](../skills/sglang-dcu-plugin-migration/references/migration-contract.md)。自动生成的分类只是审计起点，人工核对 active、语义与边界后才能验收。

状态：pending → implemented → verified；blocked 要注明缺什么，excluded 要有来源证明和接受条件。verified 必须包含该能力适用的集成与正确性证据；性能宣称另需相同环境的 A/B。运行时 Hook 数量、源码中候选数量、处置条目数量分别报告。

日常开发保持官方树不改，禁止运行时 import DAS fork。回溯模式允许在新生成的输出树内嵌代码和改 bootstrap/loader/packaging；来源树保持原样。回溯结果是源码内嵌 Hook 组合，不是历史 fork 的逐 commit 复刻。

本次仅交付方法库。未来增加运行时能力时，逐项建立可停用路径、登记生效目标、依赖和默认值，并用实际日志证明启用/停用分支命中；此处没有虚构 `HCU_PLUGIN_*` 或 adapter 的快关实现。
