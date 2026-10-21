# 迁移时的架构契约

entry point → 单一聚合入口 → feature register → HookRegistry → official apply_hooks，逐段证明可达。旧中文简析与 pinned 代码有偏差，以实际入口和导出集合为准。AROUND 负责条件优化与原函数回退；REPLACE 必须兼容全部输入/输出/副作用；类替换单独验证类型行为。

缓存写入、Mamba/conv 状态、权重映射、量化 metadata、collective 是行为的一部分。module patch 应说明何时处理常量/registry/旧 import alias。每次上游 SHA 变化都重验 target、signature 和执行语义。

缺依赖与明确不支持的分支才回退；kernel 和状态错误保留。正式组合为 official + adapter（需要时）+ 外部 plugin + 声明的依赖；运行时不能 import fork。选位以 [双仓治理](../../../docs/governance.md) 为准。
