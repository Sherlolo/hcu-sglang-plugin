# 迁移架构契约：注册链与有效基线

> **用途**：迁移或升级时审查从 entry point 到运行时的接线，以及与 adapter 共用目标的兼容性。完整规则见 [治理](../../../docs/governance.md)，执行步骤见 [迁移手册](../../../docs/migration.md)。

## 1. 注册与解析

entry point → 唯一聚合链 → feature register → HookRegistry/扩展 registry → 框架 apply，逐段证明可达。旧中文简析与 pinned 代码可能有偏差，以实际入口和导出集合为准。检查目标属性、签名、别名与最终调用对象，不能只证明模块文件存在。

## 2. Hook 与副作用

AROUND 保留当前函数作为基线；REPLACE 兼容完整输入/输出/副作用；类替换另测类型行为。cache、conv/Mamba、权重映射、量化 metadata 和 collective 属于行为。module patch 写明常量、registry 或旧 alias 的更新阶段。

## 3. adapter 组合

两仓独立，运行时可以组合；声明 adapter 依赖、版本和目标所有权。`original_fn` 可能已适配，fallback 不绕过它。相同目标的加载顺序、重复注册、配置来源和包装链必须在组合中验证。正式运行可为 official + plugin，或 official + adapter + plugin + 声明依赖；不能 import DAS 来源工作树。

## 4. 漂移与证据

每次任一相关 SHA 变化，重验 target、signature、顺序和执行语义。缺依赖/明确不支持才使用声明的回退，真实 kernel/状态错误应保留。单仓结果不自动继承为组合支持，未执行 lane 保持 pending。
