# 测试、方法搬移与基准

> **用途**：开发/迁移实现时选择机械搬移、接线、状态和基准的证据。组合矩阵详见 [验证手册](../../../docs/validation.md)。

## 1. 来源与方法搬移

从 DAS 的 write-sglang-test / mechanical-refactor-verify 和 adapter 测试规约提取。

对方法抽取先分开准备、机械移动和语义修改；用 AST/源码对照或可重放脚本说明仅允许的重写（如常量命名映射），列出残余差异。`sync_deepseek_methods --check` 只证明其映射规则下的文本同步，不证明完整语义。

## 2. 契约与执行环境

单元测试覆盖原实现 reference、shape/dtype/模式矩阵、数值谓词和 cache/状态副作用。无服务需求时 mock；测试服务要在 setup 失败时也能清理自有进程。上游自身测试按 pinned 版本 test/README 使用其 fixtures/CustomTestCase/CI 注册；本工具库的隔离 Git 测试使用标准 unittest，不依赖 SGLang。模块导入期不启动服务或 sys.exit。

从顶层入口验证完整 expected target set、主动/回退分支、目标解析和组合顺序。GPU 算子另外覆盖 eager 和变输入 capture/replay；分布式覆盖实际使用拓扑和 collective 顺序。不要为了通过放宽阈值而掩盖首次分叉。

## 3. 基准与组合回归

使用 adapter 时增加相同目标顺序、旧 import alias、配置来源，以及 plugin 停用后适配基线仍保留的用例。单仓通过不能替代组合。

bench 预热后测，分离编译时间，固定并发/请求长度/卡型/driver/vendor 和各仓 SHA；记录正确性及相对基线，避免把冷启动或不同机器绝对 tok/s 当作加速依据。
