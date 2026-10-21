# 新能力开发流程

1. 固定 official、adapter、plugin 与可选 DAS 来源 SHA；先按 [治理](governance.md) 选仓。
2. 定义新增行为、支持的 shape/dtype/mode/TP/量化、原函数与状态副作用，先写契约用例。窄方法 Hook 优于整类替换。
3. 选择 Hook：AROUND 承载有条件优化与 `original_fn` 回退；BEFORE 只改参数；AFTER 只改结果/观察；REPLACE 需拥有完整兼容实现。常量/registry/旧 import 别名属 module patch，记录施加时序。
4. 在外部实现仓拆开映射与实现：`models/<family>/hooks/` 放目标和门控，`methods/` 放方法，`constants.py` / `_runtime.py` 收敛依赖。这是参考布局，不要求所有新能力照搬 DeepSeek。
5. 顶层 entry point 沿唯一聚合链到每个功能注册器。重复同类注册拒绝；刻意组合多个 AROUND 时写明顺序、原函数传递和组合测试。不能把两个历史 aggregator 同时打开来“补覆盖”。
6. 可选依赖在选中路径内懒加载；仅明确缺依赖或不支持条件返回 official。真实 kernel 错误、cache 写入错误不能用 broad except 静默回退。非目标硬件走原路径。
7. 证明顶层注册可达、目标解析、签名/返回/缓存/collective 语义，再分层验证。最终部署不从 fork 取运行时代码。

## 容易遗漏的语义

融合 RMSNorm/RoPE/KV store 后，后端不能再次写同一 KV；按当前 batch/layer 标记并测试标记生命周期，不能泄漏到下一请求。conv/Mamba 状态、权重名映射、量化 metadata 同样是功能契约。

decode 热路径不引入 `.item()`、`nonzero` 等主机同步；graph 验证必须 capture 后改变输入/状态再 replay。collective 的参与 rank、张量形状和调用顺序须一致。注册成功只证明接线，不能证明数值或性能。

配置与全局状态见 [环境与状态参考](../skills/hcu-plugin-dev/references/runtime-and-env.md)；测试和基准见 [测试参考](../skills/hcu-plugin-dev/references/test-contract.md)。
