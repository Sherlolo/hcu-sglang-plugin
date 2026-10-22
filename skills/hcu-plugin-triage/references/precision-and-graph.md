# 精度、cache 与 graph 定位

> **用途**：定位数值/缓存/graph 首次分叉，区分输入不对齐、算术与状态问题。按 [验证手册](../../../docs/validation.md) 保存环境、组合与结果。

提取自 DAS kl-consistency-test、debug-cuda-crash 与 adapter precision-triage。

## 定位顺序

1. 固定同一输入 token、位置、cache 初态与采样设置，做能力启停 A/B，并证明开关实际命中。短文本相同不能替代 logits/状态对拍。
2. 用最小 reference 检查算子，再逐层找“输入 bit-identical、输出首次不同”。decode 第 k 步与 prefill 的 prompt_length+k 行须消费同 token，先验证 embedding 对齐，否则所有差异无效。
3. 区分 batch-invariance（M 改变会否改变单 token 结果）与状态等价（KV/滑窗/conv/Mamba/checkpoint 是否与重算一致）。基础 match、prefill cache hit、decode cache hit 分别测。只有 prefill cache hit 异常时先查 restore/边界/淘汰，不先扩容差。跨过滑窗边界；阈值按模型和 TP 标定。
4. graph 回放绕过 Python hooks；关闭 graph 才测得一致不能排除 graph 问题。探针加最小 hit 数证明执行过；capture 中不 `.item()`/float/tolist 主机同步。对 graph 外状态复用点比较 slot、声明长度与内容，确认新 tensor 是否只绑定在 capture。
5. 使用 pinned SGLang 的 kernel API 日志/dumper 前先查该版本是否支持相关环境开关。统计/保存 tensor 会改变时序或同步，需证明问题仍复现。仅在需要请求时打开 dump、限制样本和磁盘占用，避免 warmup 写满磁盘。

确定机制后给最小 repro 与回归测试；MoE 离散路由放大微小数值差不能直接作为容差更宽的理由。DAS 文档里的 NVIDIA 调试器命令不直接移植到 HCU。

组合问题先记录 active 对象和包装链，分别做能力启停与 adapter/plugin 组合对照；不要把 fallback 到未适配路径造成的差异误判为算子数值问题。
