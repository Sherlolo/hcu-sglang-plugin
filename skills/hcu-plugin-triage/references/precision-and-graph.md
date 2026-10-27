# 精度、cache 与 Graph：沿首次分叉排查

> **用途**：区分请求协议、数学计算、状态和 Graph 复用错误。开发闭环见 [历史经验](../../../docs/development-methodology.md)；结果按 [验证手册](../../../docs/validation.md) 记录。

## 1. 先验证观察本身

保留原始 content/reasoning、finish reason、token 数和模型名。health 或 HTTP 200 只说明相应接口成功；截断、控制字符和任意位置含正确数字不能算语义通过。确定性短答案用 [离线检查](../../../docs/validation.md#7-确定性生成响应的离线验收)；thinking/tool 协议与 checkpoint/template 对齐，不能为了候选通过而改归一化标准。

确认实际 import、包装链、provider、shape/dtype、拓扑及开关生效；仅检查同名社区算子不能代表模型实际调用的 vendor 分支。adapter 的 CUDA 语义不是物理 HIP 能力证明。

## 2. 做能区分假设的最小对照

固定同一输入 token、位置、cache 初态和采样。诊断每次只改变所研究的条件并证明分支命中；关闭 Graph/融合、换 backend 等配置有独立记录，不能成为原接受配置的代用品。

| 观察 | 下一检查 |
|---|---|
| 开关启停都返回同样错误 | 当前复现不只依赖该开关，继续查共同数据流；文本相同仍不等于 logits bit-identical |
| 对照配置先因缺 kernel/依赖失败 | 尚未取得该配置的数值对照，保留失败；不能推断原路径正确或两者一致 |
| 算子小样本正确、整模型错误 | 先查真实调用链、归约/状态 owner、实际 shape/variant，别直接扩大容差 |
| 不同专家数/shape 的 native 用例失败 | 记录其范围，回到本模型的实际配置复现，不能据此替换当前算法 |
| 仅 prefill cache hit 错误 | 查 restore、索引、滑窗/conv/Mamba/checkpoint，与重算对照 |
| 仅 Graph replay 错误 | 查 capture 绑定旧 tensor、slot/长度和 cache 更新、跨流依赖 |

## 3. 找首次输入相同而输出不同的边界

先对齐 embedding。decode 第 k 步与 prefill 重算对应行须消费相同 token；从已对齐的上一层向下检查 norm/QKV、RoPE、KV 写入/索引、attention、MoE 和 residual。不要从最终乱码倒推某个警告就是根因。

对状态链同时保存合法状态来源和完成责任：生产者输出是否仍欠 TP sum、哪层消费 pending、哪层写 residual、哪层转换物理 KV 视图。旧 skip 标志不能替代当前框架合同。优先通过真实公共入口/构造器/FFN exit 建立反例，测试方法见 [测试契约](../../hcu-plugin-dev/references/test-contract.md)。

量化比较分别报告理想 FP32、同量化数据 oracle 与 native 直接调用的误差；native 近似与运输错误分开。MoE 离散路由放大小差异不是提高容差的充分理由。

## 4. Graph 和探针的限制

Graph replay 绕过 Python forward hooks；记录实际命中数。capture 中不做 `.item()`/float/tolist 等主机同步，动态 tensor 不能仅在 Python capture 时赋值。用变输入、位置、slot、缓存内容验证 replay 仍消费本次数据。

使用 pinned 版本支持的日志/dumper，限制层、样本和磁盘量，仅在需要请求时启用。保存 tensor 会改变同步或时序，须确认探针下仍能复现。模块/分布式状态未保存时，单份 tensor dump 不等于可独立重放。

## 5. 收敛与复验

保留修复前失败签名、首个分叉、最小变更与同一断言修复后通过，再回到原接受 profile 的真实模型/Graph/服务。不要把未复现的原型留在交付中。若使用 adapter，另验证组合和停用后仍保留有效适配基线。

固定版本的案例、原始失败与最终六份响应见 [V3 实践](../../../docs/v3-runtime-validation.md) §7–§18；其中的设备、版本和阈值不成为下一任务的默认值。
