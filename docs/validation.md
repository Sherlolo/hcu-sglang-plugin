# 验证与排障

验证状态分别记录 passed / failed / pending / not-applicable，并附理由和命令。工具库 L0、工具单测与外部 plugin 模型验收是不同层。

| 层级 | 证据 | 通过含义 |
|---|---|---|
| L0 | Python AST、skill 元信息/资源、相对链接、无业务代码 | 方法库结构完整 |
| CPU 工具测试 | 合成 Git 仓、差异清单、台账、回溯/原子拒绝/哈希损坏 | 工具契约可重复 |
| plugin 接线 | import、entry point、顶层预期集合、目标解析、签名、组合顺序 | 已注册且可应用 |
| 功能/数值 | reference 对拍、形状/dtype/模式、缓存/状态副作用、fallback | 指定能力正确 |
| graph | 变输入、变状态 capture/replay 对拍 | 捕获语义正确 |
| 服务集成 | 模型加载、prefill/decode、用到的 TP/DP/EP/PP/PD 与量化/KV | 指定模型配置可运行 |
| 精度/性能 | 固定数据与环境的 A/B、准确率/KL、TTFT/ITL/吞吐/显存 | 对应指标有证据 |

## 精度与 graph

先单算子 reference，再逐层定位“输入一致而输出首次不同”的边界，再查缓存/状态；最后跑整模型与指标。prefill/decode KL 要区分 batch-invariance 与状态等价，基础、prefill cache hit、decode cache hit 三种路径不能混用。TP=1 的阈值不自动适用于 TP>1；MoE 路由会放大微小数值差。

graph 回放不执行 Python forward hook。关 graph 收集到的正常张量不能排除 graph 问题；capture 期间主机同步会使探针失效。详见 [精度参考](../skills/hcu-plugin-triage/references/precision-and-graph.md)。

## Hang 与 crash

先区分某 rank 已 OOM/异常退出与 collective 真正卡死。采集各 rank 时间、调用栈、collective 顺序/shape、batch/seq/cache 状态，定位首次分叉。HIP 上优先使用环境已有的 watchdog、Python 栈、ROCm/HCU 工具；上游 CUDA coredump/cuda-gdb 命令不能直接当作 HCU 命令。详见 [分布式与性能参考](../skills/hcu-plugin-triage/references/distributed-and-performance.md)。

## 性能与发布记录

预热编译与测量分离；记录卡型、DTK/driver、torch/triton/vendor 版本、各仓 SHA、模型权重版本、shape/dtype、并发/输入输出长度、graph 状态。比较同环境基线与新能力，使用相对比值，同时记录正确性。只测冷启动或一个短回答不能宣称加速。按 [validation-record](../templates/validation-record.md) 填证据；本次交付记录见 [delivery-validation](delivery-validation.md)。
