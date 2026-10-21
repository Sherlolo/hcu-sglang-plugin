# 能力验证记录模板

日期、执行人、能力 ID、输入各仓 SHA/dirty、模型/数据版本、容器/解释器、卡型/驱动/DTK、依赖版本、日志目录。

| Lane | 命令 | 环境/配置 | 断言/阈值与来源 | 状态 | 日志/结果 | 未覆盖条件 |
|---|---|---|---|---|---|---|

接线：预期 targets / 注册 targets / apply 成功 targets，重复/组合顺序与缺失数。
数值：reference、shape/dtype/mode、误差、cache/conv/Mamba 状态、fallback。
graph：capture/replay 后输入变化与状态变化；collective rank 一致性。
服务：模型加载/prefill/decode、TP/DP/EP/PP/PD、量化/KV、准确率/KL。
性能：预热/测量次数、并发和输入输出长度、TTFT/ITL/吞吐/显存、相对基线。
处置：passed / failed / pending / not-applicable，各有证据或原因；不要空填通过。
