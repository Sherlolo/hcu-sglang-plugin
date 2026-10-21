# 后续集成事项

本次方法与工具抽取的状态见 [delivery-validation](delivery-validation.md)。以下是未来实现任务的输入条件，均未在本次启动：

| 事项 | 所需输入 | 验收 |
|---|---|---|
| 从历史 DAS plugin 拆分新增能力 | official/adapter/plugin SHA、逐项行为台账 | 每项明确 adapter/plugin/native/gap 去向 |
| 接入新模型/新能力 | seam、目标集合、依赖和支持矩阵 | 完整入口可达、fallback、数值/graph/服务测试 |
| 更新回溯到新上游 | 新 loader/bootstrap/package-data 格式 | fixture 回归及新输出 provenance 核验 |
| 接入托管 CI | 使用的平台、runner 与路径 | 明确 CPU/GPU lane，真实触发并留日志 |

领取任务时记录负责人、输入基线、产物路径、接受条件和当前阻塞；完成后将结论沉淀到文档和验证记录，不保留失效的“已通过”历史数字作为新基线。
