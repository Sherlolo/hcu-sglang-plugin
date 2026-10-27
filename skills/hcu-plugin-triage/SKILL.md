---
name: hcu-plugin-triage
description: 定位 SGLang HCU plugin 的精度、缓存/状态、Graph、分布式 hang 或性能回归，并检查 adapter/plugin 组合的目标和配置交互。用于最小复现、启停 A/B 和首次分叉定位。
---

# HCU 插件故障诊断与组合分析


## 适用场景与任务衔接

[阅读指南](../../docs/reading-guide.md) 给出本 Skill 的输入、产出和可复制请求。各项 Skill 按任务选用，不要求顺序执行；安装现有插件使用 [安装 Skill](../hcu-plugin-install/SKILL.md) 和 [安装指南](../../docs/installation.md)。

从 [症状路线与 TP4 乱码案例](../../docs/troubleshooting.md) 开始选择参考，不一次加载全部 references。定位后需要修改实现时转 hcu-plugin-dev，再回原接受配置复验；不用先做迁移或回溯。

## 用途与输入

输入能复现的模型/请求、official/plugin SHA、adapter SHA（使用时）、依赖与硬件环境、启动参数和日志。先确认原始请求/响应、finish reason、实际 import 来源、包装顺序、有效基线与能力启停是否命中。HTTP 200、数字偶然出现在长文本或同名算子全绿都不能证明模型正常；确定性短答案按 [生成验收](../../docs/validation.md#7-确定性生成响应的离线验收) 检查。

## 症状分类与诊断参考

| 症状 | 读取与行动 |
|---|---|
| 输出、cache、状态或 Graph 分叉 | [precision-and-graph](references/precision-and-graph.md)：先 token 对齐，再查首次输入同而输出不同的边界 |
| 多 rank 卡死、crash、性能 | [distributed-and-performance](references/distributed-and-performance.md)：排除 rank 已退出，比较 collective/状态，分开冷编译与热测量 |
| 仅在组合使用 adapter 时发生 | [治理](../../docs/governance.md)、[配置](../../docs/configuration.md)：检查导入期状态、相同目标、旧别名和 fallback 是否绕开适配 |
| 决定阈值、状态断言和运行组合 | [验证手册](../../docs/validation.md)：按模型/TP/环境标定，保留适用性解释 |

adapter 可能向 SGLang 暴露 CUDA 语义，不能仅凭某个 `is_cuda()` 推断真实硬件或 vendor 能力。不要用短回答、宽松阈值或 broad fallback 把错误隐藏成通过。

## 输出与完成条件

参考 [历史排障经验](../../docs/development-methodology.md) 将根因缩成算子、状态、归约 owner、Hook 或组合加载的最小复现，保留修复前失败并增加针对性回归，并按 [验证模板](../../templates/validation-record.md) 写命令、环境、首次分叉、结果和未覆盖 lane。默认仅操作自有服务；已有用户明确授权时按资源参考核实冲突归属与节点权限，不把拒绝动作记成成功。诊断降级不能代替原接受配置。

## 实践补充

V3 启动问题先区分 Python 子进程递归导入 vendor 等待 JIT 锁、CLI 旧参数歧义、具名 seam 漂移与冻结输入写入。GPU 准入同时看显存与利用率；0% 利用率但常驻大量显存不算空闲。配置预检不等于服务 ready，步骤见 [V3 实践](../../docs/v3-runtime-validation.md)。

## 升级回归与服务观察

跨版本回归先核对 [契约/测试影响与 last-good](../upgrading-sglang-hcu-upstream-first/references/contracts-and-impact.md)。模型分数先检查 [请求协议、截断与服务生命周期](../upgrading-sglang-hcu-upstream-first/references/release-validation.md)；观察会话或 SSH 中断后先核验原进程，不能重复加载或把网络超时判为模型/OOM。
