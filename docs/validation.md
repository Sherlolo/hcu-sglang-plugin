# 验证与排障：单仓契约、组合兼容与运行证据

> **本文用途**：定义每层验证的输入、断言与结论范围，指导精度、Graph、分布式和性能问题的定位。执行入口见 [CI](ci.md)，设计约束见 [治理](governance.md)，记录格式见 [validation-record](../templates/validation-record.md)。

## 1. 验证环境与运行组合

记录 official/plugin SHA，使用 adapter 时追加 adapter SHA；同时记录 dirty、实际 import 路径、镜像/挂载、Python/torch/DTK/vendor、硬件、模型/数据版本和完整启动参数。比较前先确保输入和状态对齐。

| 运行组合 | 验证目的 | 适用性 |
|---|---|---|
| 官方参照 | 行为与指标基线 | 在该参照环境可用时执行 |
| 官方 + adapter | HCU 适配基线 | 目标部署使用 adapter 时执行 |
| 官方 + plugin | plugin 声明的独立工作能力 | 仅对明确支持此组合的能力执行 |
| 官方 + adapter + plugin | 真正部署的组合行为 | 目标部署使用两仓时必须单独验收 |

adapter 与 plugin 的独立测试都通过，仍不能替代组合测试。依赖 adapter 的能力不强行要求“官方 + plugin”通过；用 not-applicable 说明设计条件。

## 2. 分层验证矩阵

| 层 | 关键断言 | 通过意味着 | 不能替代 |
|---|---|---|---|
| 方法库 L0 | AST、Skill 资源、文件链接、阶段边界 | 仓库结构可用 | 可读性、外部依赖、运行正确性 |
| 工具 CPU | 隔离 Git fixture、清单/台账、拒绝保护、哈希损坏 | 指定工具行为契约 | 真实 plugin 接线或 GPU |
| 注册与解析 | entry point、完整预期集合、target、签名/别名/顺序 | 已接到固定框架版本 | 数值与状态正确 |
| 单能力行为 | reference、shape/dtype/mode、主动/回退与副作用 | 能力在覆盖范围内正确 | 未覆盖模型/并行形态 |
| 组合接缝 | adapter 基线保留、包装链、重复目标、配置来源 | 所选版本组合可装配 | 服务数值或性能 |
| Graph | capture 后变输入、变状态 replay 与 reference 一致 | 捕获与复用语义正确 | 仅 eager 的泛化 |
| 服务集成 | 加载、prefill/decode、TP/DP/EP/PP/PD、量化/KV | 指定模型配置可运行 | 完整精度/性能指标 |
| 精度与性能 | 固定数据、阈值来源、预热 A/B、TTFT/ITL/吞吐/显存 | 所测配置有指标证据 | 任意版本/卡型支持 |

## 3. 接线和组合的检查顺序

```mermaid
flowchart LR
    I[确认实际导入来源] --> R[从顶层入口导出注册集合]
    R --> A[解析并应用目标 / 检查顺序]
    A --> F[能力启用与停用的命中证据]
    F --> S[当前基线与状态副作用对拍]
    S --> G[graph / 服务 / 指标]
```

AST 候选、register 成功、apply 成功、实际分支执行和语义正确分别留证。停用 plugin 后必须保留该部署的有效基线；启用 adapter 时，不应因 fallback 私自导入“原版”而绕过适配。

## 4. 精度、cache 与 Graph 定位

| 观察 | 优先检查 | 证据方式 |
|---|---|---|
| 所有路径都有差异 | 算子对 M/批量是否 invariant、量化/归约顺序 | 独立算子 reference 与 shape 对比 |
| 只有 prefill cache hit 异常 | restore、滑窗、conv/Mamba/checkpoint | 与重算逐状态比较，覆盖边界 |
| 某层开始分叉 | token/位置对齐后该层输入相同、输出首次不同 | 逐层探针先证明命中与输入对齐 |
| 仅 Graph 出错 | capture 绑定旧输入、状态复用、同步探针干扰 | Graph 外状态探针、变输入 replay |

先确认 embedding/token 对齐，再寻找首次分叉，不从下游误差倒推根因。基础 match、prefill cache hit、decode cache hit 分开测；阈值按模型、TP 和数值条件标定，不能把 TP=1 结论直接迁到多卡。

Graph 回放不执行 Python forward hooks。关闭 Graph 得到一致结果不能排除 Graph 问题；capture 中 `.item()` 等同步会破坏探针。记录探针命中数和问题在探针下是否仍复现。更多操作见 [精度参考](../skills/hcu-plugin-triage/references/precision-and-graph.md)。

## 5. Hang、crash 与性能定位

| 问题 | 推荐顺序 |
|---|---|
| 多 rank 卡住 | 先查 OOM/已退出 rank，再比最后 collective、shape、顺序和 batch/cache 状态，找首次分叉 |
| crash/非法内存 | 调用前保留 shape/dtype/stride/device 和最小必要输入；识别无法仅靠 tensor 重放的模块状态 |
| 性能退化 | 先对齐环境/版本/实际分支，再拆冷编译、预热和稳定测量，最后看 CPU dispatch/kernel/通信/同步占比 |
| 某版本回归 | 相同失败签名下找最后 pass 与首次 fail，排除硬件/依赖变化，在独立 checkout 缩小范围 |

HIP 环境优先使用已有 watchdog、Python 栈与 ROCm/HCU 工具；不要把 CUDA coredump/cuda-gdb 指令当作 HCU 通用命令。详细方法见 [分布式与性能参考](../skills/hcu-plugin-triage/references/distributed-and-performance.md)。

## 6. 状态与交付结论

| 结果状态 | 含义 |
|---|---|
| passed | 本次在记录环境下执行且满足断言 |
| failed | 执行后未满足断言，保留签名与日志 |
| pending | 缺资源、依赖或尚未执行，写解除条件 |
| not-applicable | 该能力设计上不适用，关联支持范围和理由 |

性能记录分离预热/编译和测量，报告相同环境下相对基线及正确性。短回答、冷启动或不同机器绝对 tok/s 不构成加速证据。按 [模板](../templates/validation-record.md) 填写；历史交付与本次结果见 [交付记录](delivery-validation.md)。

## 升级与发版验收入口

版本升级使用 [升级记录模板](../templates/upgrade-release-record.md) 将补丁处置与本次必测行关联；provider/fallback、源码/交付产物、单仓/adapter 组合分别留证。模型协议、原始分数、服务中断恢复与实际发布形态的补充方法见 [发版验收](../skills/upgrading-sglang-hcu-upstream-first/references/release-validation.md)。沿用本文状态，pending 另注 blocked/not-run；不以历史成绩、配置预检或单次 HTTP 200 关闭整个矩阵。

## 7. 确定性生成响应的离线验收

对有明确短文本 oracle 的非流式 OpenAI chat completion，可用 [validate_generation_response.py](../tools/validate_generation_response.py)。它只读取已保存的原始 JSON，要求单一 assistant 文本、正常 `finish_reason=stop`，拒绝控制字符、tool/refusal 和错误最终答案。接受项是**完整最终文本**，不搜索任意位置的数字或关键词；仅去掉两端空白。

从本方法库根目录执行，替换响应、模型名与 oracle：

```bash
python3 tools/validate_generation_response.py \
  --response "${ARTIFACTS:?请设置本次证据目录}/arithmetic.json" \
  --expected-model "${SERVED_MODEL_NAME:?请设置实际服务模型标识}" \
  --expected-final-text 2 \
  --output "$ARTIFACTS/arithmetic-check.json"
```

若该 checkpoint 的实际协议把推理保存在 content 中，且本行允许一个完整的前导 `<think>...</think>`，显式加 `--allow-think-prefix`；未闭合、嵌套或含控制字符仍失败。独立 `reasoning_content` 保留为原字段，不把某模型的 `enable_thinking=False` 当作所有 checkpoint 都支持的协议。确实允许多个标点/拼写版本时，重复 `--expected-final-text` 枚举它们，而不是验证失败后临时放宽。

工具返回 passed/failed、提取的最终文本、停止原因、模型和原文件 SHA256；0 表示该响应满足 oracle，1 表示响应读取/格式/语义失败，2 表示参数或报告写入错误。报告不得覆盖原响应，包括 symlink/hardlink 指向同一文件。原始证据不归一化、不改写。

该工具不证明 HTTP 状态、监听者归属、cache hit、长上下文、其他样本或清理；这些仍在服务行分别记录。流式、多 choice、工具调用、开放题与数据集评分不属于此工具的接受范围，按任务选择其专门 oracle。服务里曾发生的包含数字误判及负例回放见 [V3 实践](v3-runtime-validation.md) §17；按症状的起步路线见 [排障入口](troubleshooting.md)，历史修复经验见 [经验索引](development-methodology.md)。
