# 交付验证记录：版本、范围与证据

> **本文用途**：记录每次交付实际改变了什么、做了哪些检查、结果适用于什么环境。用于验收和交接，不代替 [CI 规则](ci.md) 或 [验证方法](validation.md)。历史结果保持原版本含义。

## 1. V2 文档与架构修订（2026-10-21）

输入需求为 `agent-work/plugin-v2-1021.txt`；修改基线为本仓 `331f10c`。参考 adapter `dac7de1` 的治理/软件分层与 plugin-das `ce870e2` 的接入说明。

| 交付范围 | 内容 |
|---|---|
| 架构 | 两仓独立与组合运行、加载/fallback 契约、现有与未来目录、协作闭环 |
| 文档 | README 扩充；文档中心/用途/导航；配置与协作专篇；操作和验证手册重整 |
| skill 与模板 | 四个任务入口和参考同步组合契约，验证模板补目标所有权与运行组合 |
| 图文导览 | 根 architecture.html，离线可读的 SVG 关系图、目录与流程卡片 |
| 实现边界 | 不改工具、测试、门禁代码；不增加运行时业务包或外部服务 |

### V2 实际检查

| 检查 | 环境与结果 |
|---|---|
| L0 | 本地 Python 3.11、NMZ_zz_26 Python 3.10.12 均 passed；10 个 Python 文件、4 个 skill |
| 文档链接与标题锚点 | 235 个文件链接、15 个片段锚点检查通过，含 HTML 导览链接 |
| Mermaid | 11 段图使用 Mermaid 11 解析通过 |
| skill 元信息 | skill-creator quick_validate：4/4 passed |
| 浏览器导览 | 实际打开检查关系图、目录/协作卡片和页内导航，正常显示 |
| 工具回归 | NMZ_zz_26 Python 3.10.12：14 passed |
| 变更范围 | tools/ci/tests、.gitignore 与两个 V1 来源 JSON 的字节内容保持基线不变 |

远端复核使用已有 `rye_sglang_open` 容器、用户 1156:1156 与 `PYTHONDONTWRITEBYTECODE=1`，只运行 CPU gate/临时 Git fixture。本次未启动模型、调用 GPU 或测试真实 adapter/plugin 运行组合；图文描述的是设计契约，不是新运行时支持结论。

本地执行日志、链接与图语法检查汇总及远端结果留在 `artifacts/v2-20261021/`（Git 忽略）。Mermaid/DOM/锚点检查依赖仅位于维护者临时环境，没有加入本库执行依赖。旧版来源 JSON 保留快照，不改写捕获日期。

V2 修订基于 `331f10c`，包含 26 个已有文件修改和 4 个新文件；于 2026-10-22 按用户要求创建 Git 提交，未推送。

## 2. V1 方法与工具抽取（2026-10-20）
日期：2026-10-20。对象为方法库与工具契约，不是 plugin 运行时。完整测试日志位于当前 checkout 的 `artifacts/delivery-20261020/`（该目录已 Git 忽略，保留作本地执行证据）。

| 检查 | 环境 | 结果 |
|---|---|---|
| L0：10 个 Python AST、4 个 skill 与资源/文件链接、业务代码排除 | 本地 Python 3.11；NMZ_zz_26 Python 3.10.12 | passed |
| 工具行为测试 | 本地 Python 3.11；远端 Python 3.10.12 | 14 passed |
| skill-creator quick_validate | 本地隔离校验环境，PyYAML | 4/4 passed |
| agents/openai.yaml YAML 与描述长度 | 本地隔离校验环境 | 4/4 passed |
| 工具 --help | 本地 Python 3.11；远端 Python 3.10.12 | 7/7 passed |

远端使用已有 `rye_sglang_open` 容器，用户 1156:1156、`PYTHONDONTWRITEBYTECODE=1`，从 `/public4/home/yangyn1/work_space/2026/sglang_main/sglang-plugin/hcu-sglang-plugin` 读取交付树；测试仅在临时目录创建小型 Git fixture，未启动模型或调用 GPU。节点默认 Python 3.6 不满足工具要求。

14 项测试覆盖：正常回溯与 provenance、现有未知输出保护、合法生成输出显式覆盖、脏输入拒绝、anchor 漂移时不产生输出、输入输出父/子目录重叠保护、SHA256 修改/缺失、清单路径越界、校验 CLI 失败码、审计保留未跟踪文件及解析错误、merge-base 清单包含继承硬件漂移和 WORKTREE、AST/runtime-export 不误晋级 implemented、脏删除符号记录、official-ref 与 worktree HEAD 不一致时拒绝。

本次没有执行外部 plugin 的真实注册/目标解析/签名兼容、DeepSeek 方法同步、模型服务、GPU 数值/graph/TP 或性能 lane。这些验证属于后续实现任务，不继承历史 214 项 plugin 测试或 MiniMax TP4 数字作为本次结果。`--help` 成功不证明外部依赖可用。
