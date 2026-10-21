# 交付验证记录

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
