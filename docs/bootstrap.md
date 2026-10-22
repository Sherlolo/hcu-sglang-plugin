# 环境准备与快速使用

> **本文用途**：指导第一次准备本方法库的环境、运行检查、调用 skill 与工具，并区分本库和外部插件的安装步骤。项目目录见 [架构](architecture.md)，输入路径说明见 [配置](configuration.md)。

## 1. 先选择使用场景

| 你要做什么 | 所需环境 | 是否安装运行时插件 |
|---|---|---|
| 阅读文档、使用 skill | 完整仓库路径与可读资源 | 不需要 |
| 审计/清单/台账/回溯、离线测试 | Python 3.10+、Git，标准库 | 不需要 |
| 检查外部 Hook 签名或导出 targets | 指定官方 SGLang、外部 plugin 与其依赖 | 需要外部实现可导入 |
| HCU 服务/graph/性能验证 | 固定镜像、依赖、设备和指定部署组合 | 属于后续实现/集成任务 |

本仓尚无 `pyproject.toml`、运行时包和 entry point。现阶段安装方式就是获取完整源码并选择合适 Python；不要对本库执行 `pip install -e .`。

## 2. 准备目录与解释器

所有命令从仓库根目录执行。先检查解释器版本；NMZ_zz_26 默认 `python3` 为 3.6，应改用支持的解释器或已经挂载本目录的开发容器。

```bash
REPO=/public/home/yangyn1/work_space/2026/sglang_main/sglang-plugin/hcu-sglang-plugin
cd "$REPO"
python3 --version                  # 要求 3.10+
git --version
python3 ci/l0_gate.py
python3 -m unittest discover -s tests -v
```

| 成功信号 | 含义 |
|---|---|
| L0 输出 `status: passed`、空 errors | 语法、资源/文件链接与阶段结构检查通过 |
| unittest 汇总 `OK` | 当前 14 项工具契约通过 |
| 工具 `--help` 正常退出 | 参数入口可用；不证明外部运行时依赖可用 |

如需隔离解释器环境，可在已确认的 Python 3.10+ 下创建 `.venv`；离线工具没有必须安装的第三方 Python 包。skill-creator 的额外 YAML 校验依赖属于维护环境，不是本库执行依赖。

## 3. 第一次调用工具

先阅读帮助，再按对应手册填写输入与产物目录：

```bash
python3 tools/audit_repositories.py --help
python3 tools/build_migration_manifest.py --help
python3 tools/reconstruct_sglang_das.py --help
python3 tools/verify_reconstruction.py --help
```

三方审计的完整命令见 [migration](migration.md)，源码生成见 [backtracking](backtracking.md)。审计允许记录脏输入；回溯要求两输入干净，不能把两种流程的前提混用。

## 4. 使用仓库内 skill

| skill | 输入重点 | 输出重点 |
|---|---|---|
| [hcu-plugin-dev](../skills/hcu-plugin-dev/SKILL.md) | 能力需求、版本、涉及仓库、支持范围 | 接口方案、注册与回退契约、验证计划 |
| [sglang-dcu-plugin-migration](../skills/sglang-dcu-plugin-migration/SKILL.md) | official/fork/外部 plugin/产物路径 | 基线、差异清单、处置与缺口 |
| [hcu-plugin-backtrack](../skills/hcu-plugin-backtrack/SKILL.md) | 干净 official/plugin、新输出路径 | 内嵌树、清单核验、未验证运行项 |
| [hcu-plugin-triage](../skills/hcu-plugin-triage/SKILL.md) | 可复现配置、启停对比、日志 | 首次分叉/根因、最小复现和回归 |

向 agent 提供 skill 的完整路径，例如：

```text
读取 <repo>/skills/hcu-plugin-backtrack/SKILL.md。
用 <official> 和 <external-plugin> 生成 <new-output>，核验清单并记录未执行的运行时验证。
```

保留整库以解析共享 docs/tools/templates；可以把整库复制到个人目录后用新完整路径引用。不假定仅复制目录即可触发某个客户端的自动发现，也不在本次修改全局 skill 配置。

## 5. 接入已有外部 DCU plugin

以下是**参考实现仓的操作**，不是当前方法库的安装命令。先确认 pinned official 与外部 plugin 兼容，且安装到了将要运行的同一个 Python 环境：

```bash
# 只针对确实包含运行时代码和 pyproject.toml 的外部实现仓。
python3 -m pip install -e /path/to/sglang-plugin-das --no-deps
export SGLANG_PLUGINS=dcu
```

`--no-deps` 不会准备依赖，必须事先按外部仓要求完成。`dcu` 是其 entry point 名；不代表本仓已实现这个入口。导出 targets/校验签名可能执行 register，应在依赖齐全且版本明确的环境中运行。

若同时使用 adapter，固定 official/adapter/plugin SHA，核对实际 import 路径、平台判定、相同目标顺序与配置来源，再按 [组合矩阵](governance.md#3-独立与组合验证矩阵) 验证。未来本仓运行时落地后再增加对应安装步骤，见 [架构规划](architecture.md#4-未来运行时目录规划)。

## 6. 常见问题

| 现象 | 先检查 |
|---|---|
| Python 语法或版本错误 | 是否误用了宿主机 Python 3.6 |
| 找不到相对资源/文档 | 是否仅拷贝了某个 skill 子目录；是否从正确根目录操作 |
| `--help` 成功，签名校验导入失败 | 外部 official/plugin 路径、解释器和可选依赖是否一致 |
| 回溯拒绝来源树 | 是否 dirty 或输出与输入父子重叠；保留现状，不先清理用户文件 |
| 组合后路径/行为异常 | 查看 import 来源、包装链和加载时机，参照 [治理](governance.md) |

下一步：选择 [阅读指南](reading-guide.md) 中的任务路线，执行后使用 [验证模板](../templates/validation-record.md) 保存结果。
