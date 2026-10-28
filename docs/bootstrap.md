# 方法库环境准备指南

> **本文用途**：指导第一次准备本方法库的环境、运行检查、调用 Skill 与工具，并区分本库和外部插件的安装步骤。项目目录见 [架构](architecture.md)，输入路径说明见 [配置](configuration.md)。

## 1. 适用场景与环境要求

| 任务类型 | 环境要求 | 运行时插件要求 |
|---|---|---|
| 阅读文档、使用 Skill | 完整仓库路径与可读资源 | 不需要 |
| 审计/清单/台账/回溯、离线测试 | Python 3.10+、Git，标准库 | 不需要 |
| 检查外部 Hook 签名或导出 targets | 指定官方 SGLang、外部 plugin 与其依赖 | 需要外部实现可导入 |
| HCU 服务/graph/性能验证 | 固定镜像、依赖、设备和指定部署组合 | 安装本仓当前包或指定历史产物；按本次范围验收 |

本仓现有 `hcu_sglang/`、`pyproject.toml` 和 `dcu` 入口。仅使用离线方法工具时无需安装运行时；开发或安装当前插件时按 [开发迁移记录](hcu-sglang-development.md) 和 [安装指南](installation.md) 执行。

## 2. 工作目录与解释器配置

所有命令从仓库根目录执行。先检查解释器版本；低于 Python 3.10 时改用支持的解释器或已经挂载本目录的开发容器。外部路径按 [路径配置](configuration.md#1-当前工具的路径配置) 填写，不依赖特定节点或用户目录。

```bash
REPO="$(pwd -P)"
python3 --version                  # 要求 3.10+
git --version
python3 ci/l0_gate.py
python3 -m unittest discover -s tests -v
```

| 成功信号 | 含义 |
|---|---|
| L0 输出 `status: passed`、空 errors | 语法、资源/文件链接与阶段结构检查通过 |
| unittest 汇总 `OK` | 本次工具契约通过，以实际输出用例数为准 |
| 工具 `--help` 正常退出 | 参数入口可用；不证明外部运行时依赖可用 |

如需隔离解释器环境，可在已确认的 Python 3.10+ 下创建 `.venv`；离线工具没有必须安装的第三方 Python 包。skill-creator 的额外 YAML 校验依赖属于维护环境，不是本库执行依赖。

## 3. 工具调用与参数检查

先阅读帮助，再按对应手册填写输入与产物目录：

```bash
python3 tools/audit_repositories.py --help
python3 tools/build_migration_manifest.py --help
python3 tools/reconstruct_sglang_das.py --help
python3 tools/verify_reconstruction.py --help
```

三方审计的完整命令见 [migration](migration.md)，源码生成见 [backtracking](backtracking.md)。审计允许记录脏输入；回溯要求两输入干净，不能把两种流程的前提混用。

## 4. Skill 配置与使用

| Skill | 输入重点 | 输出重点 |
|---|---|---|
| [hcu-plugin-assistant](../skills/hcu-plugin-assistant/SKILL.md) | 自然语言目标、已知环境与限制 | 检索匹配、必要补问与专项执行衔接 |
| [hcu-plugin-scaffold](../skills/hcu-plugin-scaffold/SKILL.md) | 官方版本、外部工程路径、命名与首项功能 | 插件工程、入口映射及启停验证 |
| [hcu-plugin-install](../skills/hcu-plugin-install/SKILL.md) | 指定产物、目标环境、用户和安装目录 | 安装记录、入口与来源、分层加载结果 |
| [hcu-plugin-dev](../skills/hcu-plugin-dev/SKILL.md) | 能力需求、版本、涉及仓库、支持范围 | 接口方案、注册与回退契约、验证计划 |
| [sglang-dcu-plugin-migration](../skills/sglang-dcu-plugin-migration/SKILL.md) | official/fork/外部 plugin/产物路径 | 基线、差异清单、处置与缺口 |
| [upgrading-sglang-hcu-upstream-first](../skills/upgrading-sglang-hcu-upstream-first/SKILL.md) | 实际已验证基线、指定 official 目标、范围与交付形态 | 接管处置、验收矩阵、产物来源与 last-good |
| [hcu-plugin-backtrack](../skills/hcu-plugin-backtrack/SKILL.md) | 干净 official/plugin、新输出路径 | 内嵌树、清单核验、未验证运行项 |
| [hcu-plugin-triage](../skills/hcu-plugin-triage/SKILL.md) | 可复现配置、启停对比、日志 | 首次分叉/根因、最小复现和回归 |

首次使用可直接描述目标并读取统一入口；无需先填写全部任务模板。明确专项入口时，向 Agent 提供对应 Skill 的路径，例如：

```text
读取 <repo>/skills/hcu-plugin-backtrack/SKILL.md。
用 <official> 和 <external-plugin> 生成 <new-output>，核验清单并记录未执行的运行时验证。
```

保留整库以解析共享 docs/tools/templates；可以把整库复制到个人目录后用新完整路径引用。不假定仅复制目录即可触发某个客户端的自动发现，也不在本次修改全局 Skill 配置。

## 5. 运行时安装与开发入口

实际安装的完整步骤统一维护在 [安装与启用](installation.md)：选兼容环境→wheel 或 editable→entry point→严格加载→模型验收。当前运行时在本仓 `hcu_sglang/`，入口为 `dcu`；旧 `output-code/sglang-plugin` 保留为迁入来源与历史验证基线。

源码所属用户与执行用户匹配时使用常规安装；需要隔离 target 的构建环境可参考 [V3 记录](v3-runtime-validation.md) §4 的任务级 metadata 方案，安装目录必须放在同一 Python 的 sys.path 中。不要为安装修改全局 safe.directory 或 HOME。

第一次写插件从 [最小案例](first-plugin.md) 开始；给现有 dcu 加能力读 [开发指南](plugins.md)，故障读 [排障入口](troubleshooting.md)。

## 6. 常见问题

| 现象 | 先检查 |
|---|---|
| Python 语法或版本错误 | 是否误用了宿主机 Python 3.6 |
| 找不到相对资源/文档 | 是否仅拷贝了某个 Skill 子目录；是否从正确根目录操作 |
| `--help` 成功，签名校验导入失败 | 外部 official/plugin 路径、解释器和可选依赖是否一致 |
| 回溯拒绝来源树 | 是否 dirty 或输出与输入父子重叠；保留现状，不先清理用户文件 |
| 组合后路径/行为异常 | 查看 import 来源、包装链和加载时机，参照 [治理](governance.md) |

下一步：选择 [阅读指南](reading-guide.md) 中的任务路线，执行后使用 [验证模板](../templates/validation-record.md) 保存结果。
