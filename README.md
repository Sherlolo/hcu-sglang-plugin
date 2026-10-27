# hcu-sglang-plugin

SGLang HCU 插件开发方法库，提供开发与部署文档、任务 Skill 和工程工具。运行时实现独立维护；本仓库不包含插件业务代码或打包定义。

## 任务导航

| 任务类型 | 操作指南 | 对应 Skill | 验收标准 |
|---|---|---|---|
| 插件安装与启用 | [安装与启用指南](docs/installation.md) | [hcu-plugin-install](skills/hcu-plugin-install/SKILL.md) | 安装产物、入口和模块来源正确；加载及要求的服务验证通过 |
| 新插件工程初始化 | [插件工程初始化指南](docs/first-plugin.md) | [hcu-plugin-scaffold](skills/hcu-plugin-scaffold/SKILL.md) | 外部工程完成打包、入口发现、Hook 应用与启停验证 |
| 插件能力开发 | [插件能力开发指南](docs/plugins.md) | [hcu-plugin-dev](skills/hcu-plugin-dev/SKILL.md) | 实现、支持/回退契约及相关运行验证完成 |
| 故障诊断与回归修复 | [故障诊断指南](docs/troubleshooting.md) | [hcu-plugin-triage](skills/hcu-plugin-triage/SKILL.md) | 形成最小复现、根因分析、修复与回归证据 |
| 三方差异审计与迁移 | [三方迁移指南](docs/migration.md) | [sglang-dcu-plugin-migration](skills/sglang-dcu-plugin-migration/SKILL.md) | 基线、差异清单和处置台账可追溯 |
| 官方版本升级与发版 | [版本升级与发版指南](docs/upgrading.md) | [upgrading-sglang-hcu-upstream-first](skills/upgrading-sglang-hcu-upstream-first/SKILL.md) | 补丁处置、产物来源与必测验收矩阵完成 |
| 内嵌源码生成与核验 | [源码回溯指南](docs/backtracking.md) | [hcu-plugin-backtrack](skills/hcu-plugin-backtrack/SKILL.md) | 生成独立源码树并完成来源、哈希及要求的运行验证 |
| 历史开发经验查询 | [开发经验与故障案例索引](docs/development-methodology.md) | 按具体任务选择 | 根据症状检索检查方法、修复经验及原始记录 |

各操作指南均包含 Skill 执行说明，提供输入要求、调用示例和预期交付。Skill 之间按任务衔接，不构成必须依次执行的固定流程；完整对应关系见 [任务与 Skill 使用指南](docs/reading-guide.md)。

## 仓库职责与目录边界

下表表示历史交付的相对目录布局，不要求复制该布局。实际工作目录通过 `REPO`、`OFFICIAL`、`PLUGIN` 等变量指定，配置方法见 [路径配置](docs/configuration.md#1-当前工具的路径配置)，本地变量按该页说明填写。

| 目录 | 职责与使用边界 |
|---|---|
| `hcu-sglang-plugin`（本仓库） | 文档、7 个 Skill、9 个方法工具和 CPU 工具测试；不作为 `pip install -e .` 的安装对象 |
| `output-code/sglang-plugin` | 独立运行时实现仓库；包含 `pyproject.toml` 和 `sglang_dcu_plugin`，distribution 为 `sglang-dcu-plugin`，entry point 为 `dcu` |
| `input-code/sglang` | 配套官方 SGLang 源码；安装与运行需固定兼容版本 |

2026-10-26 需求当前交付为插件 v0.3.2.dev5（`afc6205`），正式输出目录、安装产物和验证范围见 [最终交付记录](docs/v3-20261026-development.md#8-最终交付与需求验收)。v0.3.1 保留为回退版本。

历史 V3 已验证组合为官方 `baa9befa` 与插件 `45e67e0`（v0.3.0）。完整 SHA、依赖、验证范围及限制见 [安装与启用指南](docs/installation.md) 和 [V3 实践记录](docs/v3-runtime-validation.md)。该记录不构成任意 SGLang 版本或 HCU 镜像的兼容声明；当前未提供公共 PyPI HCU wheel 发布承诺。

## Skill 调用方式

Skill 是 Agent 的任务执行指引。保留完整仓库以解析共享文档、工具和模板，按绝对路径读取对应 `SKILL.md` 即可；仅复制 Skill 子目录可能导致共享资源缺失。不依赖客户端自动发现。已被客户端发现的 Skill 可用 `$skill-name` 调用，其余情况使用下述路径方式。

以初始化新插件工程为例，替换所有占位符后提交给 Agent：

```text
读取 <方法库绝对路径>/skills/hcu-plugin-scaffold/SKILL.md，按该 Skill 执行。
官方源码：<official 绝对路径及 SHA>。
外部工程目录：<方法库与官方源码目录之外的新目录>。
distribution / Python 包 / entry point：<三个名称>。
首项功能：<功能描述或最小示例>；执行环境：<Python/容器>。
证据目录：<本次任务证据目录>。
完成工程创建、隔离安装、实际调用及启停验证，记录未验证项。
```

其他任务的调用示例维护在上表对应操作指南中，避免将所有执行细节集中在导航页面。

## 方法库维护与质量检查

使用 Python 3.10+ 和 Git，在仓库根目录执行：

```bash
python3 ci/l0_gate.py
python3 -m unittest discover -s tests -v
```

上述检查覆盖方法库结构与工具契约。模型、GPU 和运行时验收按对应任务执行。解释器与工具环境见 [环境准备指南](docs/bootstrap.md)。

文档索引见 [文档中心](docs/README.md)，静态导览见 [architecture.html](architecture.html)。组件职责、adapter 组合与协作约定见 [架构设计](docs/architecture.md)、[职责治理](docs/governance.md)、[AGENTS.md](AGENTS.md) 和 [协作流程](docs/contributing.md)。来源、许可和历史交付见 [来源地图](docs/extraction-map.md)、[来源许可证](licenses) 及 [交付记录](docs/delivery-validation.md)。
