# hcu-sglang-plugin

SGLang HCU 插件代码与开发方法仓库。运行时代码位于 `hcu_sglang/`，同时维护开发、安装、排障和升级文档、Skill 与方法工具。2026-10-28 起本仓提供可安装的 `hcu-sglang-plugin`，入口 `hcu = hcu_sglang:register`；迁入步骤、提交和验证见 [开发迁移记录](docs/hcu-sglang-development.md)。

面向研发同事评审 `sglang-das → sglang-plugin` 时，先打开 [模块盘点与开发计划](migration-plan.html)：可查看三方基线、26 个工作域、完整源码与符号明细，以及分阶段建议任务。统计和维护方法见 [迁移规划指南](docs/sglang-das-migration-plan.md)。

## 1. 首次使用

保留完整仓库，在 Agent 中打开或提供仓库位置，然后描述你想完成的工作。无需先了解全部文档或记住专项 Skill 名称。

推荐从统一入口 [`hcu-plugin-assistant`](skills/hcu-plugin-assistant/SKILL.md) 开始。在 Agent 已打开本仓库时，直接发送：

```text
读取 skills/hcu-plugin-assistant/SKILL.md，按该 Skill 执行。
我的目标：开发一个新的 SGLang HCU 插件，先完成最小工程与启停验证。
请先检索相关 Skill 和文档；缺少必要输入时向我提问，再继续执行。
```

若 Agent 未打开本仓库，将第一行改为 `读取 <方法库路径>/skills/hcu-plugin-assistant/SKILL.md`，把占位符替换为实际位置。客户端已经发现该 Skill 时也可用 `$hcu-plugin-assistant`；路径方式不依赖自动发现。不要只复制一个 Skill 子目录，它依赖同仓的文档、工具和模板。

统一入口会按下面的方式工作：

1. 识别你要咨询方法、执行开发还是排查故障，检索当前仓库的 Skill 与相关文档。
2. 选择匹配的专项入口，说明使用理由；优先读取已有环境与日志，补问当前阶段真正缺少的信息。
3. 读取专项 `SKILL.md` 后继续执行，按需要衔接后续任务，交付产物、验证结果和未完成条件。

它是 Agent 的执行指引，不是独立服务或后台调度器。只询问方法时，可以直接得到阅读路线和示例；要求实际创建、安装或启动服务时，才需要对应路径、版本及环境信息。详细匹配与补问规则见 [任务与 Skill 使用指南](docs/reading-guide.md)。

## 2. 功能与适用场景

本库包含 **1 个统一入口和 7 个专项 Skill**。已明确任务时，可直接读取下表专项入口，跳过统一入口。

| 任务类型与适用场景 | 操作指南 | 对应 Skill | 交付结果 |
|---|---|---|---|
| 第一次创建插件包，验证接入链 | [工程初始化](docs/first-plugin.md) | [hcu-plugin-scaffold](skills/hcu-plugin-scaffold/SKILL.md) | 外部工程、打包入口、实际 Hook 调用与启停结果 |
| 安装已有源码或 wheel，检查是否真正生效 | [安装与启用](docs/installation.md) | [hcu-plugin-install](skills/hcu-plugin-install/SKILL.md) | 安装记录、入口与模块来源、加载及要求的服务验证 |
| 给已有插件增加融合、后端或专项能力 | [能力开发](docs/plugins.md) | [hcu-plugin-dev](skills/hcu-plugin-dev/SKILL.md) | 实现、注册与回退契约、针对性测试和证据 |
| 出现错误答案、异常、Graph 或多卡回归 | [故障诊断](docs/troubleshooting.md) | [hcu-plugin-triage](skills/hcu-plugin-triage/SKILL.md) | 最小复现、首次分叉、根因及修复回归 |
| 比较 official/fork/plugin，整理迁移范围 | [三方迁移](docs/migration.md) | [sglang-dcu-plugin-migration](skills/sglang-dcu-plugin-migration/SKILL.md) | 固定基线、差异清单、处置台账与缺口 |
| 升级指定官方版本，组织交付与验收 | [版本升级](docs/upgrading.md) | [upgrading-sglang-hcu-upstream-first](skills/upgrading-sglang-hcu-upstream-first/SKILL.md) | 补丁接管处置、验收矩阵、产物来源与有效回退版本 |
| 从 official + plugin 生成内嵌源码 | [源码回溯](docs/backtracking.md) | [hcu-plugin-backtrack](skills/hcu-plugin-backtrack/SKILL.md) | 新源码树、来源清单、哈希核验与独立运行验证记录 |
| 查设计原则或相似故障经验 | [架构](docs/architecture.md)、[经验索引](docs/development-methodology.md) | 统一入口检索，或直接阅读 | 对应章节、检查方法及适用边界 |

Skill 按实际目标选择，不构成必须全部执行的流水线。已有插件增加能力不需要重新创建包；普通安装不要求三方迁移。adapter 与 plugin 独立维护、可以组合运行，组合兼容需单独验证，见 [职责治理](docs/governance.md)。

## 3. 仓库文件结构

```text
hcu-sglang-plugin/
├── README.md                        使用入口、功能与案例
├── AGENTS.md                        Agent 工作约定
├── hcu_sglang/                      HCU 插件运行时代码与原生资源
├── pyproject.toml                   安装、入口与资源打包定义
├── architecture.html                离线网页导览、检索与文档阅读器
├── migration-plan.html              DAS 模块盘点与插件迁移开发计划
├── plans/                           可复核的固定版本盘点 JSON
├── docs/
│   ├── README.md                    完整文档目录与维护规范
│   ├── reading-guide.md             任务匹配、输入与 Skill 衔接
│   ├── first-plugin.md              最小插件开发案例
│   ├── installation.md              已有插件安装与启用
│   ├── plugins.md / troubleshooting.md  能力开发与故障诊断
│   ├── migration.md / upgrading.md / backtracking.md
│   ├── development-methodology.md   历史经验与故障案例索引
│   └── 其他设计、规范、来源清单与验证记录
├── skills/
│   ├── hcu-plugin-assistant/         统一任务入口
│   └── 各专项 Skill/                SKILL.md、agents/ 与按需 references/
├── tools/                           审计、迁移、生成、验证及网页构建工具
├── templates/                       处置、历史、验证模板与网页源模板
├── tests/                           离线工具契约测试
├── ci/l0_gate.py                    结构、链接与 Skill 元信息门禁
├── .workflow/l0-gate.yml             中立 runner 配方，尚需接入实际 CI
├── licenses/                        来源许可证
└── artifacts/                       本地证据与日志，Git 忽略，按需产生
```

`docs/` 负责解释与指导，`skills/` 规定 Agent 如何选择和执行，`tools/` 提供可运行的方法实现，`templates/` 用于保存本次真实记录。历史经验不是当前版本的兼容性承诺，ignored 证据也不会随 Git clone 自动分发。

官方源码、HCU fork 和生成目录可放在任意位置；当前运行时实现位于本仓 `hcu_sglang/`。通过 `OFFICIAL`、`FORK`、`PLUGIN`、`OUTPUT`、`ARTIFACTS` 等本地变量传入，见 [路径配置](docs/configuration.md#1-当前工具的路径配置)。本仓不固定个人目录、机器名称、容器或 UID。

## 4. 开发使用案例

### 4.1 创建并开发新插件

从第 1 节的统一入口请求开始。假设目前只有“先跑通最小插件”的目标，助手会选择 `hcu-plugin-scaffold` 和 [最小案例](docs/first-plugin.md)，检查已有信息，再补齐官方源码与目标版本、外部新工程目录和 Python 环境。若目录尚未确定，会先询问写入位置，期间仍可说明结构和接入方式。

资料齐备后，Agent 在外部目录创建工程，完成包安装、entry point 发现、Hook 实际调用及开关对照。若你进一步要求实现 HCU 功能，再衔接 `hcu-plugin-dev`，明确支持条件与回退方式并执行相关验证。工程初始化通过不代表完整模型已经验收。

已有信息可直接追加到请求中，减少补问：

```text
官方源码与版本：<实际路径及版本或 SHA>。
外部新工程目录：<可写的新目录>。
首项功能：<功能描述，或采用文档中的最小示例>。
执行环境：<Python/容器>；证据目录：<本次任务目录>。
```

这些字段按任务逐步补充，无需在咨询阶段全部填完。distribution、Python 包和入口名称未指定时，助手按初始化规则提出常规命名并记录。

### 4.2 安装已有插件

```text
读取 skills/hcu-plugin-assistant/SKILL.md。
我要安装已有插件，产物是 <wheel 文件或源码目录>，目标环境是 <Python/容器>。
请确认正确的安装方式并验证入口、模块来源和加载结果；本次不要求模型服务。
```

助手选择 `hcu-plugin-install`，核对实际版本、兼容依赖和可写安装位置。当前包可从本仓安装；使用旧版外部产物时须核对对应命名空间和入口。完整命令维护在 [安装指南](docs/installation.md)。

### 4.3 排查运行异常

```text
读取 skills/hcu-plugin-assistant/SKILL.md。
插件已加载，但 TP4 服务输出错误。实现仓与官方源码：<路径>。
触发命令和日志：<已有文件位置>；预期输出：<预期行为>。
先检索本仓相似案例，再定位本次首次分叉并验证修复。
```

助手会区分安装层与运行层问题，选择诊断 Skill，并按症状查询经验索引。Graph、TP、多卡、响应校验等历史方法只作为线索；修复必须回到本次输入和真实运行证据上。

## 5. 不使用 Agent 的操作方式

直接从 [文档中心](docs/README.md) 或 [任务指南](docs/reading-guide.md) 选择手册，按其中输入、命令和验收标准操作。离线方法工具要求 Python 3.10+ 和 Git，先查看所需工具的 `--help`：

```bash
# 从本方法库根目录执行；选择当前任务需要的工具。
python3 tools/audit_repositories.py --help
python3 tools/reconstruct_sglang_das.py --help
python3 tools/validate_plugin_runtime.py --help
```

审计、清单和源码生成大多可离线执行；运行时加载验证需要对应 SGLang、插件及 vendor 依赖。GPU 与完整服务验证还需要匹配硬件、模型和配方，不能以工具测试通过替代。

## 6. 网页导览与维护检查

直接用浏览器打开 [architecture.html](architecture.html)。页面可离线检索文档、选择任务、复制统一入口请求和阅读资料；内置阅读器中的链接可继续跳转，不依赖浏览器原生渲染 Markdown。目录侧栏在窄屏通过导航按钮展开。详细规则的来源仍是仓库原文。

网页是文档快照。修改 README、Skill、文档或网页模板后，从本仓根目录重新构建并检查：

```bash
python3 tools/build_guide.py
python3 tools/build_guide.py --check
python3 ci/l0_gate.py
python3 -m unittest discover -s tests -v
```

网页源模板为 `templates/architecture.html`，内容从当前共享文件生成，无外部 CDN 或网络字体依赖。`--check` 只检查快照是否需要更新，不写入文件。完整环境要求见 [环境准备](docs/bootstrap.md)，协作与验收规范见 [协作流程](docs/contributing.md) 和 [CI](docs/ci.md)。

## 7. 运行时交付与历史证据

外部运行时最近记录为 v0.3.2.dev5（`afc6205`），固定官方基线为 `baa9befa`。范围、产物和未验证项见 [最终交付记录](docs/v3-20261026-development.md#8-最终交付与需求验收)，安装条件见 [安装指南](docs/installation.md)。本次方法库导航更新不重新声明 GPU、模型精度或性能结果，也不提供公共 PyPI 发布承诺。

来源与许可见 [来源地图](docs/extraction-map.md) 和 [来源许可证](licenses)。方法查询先使用 [开发经验索引](docs/development-methodology.md)；复核某次交付时再按日期读取实施记录和原始证据。
