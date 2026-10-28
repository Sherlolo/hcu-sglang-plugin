# sglang-das 模块盘点与插件迁移规划指南

> **本文用途**：指导研发同事使用模块盘点网页，评审 `sglang-das → sglang-plugin` 的迁移范围、依赖与验收计划，并说明如何重新采集。当前方法库只维护盘点、计划与方法，实际插件实现位于独立外部工程。

## 1. 研发使用入口

打开 [迁移盘点与开发计划网页](../migration-plan.html)，按下面顺序使用：

1. 在“迁移总览”确认三方 HEAD、当前树文件关系和历史验证边界。
2. 在“模块盘点”选择工作域，查看职责边界、源目录、现有插件静态线索、前置模块和验收标准。
3. 在“源码明细”按路径、类名、函数名或合同目标检索，展开源码行号、对象 ID、声明位置及来源哈希线索；可导出当前筛选为 CSV。
4. 在“开发计划”按 M0–M4 评审、裁剪并认领建议工作项，确定本轮实际交付范围。
5. 使用下述 Skill 请求进入差异审阅和能力开发，完成后按本次证据维护实际工单与交付记录。

网页可离线打开，无外部脚本、字体或数据请求。完整结构化记录见 [原始盘点 JSON](../plans/sglang-das-migration.json)。人员、资源和迭代容量尚未给定，因此建议项不含虚构负责人、工期或完成率。

## 2. 当前采集范围与基线

| 输入 | 固定 HEAD | 受版本管理路径 |
|---|---|---|
| HCU 来源 `sglang-das` | `15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e` | 11,418 |
| 官方目标 SGLang | `baa9befa381511cd4d56f83980951e5a45fffc6d` | 10,594 |
| 现有插件 | `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c` | 685 |

2026-10-28 采集时三份输入均干净。运行时范围为 `python/sglang/`：来源共 5,937 个路径，其中 4,714 个 Python 文件；比较清单含 1,614 个不同路径（包含仅官方目标中的路径）。全部两树比较清单共 11,497 个路径：同内容 8,487、同路径有差异 2,028、仅 HCU 来源 903、仅官方目标 79。

按 26 个研发工作域组织完整清单，提供 78 条待评审建议工作项。现有插件的 363 个声明合同目标均能定位到 `.py` 模块或经顶层定义核对的包入口；450 个来源文件存在合同、来源清单或 native 字节映射线索。**这些数量不是迁移完成率或实际运行覆盖率。**

统计只读固定 HEAD 的 Git 对象。工作区 dirty 状态另行记录，未提交内容不进入清单，不要求清理来源来满足审计。子模块仅记录 Gitlink 版本，不读取其内部文件。当前树比较未自动归因上游版本变化、重命名或完整调用链行为，差异须继续人工评审。

## 3. 模块归属与实施阶段

运行时分为启动参数、模型与配置、执行与装载、Attention、KV、量化、MoE、基础层、通信、PD、调度、Graph、推测解码、采样、native/JIT/AOT、LoRA、多模态、可观测及公共设施等领域；另保留客户端、Gateway/Rust、测试评测、打包、CI、文档和其他组件清单。

默认复用未变化的官方能力。已有行为的平台兼容由 adapter 按职责承担，新增能力通过 plugin 正式 seam 或窄 Hook 接入；独立 Gateway、客户端或上游通用演进不整体复制到插件。每个模块的具体前置依赖、建议承载位置、任务与验收门槛维护在网页中。

| 阶段 | 主要目标 | 准入/退出依据 |
|---|---|---|
| M0 | 基线、当前差异归因与边界评审 | 明确复用、上游、adapter、plugin、独立组件归属 |
| M1 | 入口配置、native 构建、基础层与量化 | 支持/回退契约及真实调用可复现 |
| M2 | Attention、KV、Graph 与目标模型 | 真实构造/消费者、变输入和指定服务分别验收 |
| M3 | CP/EP/PD、推测解码及多模态等扩展 | 按实际拓扑、模型和完整配方执行专项验证 |
| M4 | 源码/wheel/embedded 与持续交付 | 本轮必测闭合、产物身份、CI、许可及回退记录 |

阶段不要求将所有扩展都纳入同一版本。当前已有 MiniMax TP4 历史配置结果；完整 DeepSeek 引擎、完整 CP/PD 模型、DSpark streaming/source-event/ACK、W4A8/DeepEP 共享专家、FP4 AOT 和全部 native/gfx 分支仍需独立验收，范围以 [V3 最终交付](v3-20261026-development.md#8-最终交付与需求验收) 为准。本轮未重新执行模型、GPU 或 Hook apply。

## 使用 Skill 评审并实施迁移任务

主要入口为 [sglang-dcu-plugin-migration](../skills/sglang-dcu-plugin-migration/SKILL.md)。需要实施新增能力时再衔接 [hcu-plugin-dev](../skills/hcu-plugin-dev/SKILL.md)，指定官方升级或 embedded 交付分别按对应专项入口。尚不清楚模块归属时使用 [统一任务助手](../skills/hcu-plugin-assistant/SKILL.md)。

**输入**：本轮模块或任务 ID、三方路径与版本、期望行为/支持范围、实际资源和证据位置。可访问版本与日志由 Agent 读取；方法咨询不要求先填写全部环境字段。

```text
读取 skills/hcu-plugin-assistant/SKILL.md。
按 migration-plan.html 的 attention 模块评审本轮迁移范围。
三份源码：<官方、DAS 来源和外部插件实际路径>。
先使用三方迁移方法核对当前差异、已有映射与缺口；
确认上游/adapter/plugin 归属及验收矩阵，再用开发 Skill 实施本轮选定任务。
源码实现应位于外部插件工程；缺少影响写入或验证的信息时补问。
```

**交付**：本轮版本、差异归因与符号映射、模块任务与依赖、实现及支持/回退、真实验证证据和未覆盖项。静态合同、来源声明、资源字节相同与运行验收分别记录，不把旧日志或 AST 映射升级成当前功能完成证明。

## 4. 重新采集与维护

先按 [路径配置](configuration.md#1-当前工具的路径配置) 设置本地输入，从方法库根目录执行：

```bash
: "${OFFICIAL:?请设置固定官方源码路径}"
: "${FORK:?请设置 HCU 来源路径}"
: "${PLUGIN:?请设置外部插件实现路径}"
python3 tools/build_migration_plan.py \
  --official "$OFFICIAL" --fork "$FORK" --plugin "$PLUGIN"
python3 tools/build_migration_plan.py --check
python3 tools/build_guide.py
```

[采集工具](../tools/build_migration_plan.py) 使用 Python 3.10+ 和 Git，读取对象而不导入 SGLang/vendor。输出默认是 `plans/sglang-das-migration.json` 与 `migration-plan.html`，必须在三个输入仓库之外。输入 Git 操作使用 `GIT_OPTIONAL_LOCKS=0`，不更新来源 refs、工作区或索引缓存。

研发模块规则、职责与建议任务维护在 [计划定义](../templates/sglang-das-migration-plan.json)，页面样式交互维护在 [网页模板](../templates/migration-plan.html)。修改计划定义或来源版本后重新采集，避免保留过时分类和任务。仅调整样式时可使用保存的数据重新渲染：

```bash
python3 tools/build_migration_plan.py --render-only
python3 tools/build_migration_plan.py --check
python3 tools/build_guide.py
python3 tools/build_guide.py --check
python3 ci/l0_gate.py
python3 -m unittest discover -s tests -v
```

`--check` 只读核对当前计划定义、保存的数据与模板生成页是否一致，不访问输入仓库，也不证明来源 HEAD 仍是最新。源码变化后必须显式重新采集。生成页和原始数据是可共享快照；不要手改它们来标记任务完成。

## 5. 本次质量记录

源码采集、统计交叉核对、工具契约、静态线索和无网络 DOM 交互分别留证；浏览器预览使用只读、仅提供本页的回环 HTTP 服务。验收结果记录在 ignored 目录 `artifacts/migration-plan-20261028/`，以其中最终报告和日志为准。三份原始输入的 HEAD、工作区与索引保持原状；方法库中的网页、计划和工具变更另行提交。本页建设不等于已执行全量源码转换。
