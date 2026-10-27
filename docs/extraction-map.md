# 方法来源与版本演进

> **本文用途**：解释文档、skill 和工具从何而来、哪些被改写或排除，以及 V1 抽取快照与 V2 文档设计的关系。用于来源审计和方法维护；实际验证见 [交付记录](delivery-validation.md)。

## 1. V1 抽取基线与来源

日期：2026-10-20。所有来源为历史验证节点的本地快照，未拉取网络版本。各树读取时均干净。机器可读基线与文件来源见 [source-manifest](source-manifest.json)，DAS 的 25 个 skill 全量索引、原始描述、SHA256、取舍与目的地见 [source-inventory](source-inventory.json)。

| 来源 | 抽取对象 | 新仓落点 |
|---|---|---|
| hcu-sglang-adapter | README/docs/skills/references/tools/tests/ci 组织；双仓判别与分层证据 | README、architecture、governance、reading-guide、ci |
| sglang-plugin-das | entry point → register → apply 生命周期；Hook 与 fallback；迁移 skill 的五个脚本 | plugins、migration、sglang-dcu-plugin-migration、tools |
| agent_work/plugin_work | reconstruct 与 5 个原测试、三方 repo_audit、历史回溯总结 | tools/reconstruct、tests、backtracking |
| sglang-das/.claude/skills | 12 项方法按 plugin 场景改写；13 项仅索引说明边界 | dev / triage references、ci、source-inventory |

`plugin_work/skills/sglang-dcu-plugin-migration` 与实现仓 skill 逐文件比较一致；旧 `repo_audit.py` 与 skill 的 `audit_repositories.py` 功能接近但并非相同：前者提取 register/plugin_hook 的字面目标，后者索引所有包含 sglang 的字符串。新工具合并保留 `literal_hook_targets` 与 `sglang_target_strings` 两类信息，避免重复维护入口。历史 history/manifest/巨大审计结果仅用于理解方法；新仓提供空 overrides 与证据模板，不把历史“verified”当作新能力验证。源码内嵌的结果仓库、插件业务代码、容器启动脚本与模型结果文件不搬入。

## 2. V1 取舍与工具修正

- 参考 plugin 的旧中文简析描述 DeepSeek 未接线，当前顶层源码已经接线；新文档按 pinned SHA 与调用链描述。
- 旧迁移 skill 的范围同时覆盖平台适配与新增能力，提取后增加 adapter/plugin 选位步骤，不继承其混合边界。
- skill 入口重写为四个任务入口，共享脚本只存 tools；详细规则放按需 references。
- ledger 明确 AST 候选与 runtime-export 的计数来源；仅模块匹配不再自动声称 active/implemented/属性已解析，保持 pending。
- manifest 要求 official-ref 与 official 工作树 HEAD 一致，避免比较混合基线；脏删除不再回退到旧 HEAD 内容。
- reconstruct 新增输出与来源父/子目录重叠拒绝，其余沿用原生成机制；新增独立 SHA256 verifier。
- 未搬移整个上游 skill 引擎或其自动运维命令，保留适用于插件的决策方法；具体可执行脚本范围以 README 工具表为准。

12 项提取：环境变量、运行时状态、测试、机械搬移、KL、crash、分布式 hang、回归二分、profile 生成/分析、torch profiler、CI。其余专项 skill 保留索引，未来对应任务可回到固定来源读取，不扩张当前交付。


## 3. V2 架构与文档修订（2026-10-21）

V2 再次核对参考 adapter `dac7de12eae473d9ba4dbd3658e67a517639a024` 的治理 §0/§1 与架构软件分层，以及外部 plugin `ce870e225df6b6d3db9e019cdbc620517f53648a` 的接入说明。两参考仓均保持只读。

| 修订 | 原因 | 新落点 |
|---|---|---|
| 组件关系改为独立仓共同接入 | V1 的二分流程容易把代码归属误读为部署互斥 | architecture / governance / README / skill |
| 明确 fallback 到当前有效基线 | 原函数可能已由 adapter 或前序 Hook 包装 | plugins / validation / dev 与 migration 参考 |
| 区分当前、参考与未来 | 防止把外部 dcu 入口或未来包名当当前实现 | architecture / bootstrap / ci |
| 增加文档中心、用途和维护触发点 | 让设计、操作、规则与证据可查可维护 | docs/README / reading-guide |
| 增加配置与协作专篇 | 统一跨仓依赖、版本组合、默认值和评审流程 | configuration / contributing |
| 提供图文导览 | 便于浏览仓库关系与未来结构 | 根 architecture.html |

## 4. JSON 快照使用边界

| 文件 | 固定内容 | V2 之后的读法 |
|---|---|---|
| `source-inventory.json` | V1 对 25 项上游 skill 的来源与取舍 | 目的地可能继续发展，原始源哈希仍指当时快照 |
| `source-manifest.json` | V1 的输入 SHA、源文件及当时目标哈希 | 不是当前全仓完整性清单；修改过的目标以 Git diff/当前版本验证追踪 |

V2 保留这两个来源快照，不重写原捕获日期或把新文案伪装为旧来源。新目录和包名是设计提案，运行时业务代码仍未引入。后续更新来源时新增版本记录并说明改变，不仅覆盖历史通过数字。

## 5. vLLM 发版方法迁入（2026-10-22）

从用户指定的同级 `upgrading-vllm-hcu-upstream-first` 抽取通用升级、上游接管、选测、ABI/产物和发版验收方法，重写为第 5 个 [SGLang 升级发版 skill](../skills/upgrading-sglang-hcu-upstream-first/SKILL.md)。现有四个入口按任务引用它，不搬移 vLLM runtime、专属 API/脚本、固定模型命令和历史通过数字。完整取舍见 [迁移分析](vllm-skill-migration.md)，本次源字节哈希独立保存在 [source-map](../skills/upgrading-sglang-hcu-upstream-first/references/source-map.json)，不覆盖 V1 JSON。


## V3 实践经验与方法来源

[开发经验与故障案例索引](development-methodology.md) 与重整后的开发/排障参考，来自本仓 [V3 运行记录](v3-runtime-validation.md) §7–§18：实际 caller/readout、collective owner、原生/理想 oracle、builder 环境、严格生成和产物闭环。新离线生成校验工具是本次实现，不来自 vLLM 运行时；案例的设备、版本与参数不成为通用默认值。V1 source-inventory/source-manifest 和升级 skill 的 source-map 保留各自原始快照语义，未为了本次方法改写而重写历史哈希。

## 来源与许可声明

hcu-sglang-plugin 方法与工具抽取，2026-10-20。

迁移审计脚本、迁移证据契约来自 sglang-plugin-das；回溯生成器与原始测试来自 agent_work/plugin_work。
文档组织与部分方法参考 hcu-sglang-adapter，插件测试/排障方法参考 sglang-das/.claude/skills。
来源基线、原始文件哈希及改写说明见 [source-manifest.json](source-manifest.json)、[source-inventory.json](source-inventory.json) 和本文。
保留源文件已有声明；三个参考仓库的许可原文分别位于 [licenses/](../licenses)。
本次未复制插件运行时、vendor 二进制或模型权重。

2026-10-22：另参考用户提供的 upgrading-vllm-hcu-upstream-first，按 SGLang 当前框架与方法库边界重写升级、选测、兼容产物与发版验收方法；未复制其运行时或厂商实现。
方法迁移说明及源文件哈希见 [vllm-skill-migration.md](vllm-skill-migration.md) 和 [source-map.json](../skills/upgrading-sglang-hcu-upstream-first/references/source-map.json)。原来源目录保持只读。

以上声明由原根目录 NOTICE 合并至本页，便于与来源版本和抽取范围一起维护。
