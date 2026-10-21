# 来源与抽取地图

日期：2026-10-20。所有来源为 NMZ_zz_26 本地快照，未拉取网络版本。各树读取时均干净。机器可读基线与文件来源见 [source-manifest](source-manifest.json)，DAS 的 25 个 skill 全量索引、原始描述、SHA256、取舍与目的地见 [source-inventory](source-inventory.json)。

| 来源 | 抽取对象 | 新仓落点 |
|---|---|---|
| hcu-sglang-adapter | README/docs/skills/references/tools/tests/ci 组织；双仓判别与分层证据 | README、architecture、governance、reading-guide、ci |
| sglang-plugin-das | entry point → register → apply 生命周期；Hook 与 fallback；迁移 skill 的五个脚本 | plugins、migration、sglang-dcu-plugin-migration、tools |
| agent_work/plugin_work | reconstruct 与 5 个原测试、三方 repo_audit、历史回溯总结 | tools/reconstruct、tests、backtracking |
| sglang-das/.claude/skills | 12 项方法按 plugin 场景改写；13 项仅索引说明边界 | dev / triage references、ci、source-inventory |

`plugin_work/skills/sglang-dcu-plugin-migration` 与实现仓 skill 逐文件比较一致；旧 `repo_audit.py` 与 skill 的 `audit_repositories.py` 功能接近但并非相同：前者提取 register/plugin_hook 的字面目标，后者索引所有包含 sglang 的字符串。新工具合并保留 `literal_hook_targets` 与 `sglang_target_strings` 两类信息，避免重复维护入口。历史 history/manifest/巨大审计结果仅用于理解方法；新仓提供空 overrides 与证据模板，不把历史“verified”当作新能力验证。源码内嵌的结果仓库、插件业务代码、容器启动脚本与模型结果文件不搬入。

## 取舍与修正

- 参考 plugin 的旧中文简析描述 DeepSeek 未接线，当前顶层源码已经接线；新文档按 pinned SHA 与调用链描述。
- 旧迁移 skill 的范围同时覆盖平台适配与新增能力，提取后增加 adapter/plugin 选位步骤，不继承其混合边界。
- skill 入口重写为四个任务入口，共享脚本只存 tools；详细规则放按需 references。
- ledger 明确 AST 候选与 runtime-export 的计数来源；仅模块匹配不再自动声称 active/implemented/属性已解析，保持 pending。
- manifest 要求 official-ref 与 official 工作树 HEAD 一致，避免比较混合基线；脏删除不再回退到旧 HEAD 内容。
- reconstruct 新增输出与来源父/子目录重叠拒绝，其余沿用原生成机制；新增独立 SHA256 verifier。
- 未搬移整个上游 skill 引擎或其自动运维命令，保留适用于插件的决策方法；具体可执行脚本范围以 README 工具表为准。

12 项提取：环境变量、运行时状态、测试、机械搬移、KL、crash、分布式 hang、回归二分、profile 生成/分析、torch profiler、CI。其余专项 skill 保留索引，未来对应任务可回到固定来源读取，不扩张当前交付。
