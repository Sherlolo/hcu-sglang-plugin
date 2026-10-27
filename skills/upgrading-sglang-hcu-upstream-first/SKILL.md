---
name: upgrading-sglang-hcu-upstream-first
description: 将 SGLang HCU 外部插件升级到指定官方 tag、分支或 commit，并准备 wheel、镜像或 embedded 发版验收。用于上游接管旧补丁、接口漂移、测试选取和发布证据闭合；仅做三方差异盘点时使用 sglang-dcu-plugin-migration。
---

# SGLang HCU 上游优先升级与发版

操作步骤与调用示例见 [版本升级与发版指南](../../docs/upgrading.md)。

本仓是方法与工具库；实现、构建和运行测试在指定的外部 plugin/adapter/输出仓执行。
目标由本次任务决定，不预设 v0.5.22、某个 gfx、镜像、容器或模型。

## 适用场景与任务衔接

[阅读指南](../../docs/reading-guide.md) 给出本 Skill 的输入、产出和可复制请求。各项 Skill 按任务选用，不要求顺序执行；安装现有插件使用 [安装 Skill](../hcu-plugin-install/SKILL.md) 和 [安装指南](../../docs/installation.md)。

## 执行主线

1. 固定实际已验证基线、plugin 起点、fork 官方基点与用户目标 ref/SHA，记录 dirty、环境及功能范围。目标缺失时先整理基线，再澄清；不把本地 HEAD 或历史案例作为默认目标。复制 [升级记录模板](../../templates/upgrade-release-record.md) 建立两张表。
2. 读取 [升级与接管](references/upgrade-workflow.md)，用 [三方迁移 Skill](../sglang-dcu-plugin-migration/SKILL.md) 盘点全部维护面；枚举 Hook 之外的具名 seam、registry、alias、native 与构建/配置差异。先判定官方能否接管，再决定保留、窄化或删除；自动台账仍是 pending。
3. 按 [契约与影响分析](references/contracts-and-impact.md) 审查新 owner、签名及完整数据流，建立改动到测试的映射，按 [历史经验](../../docs/development-methodology.md) 保留真实公共分发、readout、FFN 出口及实际 TP>1 的反例，不以旧单卡 mock 的成功继承语义正确性。优先固定目标已有的 seam/注册入口；不能把 vLLM Platform、CustomOp 或 MRV2 机制假定为 SGLang 接口。
4. 按 [兼容与产物](references/compatibility-and-artifacts.md) 选择可复用产物或最小必要构建，隔离验证父子进程来源。源码固定 SHA 用于复现；运行时兼容条件由实际能力/ABI 和项目策略决定，不默认绑定一个完整 wheel 字符串。
5. 执行快速反馈后，完成本次必测矩阵。涉及 native、KV、FLA、Graph、collective 时读 [原生与状态契约](references/native-contracts.md)；服务、准确率、性能与交付按 [发版验收](references/release-validation.md)。保留用户指定 backend、Graph、精度、prefix、投机及拓扑；诊断降级单独记录。
6. 核对旧补丁全部有处置、必测行全部 passed、交付内容与证据一致，再更新对应范围的 last-good。缺设备/依赖/标准、未执行或仅 fallback 通过均不能记完整通过。提交、推送、镜像发布按本次授权执行；编写 Skill 不会自动启用发布或定时跟进。

## 按需读取

| 当前任务 | 参考 |
|---|---|
| 上游接管、补丁去重、窄 seam、退出条件 | [upgrade-workflow](references/upgrade-workflow.md) |
| API 变化、选测、失效与候选/last-good | [contracts-and-impact](references/contracts-and-impact.md) |
| release-line、ABI、wheel/image/native 资源 | [compatibility-and-artifacts](references/compatibility-and-artifacts.md) |
| KV/FLA/MoE、状态、Graph、分布式 | [native-contracts](references/native-contracts.md) |
| 服务生命周期、模型协议、准确率、性能、CI | [release-validation](references/release-validation.md) |
| 来源、迁移取舍与未采用的 vLLM 专属规则 | [迁移分析](../../docs/vllm-skill-migration.md)、[来源快照](references/source-map.json) |

## 交付

交付源码差异、补丁处置台账、验收矩阵、产物摘要与复现命令；说明剩余复制实现、临时 patch 退出条件、失败及未测项。
adapter/plugin 的实现归属和部署组合分别记录，fallback 保留当前有效适配基线。embedded 交付另走 [回溯 Skill](../hcu-plugin-backtrack/SKILL.md)；生成和哈希正确不能代替服务验收。

生成或修改源码遵守[生成代码约定](../../docs/contributing.md#生成代码约定)：不新增说明性注释、生成标记、来源 docstring 或 lint 抑制注释；保留来源已有内容与许可声明，说明写入独立文档和清单。
