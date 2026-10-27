# SGLang HCU 插件版本升级与发版指南

> **本文用途**：组织现有外部插件向指定官方 ref/SHA 升级，完成补丁处置、兼容分析、交付产物和验收矩阵。三方差异盘点见 [迁移指南](migration.md)，方法来源见 [vLLM Skill 迁移分析](vllm-skill-migration.md)。

## 适用范围与前置条件

输入包括当前有效的 official/plugin/可选 adapter 基线、fork 来源、指定官方目标、交付形态（源码、wheel、镜像或 embedded）、功能范围与验收标准。目标由本次任务指定，不默认沿用历史 V3 版本或模型。单纯安装使用 [安装指南](installation.md)，新能力开发使用 [开发指南](plugins.md)。

## 使用 Skill 执行版本升级

对应 Skill：[upgrading-sglang-hcu-upstream-first](../skills/upgrading-sglang-hcu-upstream-first/SKILL.md)。该 Skill 负责从上游接管分析到本次交付验收的组织，不自动执行外部发布。

```text
读取 <方法库绝对路径>/skills/upgrading-sglang-hcu-upstream-first/SKILL.md，按该 Skill 执行。
当前有效基线：<official/plugin/可选 adapter 路径及 SHA>。
HCU 来源：<fork 路径与基点>；官方升级目标：<明确 ref/SHA>。
功能范围：<本次需保持的能力和部署组合>。
交付形态：<源码/wheel/镜像/embedded>；验收标准：<模型、配置和指标>。
执行环境与证据目录：<实际路径和资源条件>。
建立补丁处置及验收矩阵，优先评估官方接管，完成必要实现与实际产物验证。
保留失败及未执行项；提交、推送或发布按本次明确授权执行。
```

预期交付：升级后的实现、补丁处置台账、兼容与测试影响分析、交付产物身份、验收矩阵、候选与 last-good 状态。

## 执行流程与参考资料

| 阶段 | 执行内容 | 参考资料 |
|---|---|---|
| 基线与差异分析 | 固定当前有效版本、目标与 dirty；建立维护面和待处置项 | [三方迁移](migration.md)、[升级与接管](../skills/upgrading-sglang-hcu-upstream-first/references/upgrade-workflow.md) |
| 接口与职责迁移 | 核对官方是否已接管能力；审查签名、状态、布局和 collective 所有权 | [契约与影响分析](../skills/upgrading-sglang-hcu-upstream-first/references/contracts-and-impact.md) |
| 最小实现调整 | 保留必要 HCU 差异，删除或收敛已被官方覆盖的补丁 | [能力开发](plugins.md)、[原生与状态契约](../skills/upgrading-sglang-hcu-upstream-first/references/native-contracts.md) |
| 构建与来源核验 | 核对 ABI、依赖、资源和实际模块路径，按交付形态留证 | [兼容与产物](../skills/upgrading-sglang-hcu-upstream-first/references/compatibility-and-artifacts.md) |
| 验收与交接 | 执行本次必测行，记录已验证范围、剩余问题和回退版本 | [发版验收](../skills/upgrading-sglang-hcu-upstream-first/references/release-validation.md)、[验证规范](validation.md) |

## 验收标准与执行边界

复制 [升级记录模板](../templates/upgrade-release-record.md)，将每项补丁处置与相关验证行关联。仅在本次必测项通过、实际交付来源一致且未完成事项明确后，更新相应范围的 last-good。历史结果、单次 health 或 fallback 通过不替代目标能力验证。

要求 embedded 交付时，另用 [源码回溯 Skill](../skills/hcu-plugin-backtrack/SKILL.md) 生成与核验；它不替代模型服务验证。源仓脏文件与用户编辑保持原状；构建和运行修改在外部实现仓或独立输出目录完成。升级工作流本身不授权推送、镜像发布、通知他人或定时任务。
