---
name: hcu-plugin-install
description: 在指定环境安装现有 SGLang HCU 插件源码或 wheel，核验版本、入口、模块来源和实际加载，并记录设备与服务验收边界。用于插件安装、启用和安装层诊断；运行时行为回归使用 hcu-plugin-triage。
---

# HCU 插件安装与加载验证

## 适用范围与输入

按 [安装与启用指南](../../docs/installation.md) 操作已有运行时产物。需要官方 SGLang 路径/版本、插件源码或 wheel 路径/哈希、目标 Python/容器、安装方式与可写目录、实际运行用户、证据目录；仅在要求服务验证时再提供模型与完整配方。

当前方法库不是可安装插件。安装位置优先使用指定个人环境或任务 target，不修改共享 site-packages、全局 Git 配置或 HOME；保留当前部署依赖的 adapter/platform 基线。目标为任意新插件时，使用其实际 distribution 和 entry point，不能假定都叫 dcu。

## 执行流程

1. 按 [路径配置](../../docs/configuration.md#1-当前工具的路径配置) 读取本次显式路径；容器使用其挂载内路径，不沿用历史机器、用户或 UID。固定实际解释器、版本、源码/wheel 来源和兼容条件。`--no-deps` 不准备 Torch/HCU/vendor 依赖；缺失项明确记录，不能靠升级全部依赖猜测修复。
2. 按安装指南选择 wheel 或 editable，记录安装目录。editable 构建使用源码所属用户；目标目录中的 `.pth` 不会仅因加入 PYTHONPATH 自动执行，按指南显式保留源码路径与入口 metadata。运行用户可与构建用户不同，但需使用同一环境并能读取产物。
3. 在新进程确认 distribution metadata、入口、official/plugin 实际路径；wheel 验证移除同名源码路径，避免源码遮蔽安装产物。可写缓存逐库配置，AITER_JIT_DIR 不由 XDG_CACHE_HOME 自动替代。
4. 在实际运行用户中检查 HCU 访问与可见设备。安装/metadata 可通过而设备初始化失败，分层保留状态；遵守节点权限，不 mock 硬件检测或改变设备权限来制造通过。
5. 对依赖就绪的运行时执行加载合同。当前 DCU 版本可使用指南中的严格验证工具；其他插件使用自己的目标/seam 合同。检查非空目标、全量应用、模块来源与加载软错误。首次编译与注册错误分开，保留尝试日志；观察中断先核验原进程，不并发重启同一个构建。
6. 仅在任务要求时，以指定模型配方验证本任务服务归属、health、模型身份、原始响应与预期结果，并清理自有进程。错误答案、Graph、TP、hang 或性能回归转 [hcu-plugin-triage](../hcu-plugin-triage/SKILL.md)。

## 输出与验收标准

记录解释器/用户、安装产物及来源、环境配置、入口、实际加载报告，分别给安装、metadata、设备、加载和服务状态。不将入口存在等同于加载成功，不将 health 200 等同于生成正确；未要求或无法执行的服务项不写成已验证。

停用在新进程中验证。本次 pinned loader 的空 SGLANG_PLUGINS 会加载所有插件，需使用明确白名单；仅当用户请求卸载时操作对应个人环境。示例见 [文档中的 Skill 调用指南](../../docs/installation.md#使用-skill-安装与验证插件)，安装与设备问题的症状路线见 [故障诊断指南](../../docs/troubleshooting.md)。
