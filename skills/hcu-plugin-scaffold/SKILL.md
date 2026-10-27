---
name: hcu-plugin-scaffold
description: 在指定外部目录创建独立 SGLang 插件工程，确定包与入口命名、实现最小 Hook，并验证打包、发现、应用和启停行为。用于首次创建插件工程；已有插件的能力开发使用 hcu-plugin-dev。
---

# HCU 插件工程初始化

## 适用范围与输入

基于 [插件工程初始化指南](../../docs/first-plugin.md) 创建新的运行时插件工程。输入为官方 SGLang 路径与 ref/SHA、外部输出目录、distribution/Python 包/entry point 名称、首项功能或学习示例、Python 环境和证据目录。使用 adapter 时补充其版本与依赖关系。未指定命名时按目标能力提出并记录命名，常规命名选择不构成额外确认步骤。

本方法库只维护文档、Skill 和方法工具。插件包、pyproject.toml 与生成的测试必须位于用户指定的外部目录；学习示例使用独立临时目录。已存在实现仓的能力改动转 [hcu-plugin-dev](../hcu-plugin-dev/SKILL.md)，不重新创建包。

生成或修改源码遵守[生成代码约定](../../docs/contributing.md#生成代码约定)：不新增说明性注释、生成标记、来源 docstring 或 lint 抑制注释；保留来源已有内容与许可声明，说明写入独立文档和清单。

## 执行流程

1. 核对官方接口、输出路径与当前文件。输出不得落在方法库或官方输入树内；已有目录先保留内容，依据用户意图复用或选择新的子目录，不覆盖现有工程。
2. 依据初始化指南的最小示例确定三个名称：distribution、Python 包、`sglang.srt.plugins` entry point。指南的 demo 名称与固定目标只是学习案例，真实工程使用本次名称和 pinned API。
3. 在外部目录创建打包定义、轻量 `register` 入口及首个功能模块。存在正式 seam 时优先使用；否则选择窄 Hook，保留原参数、返回值、异常与有效基线，不在 import 时启动服务。
4. 采用 [安装指南](../../docs/installation.md) 的同解释器、隔离安装和来源核验方式。启动新进程验证 entry point 发现、预期目标注册/应用、实际调用与日志；泛化框架会吞部分错误时保留首条 traceback，不以退出 0 单独判成功。
5. 运行功能启用、功能关闭、插件未加载三个对照，断言业务结果及副作用。当前 pinned loader 的空 SGLANG_PLUGINS 表示加载全部插件，不能用于停用对照。HookRegistry 内部字段仅用于固定版本诊断，升级时重新核对。
6. 若任务还要求真实 HCU 功能，按 [hcu-plugin-dev](../hcu-plugin-dev/SKILL.md) 完成相应实现、支持/回退和数值/状态验证；需要模型或 GPU 时按用户要求和资源条件执行。初始化通过不代表模型精度、性能或 adapter 组合兼容。

## 输出与验收标准

交付外部工程路径、命名与入口映射、官方版本、安装/执行命令、模块来源、三个对照结果和未验证项，记录到 [验证模板](../../templates/validation-record.md)。对于仅初始化的任务，完成标准是新工程可打包、可发现、可实际调用且关闭后保留基线；不要求额外迁移、升级或生成 embedded 产物。

任意新插件都不能套用 DCU 专用的 `validate_plugin_runtime.py` 合同；该工具要求 sglang_dcu_plugin 与其具名 seam。新工程按自己的入口和目标建立断言。使用示例及输入填写方式见 [文档中的 Skill 调用指南](../../docs/first-plugin.md#使用-skill-创建新插件)。
