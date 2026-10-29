# 仓库工作约定

先读 docs/README.md 和 docs/reading-guide.md；目标尚未映射到专项入口时，使用 skills/hcu-plugin-assistant/SKILL.md 检索并补齐必要输入；已明确任务时直接选一个主要专项 skills/*/SKILL.md。安装看 docs/installation.md（hcu-plugin-install），工程初始化看 docs/first-plugin.md（hcu-plugin-scaffold），能力开发看 docs/plugins.md（hcu-plugin-dev），故障诊断看 docs/troubleshooting.md（hcu-plugin-triage）。各功能文档含本任务 Skill 调用示例。
本仓按 plugin-v3-1028 要求已包含 hcu_sglang 运行时与 pyproject.toml；方法、Skill、工具和代码分目录维护。旧 output-code/sglang-plugin 是迁入来源，不能继续解释为本仓禁止放运行时代码。新建其他独立插件或生成 embedded 产物仍使用指定的新目录。每次运行时代码修改更新 docs/hcu-sglang-development.md，记录步骤、提交范围、实际验证和未验证项；原始迁入清单不伪装成未来代码完整性清单。
adapter 与 plugin 独立开发、运行时可组合；实现归属按 docs/governance.md 拆分，不能把两仓部署解释成二选一。跨仓目标、加载顺序、配置及有效基线回退必须单独验证。
文档按 docs/README.md 的术语与编写规范统一标题、用途、范围、执行/检查和当前/规划状态，新增入口同步导航与 Skill；功能文档需说明对应 Skill 的输入、调用及交付边界。
共享文件使用相对路径或显式路径变量；本机位置放 ignored .env，配置方式见 docs/configuration.md。历史来源位置使用脱敏占位符，保留原始版本、文件哈希与验证边界；不改写 ignored 原始日志。
修改工具后运行 python3 ci/l0_gate.py 与 python3 -m unittest discover -s tests -v（Python 3.10+）。文档/skill 修改执行结构、链接/锚点和元信息检查；共享内容修改后用 python3 tools/build_guide.py 重建导览，再用 --check 检查快照；图文导览还需检查实际浏览器渲染、交互与窄屏布局；检查受阻时记录原因和未验证项，不以 DOM 检查代替截图验收。
审计可记录脏输入，回溯必须拒绝脏输入；不清理来源仓库来迎合工具。历史运行证据不能记作本次验证，V1 来源 JSON 保留快照语义。
协作、提交与证据要求见 docs/contributing.md；当前门禁和后续规划见 docs/ci.md。

生成、抽取或修改外部 plugin/embedded 代码时，不新增说明性注释、生成标记、来源说明 docstring 或 lint 抑制注释。保留来源已有注释、docstring、版权和许可声明；来源、差异与验证说明写入独立文档或 manifest。具体规则见 docs/contributing.md 的“生成代码约定”。
