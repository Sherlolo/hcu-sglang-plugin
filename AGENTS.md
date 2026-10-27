# 仓库工作约定

先读 docs/README.md 和 docs/reading-guide.md，按任务选一个主要 skills/*/SKILL.md。安装看 docs/installation.md（hcu-plugin-install），工程初始化看 docs/first-plugin.md（hcu-plugin-scaffold），能力开发看 docs/plugins.md（hcu-plugin-dev），故障诊断看 docs/troubleshooting.md（hcu-plugin-triage）。各功能文档含本任务 Skill 调用示例。
本仓当前是方法与工具库，不放 plugin 运行时实现或打包定义。示例代码放文档代码块，实跑文件放外部临时目录；已有业务实现位于独立 output-code/sglang-plugin。docs/architecture.md 的本仓业务目录只是规划，未经用户要求不落地。
adapter 与 plugin 独立开发、运行时可组合；实现归属按 docs/governance.md 拆分，不能把两仓部署解释成二选一。跨仓目标、加载顺序、配置及有效基线回退必须单独验证。
文档按 docs/README.md 的术语与编写规范统一标题、用途、范围、执行/检查和当前/规划状态，新增入口同步导航与 Skill；功能文档需说明对应 Skill 的输入、调用及交付边界。
共享文件使用相对路径或显式路径变量；本机位置放 ignored .env，配置方式见 docs/configuration.md。历史来源位置使用脱敏占位符，保留原始版本、文件哈希与验证边界；不改写 ignored 原始日志。
修改工具后运行 python3 ci/l0_gate.py 与 python3 -m unittest discover -s tests -v（Python 3.10+）。文档/skill 修改执行结构、链接/锚点和元信息检查；图文导览还需检查渲染。
审计可记录脏输入，回溯必须拒绝脏输入；不清理来源仓库来迎合工具。历史运行证据不能记作本次验证，V1 来源 JSON 保留快照语义。
协作、提交与证据要求见 docs/contributing.md；当前门禁和后续规划见 docs/ci.md。

生成、抽取或修改外部 plugin/embedded 代码时，不新增说明性注释、生成标记、来源说明 docstring 或 lint 抑制注释。保留来源已有注释、docstring、版权和许可声明；来源、差异与验证说明写入独立文档或 manifest。具体规则见 docs/contributing.md 的“生成代码约定”。
