# 仓库工作约定

先读 docs/README.md、docs/reading-guide.md 和当前任务对应 skills/*/SKILL.md。
本仓当前是方法与工具库，不含 plugin 运行时实现；业务目录和 entry point 在 docs/architecture.md 标为规划。
adapter 与 plugin 独立开发、运行时可组合；实现归属按 docs/governance.md 拆分，不能把两仓部署解释成二选一。跨仓目标、加载顺序、配置及有效基线回退必须单独验证。
文档按 docs/README.md 的约定写清用途、范围、执行/检查和当前/规划状态，新增入口同步导航与 skill。
修改工具后运行 python3 ci/l0_gate.py 与 python3 -m unittest discover -s tests -v（Python 3.10+）。文档/skill 修改执行结构、链接/锚点和元信息检查；图文导览还需检查渲染。
审计可记录脏输入，回溯必须拒绝脏输入；不清理来源仓库来迎合工具。历史运行证据不能记作本次验证，V1 来源 JSON 保留快照语义。
协作、提交与证据要求见 docs/contributing.md；当前门禁和后续规划见 docs/ci.md。
