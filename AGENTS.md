# 仓库工作约定

先读 docs/reading-guide.md 和当前任务对应 skills/*/SKILL.md。
本仓当前是方法与工具库，不含 plugin 运行时实现；用户另行要求实现时先按 docs/governance.md 分类。
修改工具后运行 python3 ci/l0_gate.py 与 python3 -m unittest discover -s tests -v（Python 3.10+）。
审计可记录脏输入，回溯必须拒绝脏输入；不清理来源仓库来迎合工具。历史运行证据不能记作本次验证。
