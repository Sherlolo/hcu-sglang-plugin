# hcu-sglang-plugin

面向海光 HCU 的 SGLang **新增能力开发方法库**。当前交付文档、可复用 skill、迁移审计及源码内嵌回溯工具；不包含 plugin 运行时代码，也不注册 SGLang entry point。

组织方式参照 `hcu-sglang-adapter`：README 导航 → docs 设计/治理/操作 → skills 任务规约 → tools 工具 → tests / ci 分层验证。adapter 补齐上游在参照平台已有的行为；plugin 通过 seam/注册机制接入上游没有的新能力，二者都可以做性能优化。历史 `sglang-plugin-das` 同时包含这两类工作，需要先分类，再迁移。

先读 [阅读指南](docs/reading-guide.md)、[架构](docs/architecture.md) 和 [双仓治理](docs/governance.md)。来源、旧文档偏差和上游 skill 取舍见 [抽取说明](docs/extraction-map.md)。

## 快速使用

需要 Python 3.10+、Git。工具的帮助、审计、台账和回溯单测仅用标准库；运行时 Hook 签名校验另需外部 SGLang/plugin 环境。本库无需 `pip install -e .`。

```bash
python3 ci/l0_gate.py
python3 -m unittest discover -s tests -v
python3 tools/reconstruct_sglang_das.py --help
```

在 Codex 中显式提供仓库内 skill 的完整路径即可使用。也可按 [安装说明](docs/bootstrap.md) 把完整方法库放到个人目录下；skill 的相对资源路径依赖本库结构，勿只复制一个 skill 子目录。

| 任务 | Skill | 操作手册 |
|---|---|---|
| 开发新能力、选择 Hook 与回退 | [hcu-plugin-dev](skills/hcu-plugin-dev/SKILL.md) | [开发流程](docs/plugins.md) |
| 三方审计、迁移、升级版本 | [sglang-dcu-plugin-migration](skills/sglang-dcu-plugin-migration/SKILL.md) | [迁移手册](docs/migration.md) |
| official + 外部 plugin → 内嵌源码仓库 | [hcu-plugin-backtrack](skills/hcu-plugin-backtrack/SKILL.md) | [回溯手册](docs/backtracking.md) |
| 精度、hang、graph、性能回归 | [hcu-plugin-triage](skills/hcu-plugin-triage/SKILL.md) | [排障与验证](docs/validation.md) |

## 工具

| 文件 | 功能 |
|---|---|
| [audit_repositories.py](tools/audit_repositories.py) | Git 跟踪/未忽略文件哈希、AST、字面 Hook 候选及双仓差异 |
| [build_migration_manifest.py](tools/build_migration_manifest.py) | merge-base 三方清单，含 WORKTREE、双方改动及直接硬件漂移 |
| [build_disposition_ledger.py](tools/build_disposition_ledger.py) | 从清单、运行时 targets、人工 overrides 生成处置台账 |
| [sync_deepseek_methods.py](tools/sync_deepseek_methods.py) | 历史 DCU DeepSeek 布局的方法抽取/`--check`，限定外部实现仓 |
| [validate_hook_signatures.py](tools/validate_hook_signatures.py) | 外部环境中已注册模型 REPLACE Hook 的静态签名检查 |
| [reconstruct_sglang_das.py](tools/reconstruct_sglang_das.py) | 从干净输入生成独立 Git 仓库、内嵌 Hook 与 SHA256 清单 |
| [verify_reconstruction.py](tools/verify_reconstruction.py) | 复核回溯清单中的文件哈希，损坏/缺失返回非零 |

工具的范围和限制见对应手册。`artifacts/` 存本地生成物并已忽略；本次验证结果见 [交付验证](docs/delivery-validation.md)。不复制历史巨型审计 JSON、模型数据、生成源码仓库、缓存或业务实现。
