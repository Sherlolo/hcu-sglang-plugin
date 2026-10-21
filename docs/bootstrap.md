# 安装与环境

## 工具环境

使用 Python 3.10+ 与 Git。NMZ_zz_26 的宿主机默认 `python3` 为 3.6，不能直接运行本工具集；请选支持的解释器，或把仓库挂载到已有 Python 3.10+ 容器内。命令中的 `python3` 均指已确认版本的解释器。

```bash
python3 --version
git --version
python3 ci/l0_gate.py
python3 -m unittest discover -s tests -v
```

本库没有运行时包、依赖安装脚本或 GPU 服务启动副作用。`validate_hook_signatures.py` 会导入外部 SGLang/plugin 并调用 register，须在其依赖齐全的环境中运行；它的 `--help` 不导入这些运行时依赖。

## Codex skill 使用

最直接方式是在请求中指定 `skills/<name>/SKILL.md` 完整路径。例如：

```text
读取 <repo>/skills/hcu-plugin-backtrack/SKILL.md，用给定的 official 和 plugin 仓库生成一个新输出目录并核验清单。
```

需要放到个人目录时，可复制**整库**到自选目录，仍用完整路径引用 SKILL.md。保留相对目录结构，不把四个 skill 子目录单独复制而丢失共享 docs/tools。此方法库以仓库内 skill 为交付形式，不承诺仅移动目录就触发特定客户端的自动发现；本次未修改全局 skill 配置。

## 未来实现仓的启用

旧 DCU plugin 的 `pip install -e <external-plugin> --no-deps` 与 `SGLANG_PLUGINS=dcu` 属于外部实现仓用法，不能对本库执行或据此宣称已装好插件。official、adapter、外部 plugin 三者共用时固定各 SHA，并核对实际导入路径、相同目标的 Hook 顺序与环境配置归属。
