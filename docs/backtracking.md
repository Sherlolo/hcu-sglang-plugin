# 源码内嵌回溯

本工具从 `plugin_work/reconstruct_sglang_das.py` 抽取：将固定的官方 Git revision 与干净的外部 `sglang-plugin-das` 组合成新 Git 仓库，复制 Git 跟踪的 `sglang_dcu_plugin/`，在 bootstrap 提前安装 AITER 兼容，在 loader 注册内嵌 dcu，再补齐 JIT package data、依赖/第三方许可和 SHA256 清单。它保留源码 Hook，**不重建历史 fork 的逐文件/逐 commit 内容**。

## 输入与保护

- Python 3.10+、Git；两个输入都是干净 worktree（含未跟踪文件检查）。工具不会修改它们。
- official 要有 `python/sglang/srt/plugins/__init__.py`、`python/sglang/__init__.py`、`python/pyproject.toml`，且注入 anchor 必须匹配。
- plugin 要有受 Git 跟踪的 `sglang_dcu_plugin/__init__.py`、`compat/aiter.py`。本方法库没有这些运行时代码，不能作为 `--plugin`。
- 输出须与两输入路径分离，不能是任一输入的子目录或父目录；先在同级临时树生成、语法检查，成功后重命名。
- 默认拒绝现有输出。`--force` 只接受带合法 kind 的旧生成清单，但它仍会删除那个输出中的后续修改；确认无须保留后再用。首次生成建议用新目录。
- anchor 漂移时停止并针对新 official SHA 修复/测试生成器；不猜测注入位置。

```bash
python3 tools/reconstruct_sglang_das.py \
  --upstream /path/to/official-sglang \
  --plugin /path/to/sglang-plugin-das \
  --output /path/to/new-sglang-das-embedded \
  --revision <official-sha> --branch embedded-dcu
python3 tools/verify_reconstruction.py \
  --output /path/to/new-sglang-das-embedded
```

生成清单 `.sglang-dcu-embedded.json` 记录输入路径/commit、模式、分支、生成器版本与关键文件 SHA256。核验命令发现缺失、变化或越界路径时返回非零；它证明列出的文件完整性，不能证明未列出文件、来源签名、模型正确性或性能。

生成树保留官方历史；新增内嵌文件和注入内容在输出 Git 工作树里待审阅，没有自动 commit/push。查看 `git status` / `git diff --check`，检查实际 import 路径都指向输出 `python`，禁用外部 entry point 后仍能加载，再验证模型 prefill/decode/graph/精度。内嵌 dcu 不受外部 `SGLANG_PLUGINS` allowlist 控制，不应混入同名外部 dcu 插件；生成 loader 的 `setdefault` 不会覆盖已发现的同名入口。

## 历史证据与当前边界

2026-09-07 的原始总结记录 official `c3947ee...`、plugin `ce870e...`，生成 173 个 plugin 跟踪文件、179 个关键文件哈希，5 项生成器单测、214 项 plugin 单测和 MiniMax TP4 服务请求通过。这是历史记录，当前仅复用工具与方法；本次离线验证另见 [delivery-validation](delivery-validation.md)，未继承这些 GPU 结论。
