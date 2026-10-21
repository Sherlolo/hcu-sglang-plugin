# 方法库门禁

Python 3.10+，仓库根目录执行：

```bash
python3 ci/l0_gate.py
python3 -m unittest discover -s tests -v
```

L0 检查所有 Python 的语法、skill 名称/描述/UI 元信息、相对文件链接及业务代码排除；它不导入 SGLang 或触发 GPU。`.workflow/l0-gate.yml` 是通用 runner 命令配置，沿用 adapter 的目录组织，尚未绑定任何远端 CI 平台，不宣称自动触发。当前库不需要构建 wheel。

工具测试仅在 TemporaryDirectory 创建 Git 仓库验证行为，不动真实来源仓库。新增或修改脚本要运行相应行为验证；签名脚本和方法同步脚本依赖外部实现仓，离线只验证 CLI 入口/语法，实际覆盖范围见迁移手册。

外部实现仓的 GPU/服务 gate 依据 [validation](validation.md) 维护；本库没有 L1 GPU gate，不能把 CPU 工具检查包装为推理验收。
