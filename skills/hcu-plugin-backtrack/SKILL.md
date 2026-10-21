---
name: hcu-plugin-backtrack
description: 将固定的官方 SGLang 和外部 sglang-plugin-das 组合成独立的源码内嵌 Hook Git 仓库，并核验生成清单。用于回溯、复现或交付内嵌源码；不用于逐 commit 复刻历史 DAS fork。
---

# HCU Plugin 源码回溯


先读 [回溯手册](../../docs/backtracking.md)，确认 `--plugin` 指向含 `sglang_dcu_plugin` 的外部实现仓，不是本方法库。

1. 记录 official/plugin SHA；两输入须干净（含未跟踪文件）。指定 revision 与新的输出目录，三者路径相互分离。
2. 使用 [reconstruct_sglang_das.py](../../tools/reconstruct_sglang_das.py)；anchor 变更时停止并报告漂移，不猜测源码插入位置。默认不覆盖；`--force` 会丢弃已有生成树的后续编辑，使用前确认该输出可替换。
3. 执行 [verify_reconstruction.py](../../tools/verify_reconstruction.py) 验证 provenance 文件中的 SHA256；损坏/缺失要失败，不能重写清单掩盖差异。
4. 检查新树 Git diff、loader/bootstrap/package-data、依赖/许可与 import 路径。内嵌 dcu 独立于外部 allowlist，隔离同名外部 entry point。
5. 按 [验证矩阵](../../docs/validation.md) 将生成/哈希成功与实际模型正确性分别记录。历史 MiniMax 测试不是新输出的证据。
