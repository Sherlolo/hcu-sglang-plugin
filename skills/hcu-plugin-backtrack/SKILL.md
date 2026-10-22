---
name: hcu-plugin-backtrack
description: 将固定的官方 SGLang 与外部 sglang-plugin-das 组合为源码内嵌 Hook Git 仓库并核验 SHA256。用于复现或交付 embedded 源码；不用于逐 commit 复刻历史 fork，也不自动内嵌 adapter。
---

# HCU Plugin 源码回溯与清单核验


## 用途与输入

用于从两个干净 Git 来源生成新的内嵌源码树。输入 official/plugin 路径与 SHA、revision、新输出路径；`--plugin` 必须含 `sglang_dcu_plugin` 业务包，不是当前方法库。先读 [回溯手册](../../docs/backtracking.md)。

## 执行流程

1. 两输入须干净（含未跟踪文件），输出与任一输入不能相同或父子重叠。固定输入、revision 和输出分支。
2. 执行 [reconstruct_sglang_das.py](../../tools/reconstruct_sglang_das.py)。anchor 漂移时报告并停止，不猜注入位置；默认拒绝现有输出，`--force` 会丢弃已有生成树的后续编辑，使用前确认该输出可替换。
3. 用 [verify_reconstruction.py](../../tools/verify_reconstruction.py) 核验 SHA256，修改/缺失/越界应失败，不能改清单掩盖差异。
4. 审阅新树 diff、bootstrap/loader/package-data、依赖/许可与 import 来源。内嵌 dcu 不依赖外部 allowlist，隔离同名外部 entry point。
5. 若部署使用 adapter，另记录其版本与组合契约；生成器不会自动嵌入 adapter，也不证明组合兼容。按 [验证矩阵](../../docs/validation.md) 另测适用模型/graph/精度。

## 输出与完成条件

输出新 Git 工作树、来源清单、核验结果和运行验证状态。生成/哈希通过与服务正确性分别记录，历史 MiniMax 结果不是新输出的证据。未来包名变化需要更新生成器支持范围和测试，不假定历史工具自动适配。
