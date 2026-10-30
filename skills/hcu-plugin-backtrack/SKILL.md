---
name: hcu-plugin-backtrack
description: 将固定的官方 SGLang 与 HCU plugin 组合为源码内嵌 Hook Git 仓库并核验 SHA256。用于复现或交付 embedded 源码；不用于逐 commit 复刻历史 fork，也不自动内嵌 adapter。
---

# HCU 插件内嵌源码生成与核验


## 适用场景与任务衔接

[阅读指南](../../docs/reading-guide.md) 给出本 Skill 的输入、产出和可复制请求。各项 Skill 按任务选用，不要求顺序执行；安装现有插件使用 [安装 Skill](../hcu-plugin-install/SKILL.md) 和 [安装指南](../../docs/installation.md)。

## 用途与输入

用于从两个干净 Git 来源生成新的内嵌源码树。输入 official/plugin 路径与 SHA、revision、新输出路径；`--plugin` 应含当前 `hcu_sglang` 业务包（也支持旧 `sglang_dcu_plugin`）；本仓可作为当前插件输入。先读 [回溯手册](../../docs/backtracking.md)。

生成或修改源码遵守[生成代码约定](../../docs/contributing.md#生成代码约定)：不新增说明性注释、生成标记、来源 docstring 或 lint 抑制注释；保留来源已有内容与许可声明，说明写入独立文档和清单。

## 执行流程

1. 两输入须干净（含未跟踪文件），输出与任一输入不能相同或父子重叠。固定输入、revision 和输出分支。
2. 执行 [reconstruct_sglang_das.py](../../tools/reconstruct_sglang_das.py)。anchor 漂移时报告并停止，不猜注入位置；默认拒绝现有输出，`--force` 会丢弃已有生成树的后续编辑，替换前核验旧 manifest 哈希及受跟踪/暂存/未跟踪变更都在生成清单内；发现额外编辑先保留或改用新输出。
3. 用 [verify_reconstruction.py](../../tools/verify_reconstruction.py) 核验 SHA256，修改/缺失/越界应失败，不能改清单掩盖差异。
4. 审阅新树 diff、bootstrap/loader/package-data、依赖/许可与 import 来源。内嵌 `hcu`（旧包为 `dcu`）不依赖外部 allowlist，隔离同名外部 entry point。
5. 若部署使用 adapter，另记录其版本与组合契约；生成器不会自动嵌入 adapter，也不证明组合兼容。按 [验证矩阵](../../docs/validation.md) 另测适用模型/graph/精度。

## 输出与完成条件

输出新 Git 工作树、来源清单、核验结果和运行验证状态。生成/哈希通过与服务正确性分别记录，历史 MiniMax 结果不是新输出的证据。源码、隔离 wheel/image 与 embedded 的真实模块来源、行为验证分开，具体闭环见 [历史经验](../../docs/development-methodology.md#8-交付产物与模块来源验证)。未来包名变化需要更新生成器支持范围和测试，不假定历史工具自动适配。

## 实践补充

V3 增补：完整 native package-data 必须覆盖 .cu/.cpp/.hpp；哈希通过后屏蔽所有外部 entry points 验证内嵌加载，强制官方与 plugin 模块都来自输出 python。执行外部测试时使用 --import-mode=importlib 防止 pytest 路径重新引入外部实现。步骤见 [V3 实践](../../docs/v3-runtime-validation.md)。

## 作为发版产物时

embedded 若属于本次发布范围，按 [产物来源](../upgrading-sglang-hcu-upstream-first/references/compatibility-and-artifacts.md) 与 [发版验收](../upgrading-sglang-hcu-upstream-first/references/release-validation.md) 独立留行。源码哈希、隔离加载、服务正确性分别给证据；只有本次必测闭合后才更新该形态的 last-good。

## HCU 逻辑对照（需求包含该输入时）

额外固定 HCU fork SHA，按 [逻辑对照流程](../../docs/backtracking.md#8-hcu-来源逻辑对照) 分别检查生成保真、当前来源和行为。生成器仅使用 official/plugin；HCU 是独立比较来源。记录 bootstrap 等预期集成映射；未迁移领域和类拆分显式列出，不能用 manifest/AST/全量 apply 宣称整个 fork 等价。


来源校验保持只读，采用工具当前的 `GIT_OPTIONAL_LOCKS=0` 子进程设置；任务要求索引字节保留时，按 [来源索引保护](../../docs/backtracking.md#9-来源索引保护) 记录前后哈希。索引变化先区分暂存内容与缓存，不重置来源来消除差异。

生成器 v4 自动识别唯一运行时包；双包输入用 `--plugin-package` 显式选择。当前 HCU 内嵌入口为 `hcu`，JIT 和 DeepSeek V4 csrc 随源码保留。核对 manifest 中实际 package、文件哈希与隔离加载来源，不继承旧包运行结果。
