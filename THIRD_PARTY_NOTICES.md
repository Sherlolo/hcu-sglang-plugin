# 第三方与外部来源清单（THIRD_PARTY_NOTICES）

> 本文件登记仓库内非本仓原创的源码来源（HYGON 开源合规规则
> `hygon-open-source-v1.7` §4 要求）。文档与方法库的抽取来源见
> [docs/extraction-map.md](docs/extraction-map.md)；内部仓库许可证副本见
> [licenses/](licenses/)。新增移植/借源代码时同步登记本表。

## 1. 外部开源项目

### sglang（Apache-2.0）

- 项目：SGLang，Copyright 2023-2024 SGLang Team（及文件头标注的 vLLM 间接贡献者）
- 仓库：<https://github.com/sgl-project/sglang>（Apache-2.0）
- 引入方式：运行时插件以 sglang 官方树为宿主；`hcu_sglang/models/deepseek/`
  等 methods 文件按 pinned SHA 从 sglang / 内部 sglang-das fork 迁入
  （文件头保留原版权声明；逐文件映射见 docs/extraction-map.md 与
  `tools/build_migration_manifest.py` 生成的迁移清单）
- HYGON 修改：hook 接线与 HCU 适配，见各文件与 docs/migration.md

### vLLM（Apache-2.0）

- 项目：vLLM，Copyright contributors to the vLLM project
- 仓库：<https://github.com/vllm-project/vllm>（Apache-2.0）
- 本地路径：`hcu_sglang/distributed_aiter.py`——锚定 vLLM tag
  `v0.6.4.post1` 的 `custom_all_reduce.py`（文件头 Adapted from 链接）；
  deepseek methods 文件头中标注的 vLLM 声明同源保留
- HYGON 修改：aiter 适配与 HCU 路径改写

### fast-hadamard-transform（BSD-3-Clause）

- 项目：fast-hadamard-transform，Copyright (c) 2023, Tri Dao
- 仓库：<https://github.com/sgl-project/fast-hadamard-transform>（fork 自
  Dao-AILab，BSD-3-Clause）
- 本地路径：`hcu_sglang/jit_overlay/csrc/fast-hadamard-transform/`
  （fast_hadamard_transform.h / _common.h / hadamard_jit.cuh）
- HYGON 修改：JIT 接线，核体保留原声明

## 2. 内部 vendor（DeepInfer / deep_select，HYGON 内部项目）

- 本地路径：`hcu_sglang/jit_overlay/csrc/deep_select/vendor/`（kerutils
  工具头 + cuda_kernels）
- 来源：DeepInfer deep_select 组件的 vendor 快照，内部项目，无外部
  许可证义务；其中 kerutils 使用 CUTLASS/CuTe 原语
  （NVIDIA CUTLASS，BSD-3-Clause）
- 精确 commit 锚定与上游核对：deep_select 仓库固定版本待登记（复扫前
  补全 vendor 快照的引入 commit）

## 3. HYGON 内部仓库血缘（非外部第三方）

- sglang-das / hcu-sglang-adapter / sglang-plugin-das：Apache-2.0，
  许可证副本见 [licenses/](licenses/)；methods 迁移与改写边界见
  [docs/extraction-map.md](docs/extraction-map.md) 与
  [docs/sglang-das-migration-plan.md](docs/sglang-das-migration-plan.md)。

## 4. 登记规则速记

- 外部开源：保留原声明 + 本表登记，不加 HYGON 头。
- HYGON 原创源码：文件头两行（Copyright + SPDX-Apache-2.0）；存量补齐
  用 `tools/compliance/add_hygon_headers.py --from-list`（按合规报告 §3
  清单驱动；迁移/vendor 文件不机械加水头）。
