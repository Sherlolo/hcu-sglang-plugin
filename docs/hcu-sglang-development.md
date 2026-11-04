# hcu_sglang 开发迁移记录

> **本文用途**：持续记录本仓 `hcu_sglang` 每次代码迁移或开发完成的步骤、提交范围、实际验证与未完成条件。2026-10-28 起，本仓从仅维护方法推进到同时维护运行时代码；早期方法库限制不再禁止当前包落地。

## 时间索引

开发与验证记录按日期从早到晚排列；同一天沿用原记录顺序，未提供具体时刻的条目不推定先后。跨日执行的历史记录按起始日期归档，保留原文中的实际执行日期与验证边界。

- [2026-10-26：V3 实施与验证](#2026-10-26-v3-2026-10-26-实施与验证记录)
- [2026-10-27：方法库导览与统一入口](#2026-10-27-v3-方法库导览与统一任务入口实施记录)
- [2026-10-28：运行时代码迁入](#2026-10-28-运行时代码迁入与入口验证)
- [2026-10-29：继续迁移与四卡验证](#6-2026-10-29-分阶段继续迁移)
- [2026-10-30：后端与模型路径同步](#20-2026-10-30-dsa-indexer-当前-hcu-producerconsumer-同步)
- [2026-11-01：分支迁移与八卡验证](#35-2026-11-01-rmsnormgemma-hcu-分支继续迁移)
- [2026-11-02：DAS 分支逻辑对齐](#das-分支逻辑对齐2026-11-02)
- [附录：布局、操作方式与汇总](#附录布局操作方式与汇总)

两份早期独立开发记录已全文收录。历史结果不代表当前代码已重新验证。

## 2026-10-26 V3 2026-10-26 实施与验证记录

> **用途**：记录 `plugin-v3-1026.txt` 指定输入下的插件开发、内嵌源码生成、HCU 逻辑对照及本轮验证。执行跨 2026-10-26 至 2026-10-27（Asia/Shanghai）。方法库保留 Skill、工具和记录，运行时代码位于独立 `output-code` 仓库。

> **当前交付**：以下 §1–7 保留各阶段快照；正式输出、最终验证和限制统一见 [§8](#2026-10-26-8-最终交付与需求验收)。

### 2026-10-26 1. 固定输入与产物

工作根目录以 `${BASE}` 表示；下表保留历史相对布局，不要求当前部署沿用。`${BASE}` 仅是历史位置标识，执行任务使用 [路径配置](configuration.md#1-当前工具的路径配置) 中的本地变量。

| 对象 | 路径及版本 | 状态与用途 |
|---|---|---|
| 社区输入 | `input-code/sglang`；`baa9befa381511cd4d56f83980951e5a45fffc6d` | 干净来源，提供当前框架和正式插件接口 |
| HCU 输入 | `input-code/sglang-das`；`15c7204b7a3b6edaa222cff80018d8e5f2b2ad0e` | 干净来源；较上轮增加 139 个文件改动 |
| 插件输出 | `output-code/sglang-plugin`；v0.3.1，`39038d5394c23e6b918e6e50911ee5292a1e4d34` | 独立包 `sglang_dcu_plugin`，入口 `dcu` |
| 回溯输出 | `output-code/sglang-das` | 由上述社区与插件两个干净输入生成，清单记录来源 SHA |
| 方法库 | `hcu-sglang-plugin` | 本文、Skill、抽取工具修正与验证记录 |
| 执行容器 | 历史验证容器 | Python 3.10.12；宿主 Python 3.6 不用于工具执行 |
| 证据目录 | `hcu-sglang-plugin/artifacts/v3-20261026` | 原始日志、资源准入、响应、哈希与逻辑对照，Git 忽略 |

开始执行时 `output-code` 不存在。插件起点从保留的源码快照恢复，运行时逐文件与历史 `45e67e0` wheel 一致，记录见 `baseline-recovery.json`；新仓库恢复提交为 `bb76e1b`，没有冒充恢复原仓库已移除的 Git 历史。历史验证结果没有计入本轮通过数量。

### 2026-10-26 2. 已实现模块与证据边界

| 模块 / 实现位置（插件仓库内） | 本轮建设内容 | 本轮证据与限制 |
|---|---|---|
| 注册与参数解析：`__init__.py`、`server_args*.py` | 保留正式入口，新增配置与通信目标，目标合同更新为 210 项 | 严格 loader 检查非空、预期集合、全量 apply、实际来源与三个 resolution seam；注册不代表所有分支均已执行 |
| V4.1 配置：`configs.py` | HCU Channel FP8 转为 compressed-tensors channel-weight / token-activation 配置，接在官方规范化入口之后 | 通过真实 `DeepseekV41Config` 构造入口；其他量化格式和非 HCU 保留基线。未运行完整 V4.1 模型 |
| 通信：`distributed.py` | HCU native all-reduce 在 Graph capture 中使用 copy-in，保留 disabled/shape 准入和非 capture 的有效基线 | CPU 分支断言及真实 TP4：每 rank 两个 Graph 分配，三轮按 rank 交叉选择重放；不外推 AITER Fabric、DP 模型或全部拓扑 |
| 方法抽取：`models/deepseek/**/methods` | 七个 V2/V4/NextN 构造函数显式绑定拥有类；修复八处二次常量限定 | 修复前八项执行失败，修复后通过；覆盖构造父类边界和 MHC norm 分支，不能替代整模型构造与生成 |
| 常量别名：`models/deepseek/hook_utils.py` | 覆盖前保存旧引用，仅刷新已存在的 DeepSeek 消费者别名；缺失属性不视为 `None` 别名 | 两项修复前失败 / 修复后通过；不向无关 vendor 模块注入属性 |
| JIT 原生资源：`jit_overlay` | 更新 `c1.cuh`、`c2.cuh`、`c_plan.cuh`、`fp4_indexer_rope.cuh`、`fp4_utils.cuh` | 283 个 native 文件及一项许可资源，共 284 项与当前 HCU / embedded 哈希一致。新增 FP4/压缩算子的完整数值验证未执行 |
| MiniMax 与层边界：`models/minimax_m2*`、`layers/layer_boundary.py` 等 | 保留 residual-stream 与 FFN 出口归约契约；没有重新引入旧生产者重复 all-reduce | 当前 CPU、GPU 与本轮 TP4 服务结果。该兼容实现与历史 fork 文本可不同，需按最终输出与归约所有权比较 |
| 内嵌集成 | 复制 442 个 plugin 文件，注入官方 bootstrap、loader、依赖、许可及 package-data | 448 项生成清单通过；SGLang 入口 bootstrap 替代外部 `usercustomize.py`，属于显式集成差异 |

### 2026-10-26 3. 故障复现与工具修正

#### 2026-10-26 3.1 常量重写与类闭包

旧正则先把 `_is_hcu` 改成 `C._is_hcu`，又在下一轮替换 `_is_hcu`，产生 `C.C._is_hcu`。类方法抽成普通函数后，原来的 `super()` 也失去隐式 `__class__` 闭包。仅检查 AST、注册和 apply 均无法发现这些执行错误。

当前 [方法同步工具](../tools/sync_deepseek_methods.py) 根据 AST Name 的 UTF-8 字节位置修改引用：不改变属性名、字符串和注释；重复执行不会再次限定；遇到本地同名变量拒绝猜测。构造方法使用明确拥有类和 receiver；保留普通装饰器，遇到 classmethod/staticmethod/property 描述符时要求先声明绑定契约。该改进不代表历史抽取 profile 已适配当前 HCU 的类拆分，`--check` 仍需完整预检。

回归包括真实函数执行、父类调用、重复变换、中文文本偏移、局部变量冲突、装饰器行为以及来源符号漂移时不写输出。方法库完整门禁结果单独记录，不计入插件测试数。

#### 2026-10-26 3.2 预导入别名

旧 patcher 先修改 owner 常量，再调用传播函数；传播函数读到的已是新对象，因而跳过旧消费者。修正后先捕获旧值并限制作用域，只更新明确存在、仍引用旧对象的 DeepSeek 消费者。测试同时检查无关模块和缺失属性，避免修复变成全局副作用。

#### 2026-10-26 3.3 执行身份

容器 root 执行只读 Git 审计时触发仓库所有权检查。审计、源码构建及回溯改由源码 UID 执行；运行服务沿用容器已有合法设备用户。未修改全局 Git safe.directory、HOME、系统 Rust 或设备权限。

### 2026-10-26 4. 源码、内嵌与 HCU 逻辑对照

本轮执行了三个不同层次的比较，原始结果为 `fork-logic-comparison.json`：

1. **产物保真**：442 个内嵌插件文件逐字节等于本轮插件；448 项生成清单核验通过。启动集成使用生成器定义的 bootstrap 映射，不能把不生成全局 `usercustomize.py` 误报为丢文件，也不能忽略入口顺序。
2. **来源保真**：284 项 JIT 资源逐项比较当前 HCU、插件、embedded 三方；Channel FP8 转换 helper 的 AST 与当前 HCU 相同。
3. **行为与版本差异**：检查整个 `python/sglang` 当前树，社区与 HCU 有 1206 项内容不同、387 项 HCU 独有、20 项社区独有。抽查的 24 个方法中，21 项需要适配/陈旧行为复核，3 项源拥有类或方法已变更。不能把文件名、签名、AST 相似或 210 个目标全部 apply 当成整个 fork 逻辑等价。

**对照结论：本轮交付形态和明确验证的契约已有证据；整个 HCU fork 的全逻辑等价尚未证明。** 当前 HCU 的 CP LayerSplit 全链路、候选/分页 FP4 索引、PD/DSpark 辅助状态、V4.1 中间 chunk decoder、EnGram、W4A8/DeepEP，以及 AOT `rope_fp4` 构建导出没有在本轮形成完整运行验收。这些能力不得标记为已迁移或已验证。

对旧源码差异的处理遵循当前 caller/consumer 契约。例如 MiniMax 的框架 `FfnExit` 已负责归约，保留 fork 旧生产者归约会重复求和；此处应保留有回归与服务证据的适配差异，不能为了文本相等恢复旧操作。

### 2026-10-26 5. 验证矩阵

| 检查 | 插件形态 | 内嵌形态 | 原始证据 |
|---|---|---|---|
| CPU 与契约 | 304 passed / 8 skipped | 304 passed / 8 skipped | `plugin-cpu-final.log`、`embedded-cpu.log` |
| 严格加载 | 210/210，三个 resolution seam | 210/210，屏蔽外部入口 | `plugin-runtime-final.json`、`embedded-runtime.json` |
| 单设备 GPU / Graph | 8 passed | 8 passed | `plugin-gpu-final.log`、`embedded-gpu.log` |
| TP4 native collective Graph | 四 rank × 三轮通过 | 四 rank × 三轮通过 | `plugin-collective-graph-final/rank-*.json`、`embedded-collective-graph/rank-*.json` |
| MiniMax TP4 服务 | health 200；`2、2、Paris`；正常 stop；自有进程清理 | health 200；`2、2、Paris`；正常 stop；自有进程清理 | `*-service-attempt1/*-service-result.json` 与原始响应 |
| 方法库门禁 | L0 通过；45 项工具单测通过 | 同一方法库，不重复计数 | `method-l0-final.json`、`method-tests-final.log` |

原始配方为上级 `minimax-debug/d.sh`；本轮按已有授权使用物理 4–7 卡 TP4，仅修改 TP 数和 NUMA 列表，保留 FA3、FP8 权重/KV、Graph、context 131072、chunk/prefill 和服务端口。`recipe-manifest.json` 保存原文与派生脚本哈希、精确差异及可见设备映射。

确定性响应通过 [响应校验工具](../tools/validate_generation_response.py) 校验实际模型、完整最终答案、正常 stop 和控制字符。该 checkpoint 在本次响应 content 中返回闭合前导 think 段，使用本次明确允许的 `allow_think_prefix`，未将 `enable_thinking=False` 宣称为全局停用思考的保证。重复请求记为 repeat，不作为 cache hit 证据；本轮不宣称 TP8、数据集精度或性能已通过。

### 2026-10-26 6. 复现与 Skill 入口

- 三方输入/差异审计：按 [迁移指南](migration.md) 调用 [迁移 Skill](../skills/sglang-hcu-plugin-migration/SKILL.md)，固定本轮 SHA 后生成新的审计和台账。
- 修改能力或抽取方法：按 [插件开发指南](plugins.md) 调用 [开发 Skill](../skills/hcu-plugin-dev/SKILL.md)，先保留真实构造、装饰器、状态与出口边界，再执行对应回归。
- 生成新 embedded：按 [回溯指南](backtracking.md) 调用 [回溯 Skill](../skills/hcu-plugin-backtrack/SKILL.md)。两个生成输入为 official/plugin；提供的 HCU 输入用于独立逻辑对照，不替换生成来源。
- 安装交付：按 [安装指南](installation.md) 使用已有源代码或 wheel，先验证实际解释器和模块来源，再运行目标服务。

证据脚本在本轮 ignored artifacts 中固定路径与 profile，只作为可复现实验记录；未来版本按对应操作指南重新固定输入和执行范围，不复用本页通过状态。

### 2026-10-26 7. 逻辑对齐后续工作

以下保留 v0.3.1 起的检查点与后续开发过程；各阶段的 pending 状态按当时范围解释，当前需求验收见 §8。完整 HCU fork 的所有模型与分支等价不属于这些局部通过结论。

后续开发位于独立的 `output-code/.sglang-plugin-v3-parity`（分支 `v3-hcu-parity`），以本页插件提交为起点。初始提交 `f2394ab` 迁移了 CP LayerSplit 的十个模块及分页/状态/描述符回归；后续 `0.3.2.dev1` 已接入父池、容量预算、工厂和握手规范化，进展见 §7.2。完整模型前向与 PD 网络字节传输尚未闭合，不能标记为可启用的完整 LayerSplit 服务。


#### 2026-10-26 7.1 CP 索引重排的设备准入修复

候选第二次局部回归为 23 项通过、1 项失败：HCU 主机上的 CPU 张量进入 LightOP 的 device-only remap，报 `indices must be device int32 or int64`。设备能力与输入 tensor 的实际设备不是同一个条件；当前候选同时检查 HCU 与 `indices.is_cuda`，CPU 张量使用原 PyTorch 路径。

修复后，候选局部测试为 **32 passed / 136 subtests passed**，证据 `cp-layer-split-candidate-attempt3.log`。新增对照包含真实 LightOP 的 int32/int64、乱序页、静态负值 padding、缺页、负 token ID，以及三轮变输入 Graph 重放；CPU 路径另检查不导入设备专用依赖。状态机测试中的部分边界沿用 HCU 的 AST 节点提取方式，不能代替完整模块导入、pool 构造或模型运行。方法来源及适配哈希在候选 `source-provenance.json`，能力接线状态在 `PARITY-CANDIDATE.md`。


#### 2026-10-26 7.2 CP 父池、容量与握手接线

候选 `0.3.2.dev1` 的实现提交为 `fe45284b9f17e0fde572d72a0751e9b8cddebb56`，显式 GPU 测试准入提交为 `1f1ff9d75d0857e5914b8fa328bce8acafd9cf87`。与最初仅搬迁模块的检查点相比，本次保留当前 HCU 的完整缓存池类，复用 AST 相同的官方值类型和独立 unified pool，通过正式 class REPLACE 接入五个缓存池类。同步迁移 DSV4 预算器、CP 工厂、容量 MIN 及 DSpark 全局描述符规范化。原有六个局部池 Hook 由一致的池构造/状态契约替代，候选的 HCU 目标集合变为 214 项。

声明的父池、INT8 helper、masked store、预算类、工厂及握手共 37 个类/函数节点，在明确 namespace 和插件进程选项映射后与当前 HCU AST 一致，证据 `hcu-pool-source-comparison.json`。该对照覆盖声明的节点与复用边界，不能替代其余模型/网络链路审阅。

本次增加了真实父池和四个 CP rank 的构造、公共工厂、分配缓冲与传输描述符、masked BF16 Graph，以及 packed FP4 threshold / RNE 舍入和变位置 Graph 读写。四卡验证使用真实 PyNCCL、真实 `ServerArgs` 投影和池分配，覆盖三轮 SWA/低压缩比 KV、非 owner 未预取拒绝、空查询 rank 参与、共享 staging 重用、容量最小值和真实分片/未分片池握手；没有启动 DeepSeek 模型引擎。

| 检查 | 候选源码 | 候选内嵌 | 原始证据 |
|---|---|---|---|
| 原有 CPU 回归 | 304 passed | 本阶段未重复全量 CPU；按改动执行下列池/设备验证 | `cp-parent-baseline-cpu.log` |
| CP CPU 合约 | 40 passed / 1 skipped，138 subtests；排除 11 个 GPU 用例 | 局部合约 38 passed / 2 skipped，2 subtests；排除读取候选源码的 AST 测试，包含实际池和设备边界 | `candidate-cp-cpu-final.log`、`candidate-embedded-cp-contracts.log` |
| 显式 GPU lane | 18 passed / 1 skipped | 18 passed / 1 skipped | `candidate-gpu-optin-final.log`、`candidate-embedded-gpu-optin-final.log` |
| CP4 真实池/通信 | 四 rank 全部通过 | 四 rank 全部通过 | `cp-tp4-pool-attempt2/rank-*.json`、`candidate-embedded-tp4-pool/rank-*.json` |
| 严格加载 | 214/214，正确源码来源 | 214/214，屏蔽外部入口并固定内嵌来源 | `candidate-pool-runtime-final.json`、`candidate-embedded-runtime.json` |

GPU lane 包含原有八项和新增十项实际通过用例。最初未设置 opt-in 的原有 GPU 运行仅产生八项 skip，该日志不计通过；随后明确设置 `HCU_PLUGIN_RUN_GPU_TESTS=1` 执行。CPU 与内嵌局部 lane 范围不同，不按通过数直接比较覆盖率。

回溯产物位于 `output-code/.sglang-das-v3-parity`，由干净候选和官方两输入生成。466 项清单核验通过，含 460 个插件文件。内嵌检查屏蔽外部入口，并确认 SGLang 与插件模块均来自该输出树。AST 提取测试若通过测试路径定位候选源码，就不能计作 embedded 覆盖；因此内嵌局部回归排除了这类测试，另以实际导入、构造和设备行为验收。

首轮 CP4 失败是验证器误把 PyNCCL 初始化后的默认 `disabled=True` 当作不可用。直接探针确认 RCCL 2.22.3 与基础符号正常；当前先验证 `available`，再在 `change_state(enable=True)` 作用域内断言真实归约数值，随后执行池广播。初始失败日志保留，没有把跳过 collective 当作通过。

两项 INT8 池用例要求源实现支持的 gfx936；当前节点为 gfx938，按 gate 跳过。此处不声明该特性已在本节点验证。候选 wheel 已构建并核对 460 个插件文件及新增 provenance 资源；尚未把它作为隔离 wheel 安装形态运行，不计作 wheel 运行验收。

CP 父池阶段结束时仍待完成正式启用入口、模型 prefetch/write/read 消费者、prefill/decode 描述符生产、Mooncake 追加字段及匹配复制和 DSpark 辅助状态。后续 PD 协议进展见 §7.3；整体集成完成后再合入正式输出并重新验收服务。`HCU_PLUGIN_CP_CACHE_LAYER_SPLIT` 目前仅供候选池/通信契约使用，不是完整模型能力承诺。


#### 2026-10-26 7.3 PD 描述符生产、协商与复制

候选升级为 `0.3.2.dev2`，提交 `f9a838c0de11e369bee70125c9703d06636c6763`，仍位于独立 `output-code/.sglang-plugin-v3-parity`。本阶段从公开 `setup_state_kv_args` 补齐 V4 描述符生产，将 CP 拓扑、KV 布局、类型信息接入 HTTP bootstrap，连接 decode 真实 receiver 构造、ZMQ 注册与 prefill peer 发布。注册消息保留旧 0–19 字段（ZMQ 帧），20/21/22 分别追加 state formats、state types 和 V4 描述符；短旧消息仍可解析。

复制入口按照 `(family, global layer)` 匹配指针和每项字节跨度。对来源实现保留的两处问题，本次补充了失败反例与适配：

- 描述符合法但目标顺序不同，位置式 stride 前检可能先返回 `-1`；函数级四项反例覆盖未缓存 verdict、单批/分批/线程路径。现在描述符路径由身份匹配检查 ABI，位置式检查继续用于原有路径。
- 状态组件已按类型和重复次序解析目标组件，目标描述符却仍用原索引；一项重排反例在修复前拒绝合法布局。现在指针、长度、索引与描述符都沿解析后的目标组件读取。

peer 发布前检查描述符与指针/stride 数量、二进制字段与 JSON 中 KV 大小的一致性；坏的重新注册会撤销旧有效 verdict。上游 worker 保留原控制流，只增加其当前注册快照的描述符参数，保证与该次指针列表一致，未改成重新读取可变 peer 表。

32 个 HCU 来源节点及一个上游 worker 调用适配完成明确映射后的 AST 对照，包含 async HTTP handler、classmethod 恢复、显式 super owner、文档字符串缩进及上述功能修复。`pd/pd-source-comparison.json` 和插件内 `disagg/source-provenance.json` 保存范围与哈希。该结论不覆盖未迁移模型及流式状态分支。

| 验证 | 源码形态 | 内嵌形态 | 证据（本轮 artifacts 下） |
|---|---|---|---|
| CPU 基础与合约 | 369 passed / 1 skipped；11 GPU deselected，138 subtests | 本阶段按 PD 改动面验证下列项 | `pd/full-cpu-attempt1.log` |
| 新增写入 drain / 公开链路 | 3 passed（2 个 drain、1 个隔离公开入口） | PD 单测及公开链路共 28 passed | `pd/drain-public-integration.log`、`pd/embedded-contracts.log` |
| 严格加载 | 241/241，正确来源 | 241/241，外部入口屏蔽 | `pd/runtime-final.json`、`pd/embedded-runtime.json` |
| 原生 Mooncake TCP 公开链路 | 通过 | 通过 | `pd/native-public-path-final.json`、`pd/embedded-native-public.json` |
| 显式 GPU lane | 18 passed / 1 skipped | 18 passed / 1 skipped | `pd/gpu-optin.log`、`pd/embedded-gpu.log` |

公开链路采用真实 CPU 池分配、HTTP/ZMQ 协商、真实 receiver 构造和原生 Mooncake TCP session / memory registration。通过公开 KV 与状态发送方法后，逐字节比较目标缓冲，包括未应写入区域。独立的有界 CPU engine 模式保留作回归，它与原生 TCP 证据分开记录；两者均明确不是模型服务或 RDMA 硬件验收。恢复旧 peer 前检的正常顺序组合对照也通过，因此不将函数级失败反例描述为所有基线调用都会失败。

新增写入完成测试覆盖一个 future 异常或返回错误、另一写入仍在执行的情况：确认 helper 等待已启动写入结束后才返回或抛出，防止提前释放 buffer。CPU 失败处理证据不等同于完整流式源事件/ACK 生命周期已闭合。

回溯使用干净的官方 `baa9befa...` 与候选 `f9a838c...` 两个输入，475 项清单核验通过，其中 469 个插件文件；证据 `pd/embedded-manifest-verify.json`。内嵌加载屏蔽外部入口，原生 TCP 验证器同时固定 SGLang 与插件的实际导入根路径。

已验证的 v0.3.1 主输出与其 448 项生成清单保留。本阶段没有把新候选的协议/设备结果写成重新通过 MiniMax 模型服务。

下一阶段仍需模型 prefetch/write/read 消费者、DSpark hidden-state 流式生产与 ACK、source event / 在途 buffer 生命周期、候选索引/稀疏 prefill、W4A8/DeepEP 和其余模型 caller 差异。整体目标仍在进行，不将本阶段协议路径通过写作完整 HCU fork 等价。


#### 2026-10-26 7.4 MQA 类图、缓存读取与原生边界

候选 `0.3.2.dev3` 的实现提交为 `5b182d7f8e39cd1637ca2c10bafb033c9e619911`。从当前 HCU 保留完整 `MqaAttentionBase`、`MQALayer` 与 `DSparkAttention` 类，替代四个旧 MQA 方法 Hook；DSpark 显式继承候选基类，避免仅更新模块别名仍留下旧继承关系。后端新增 metadata prepare、tail reset、普通/sparse-prefill 缓存读取及原生 FlashMLA 接口，HCU 严格目标集合为 246 项。

实际池与 ForwardContext 回归覆盖 CP 空查询的 prepare→SWA/extra prefetch→wait→返回顺序；旧 forward 在同一边界的反例为 1 failed / 1 passed，保留于 `attention/empty-cp-before.log`。普通 attention 的页索引和 sparse-prefill 独立 flat indices 分别重映射，检查共享 metadata 未被改写。昂贵计算和通信载荷可替换，但不绕过决定调用顺序的控制流。

真实 GPU 验证中发现并修复三项问题：

- 独立无 residual 的 RMS 进入当前环境缺少的 vLLM `_C.rms_norm`；HCU 设备路径改用实际可用 LightOp，连续/非连续输入及变输入 Graph 有独立参考。原 residual 和 batch-invariant 路径按现有契约保留。
- overlay 哈希和 extra include 正确，但 loader 的 `DEFAULT_INCLUDE + extra` 使官方同名头文件仍优先，真实 JIT 报缺少 HIP warp 宏。注册时在 BuildSpec 冻结前调整共享默认列表的有效优先级，验证引用、幂等和非 HCU 分支；没有修改输入源码或全局工具链。
- 当前 FlashMLA dtype matcher 拒绝 FP8 FNUZ typed carrier；在 HCU native 入参边界使用 uint8 view，保持 pointer、shape、stride 与 bytes。BF16 和非 HCU 保持各自载体。

24 个来源节点/构造器后缀、9 个实际复用的官方 HIP helper 完成限定 AST 对照，证据 `attention/source-comparison.json`；插件内 `models/deepseek/current_v4/source-provenance.json` 记录源/目标哈希、装饰器映射、carrier 适配及命名空间边界。构造器沿官方初始化后追加 HCU 选项，不把整个 HCU 后端构造器标记为搬迁完成。

小型 MQA 使用真实 GPU 权重、norm、RoPE、KV store、FlashMLA 和输出投影，分别执行 FP8 与 BF16 gather。Q 对独立 PyTorch 参考、输出对直接原生 attention 加独立投影参考均为最大差值 0。backend metadata 由显式 fixture 提供；这不是完整模型引擎，也不是原生 attention 算法的独立 FP32 精度证明。

当前回溯由干净官方与 `5b182d7...` 两输入生成，482 项清单通过，其中 476 个插件文件。wheel 的全部 476 个插件文件与候选逐字节一致，含新 provenance；wheel 资源验证不计作隔离安装运行验证。验证结果及边界如下表。

| 验证 | 候选源码 | 候选内嵌 | 证据（本轮 artifacts 下） |
|---|---|---|---|
| CPU 基础与合约 | 384 passed / 1 skipped，16 GPU deselected，138 subtests | attention 与 CP 导入/构造合约 40 passed / 1 skipped，16 GPU deselected，2 subtests；排除从候选测试路径读源码的 AST 用例 | `attention/full-cpu-corrected.log`、`attention/embedded-contracts.log` |
| 严格加载 | 246/246 | 246/246，屏蔽外部入口 | `attention/runtime-final.json`、`attention/embedded-runtime.json` |
| 显式 GPU / Graph | 23 passed / 1 skipped | 23 passed / 1 skipped | `attention/gpu-regression.log`、`attention/embedded-gpu-attempt2.log` |
| 小型 MQA FP8 / BF16 gather | 两种模式均通过 | 两种模式均通过，实际后端来自内嵌树 | `attention/mqa-gpu-attempt5.json`、`attention/mqa-bf16-gpu.json`、`attention/embedded-mqa-*.json` |

GPU lane 新增 packed KV/upconvert、变选择与长度 Graph、超过 int32 的 4GiB output stride，以及独立 RMS；gfx936 专用 INT8 用例在当前 gfx938 继续跳过。首次 CPU 回归的唯一失败是旧注册测试同时要求并禁止 class REPLACE，修正为按平台检查目标集合后全量通过。

内嵌 GPU 首轮为 22 passed / 1 failed / 1 skipped：MiniMax 测试的 PyTorch 参考在 `hipblasCreate` 报分配失败。该例隔离运行通过，同一完整 GPU 组重跑为 23 passed / 1 skipped；未改生产代码，也未确定首次失败的根因，保留 `attention/embedded-gpu.log` 与 `attention/embedded-router-isolated.log`。

完整 CP compressor/indexer、DecoderReplayTail runner/模型循环、DSpark streaming source-event/ACK、W4A8/DeepEP 和完整 DeepSeek 服务仍待集成。本阶段不替换原 v0.3.1 主输出，也未将局部 attention 结果记作重新通过 MiniMax 服务。相关经验与开发 Skill 使用方式见 [类图与空查询](development-methodology.md#14-attention-类图与空查询通信顺序)、[JIT 头文件](development-methodology.md#15-jit-实际头文件优先级) 和 [原生接口](development-methodology.md#16-原生算子-provider-与-fp8-carrier)。


本阶段保留性审计还发现官方索引文件 SHA 从 `606b51f2...` 变为 `acca1b00...`，mtime 对应回溯来源检查；HEAD 和源码工作区均未变，当前 index tree 与 HEAD tree 同为 `27762040...`，暂存 diff 为空。原始索引字节未备份，故不声称原索引字节保持不变，细节见 `attention/official-index-refresh.json`。生成器现对子进程设置 `GIT_OPTIONAL_LOCKS=0`，新增真实双来源 mtime 反例；旧实现两项子检查失败，修复后方法工具 46 项单测通过。后续实际回溯再次检查前后输入索引哈希，结果保存在 `attention/index-preservation-regeneration.json`。这一修复不修改插件实现，内嵌运行结论仍限定于已核验的相同产物字节。


#### 2026-10-26 7.5 低压缩比缓存的持久写入边界

候选 `0.3.2.dev4` 已提交为 `02e11c96418310bd3a69fcdfc606c988aea29791`：正式接入 C1/C2 decode、C2 prefill 的 Python wrapper 与 backend `_low_ratio_compress_fused`，新增四个目标，HCU 合同总数为 250。该调用点使用 `get_extra_key_buffer_for_write`，不再把读取用紧凑 staging 当持久 KV；非 owner 取得 None，由 Python 零页载体与既有 native no-store 分支配合，继续计算 latent、维护 C2 ring 并发布复制的索引 K。

7 个 HCU 来源函数完成 AST 对照，仅调整模块级相对 import 路径；使用的 `c1.cuh`、`c2.cuh` 与固定 HCU 来源逐字节一致。证据为 `compressor/source-comparison.json` 和插件内 `compressor-provenance.json`，不动态导入 fork。

反例分为两层：旧 wrapper 的三个无主 KV 场景在 `.view` 处失败；旧 backend 的六个池边界对照为 3 failed / 3 passed，失败均为非 owner 错把读 staging 传作写目标。新的六个真实 CP 池边界均通过，断言 no-store 的同时仍发布索引 latent，且共享 staging 字节不变。首版测试误以为读 getter 在所有阶段均应抛错，改为按实际池生命周期验证写入所有权，没有为迎合测试修改生产池行为。

原生六项 GPU 验证覆盖 C1、C2 decode、C2 prefill 的 owner/no-store，每项三轮变输入 Graph；对照独立 PyTorch latent 与 pair-state 参考，另检查 V4 页内 payload、scale 和保留区的写入范围。正式 loader 的函数 REPLACE 使用 `functools.wraps` 保留原 metadata，首次 provider 断言因把 `__module__` 当实现来源而失败；后续从正式公开函数调用验证实际行为，由严格加载报告核对目标。

| 验证 | 源码形态 | 内嵌形态 | 证据（本轮 artifacts 下） |
|---|---|---|---|
| CPU | 390 passed / 1 skipped，22 GPU deselected，138 subtests | 本阶段按 attention/写入改动验证，18 passed，11 GPU deselected | `compressor/full-cpu.log`、`compressor/embedded-contracts.log` |
| 严格加载 | 250/250 | 250/250，屏蔽外部入口且固定内嵌来源 | `compressor/runtime-final.json`、`compressor/embedded-runtime.json` |
| 显式 GPU / Graph | 29 passed / 1 skipped | 29 passed / 1 skipped | `compressor/gpu-full.log`、`compressor/embedded-gpu.log` |

回溯的 485 项文件哈希通过，包含 479 个插件文件；输入索引前后相同，见 `compressor/index-preservation-reconstruction.json`。wheel 内全部 479 个插件文件与当前候选相同，含新增来源记录；仅做资源核验，尚未作隔离 wheel 运行验收。

完整 CP query-lens、candidate/top-k indexer、模型循环和流式 ACK 生命周期仍待集成；本节不构成完整 CP 模型服务声明。新增经验与开发 Skill 的检查入口见 [持久写入边界案例](development-methodology.md#18-缓存读取-staging-与持久写入目标)。


#### 2026-10-26 7.6 完整后端依赖图与 CP 入口集成

候选进入 `0.3.2.dev5`，在此前逐方法接线后改为完整 `DeepseekV4AttnBackend` 类及相关 metadata、C1/C2/C4 indexer、compressor、候选与 CP strategy 依赖图。固定 HCU 来源的 42 个模块共 482 个顶层类/函数节点、9 个额外桥接函数节点完成声明映射后的 AST 对照；`backend/source-comparison.json` 与插件内 `hcu_backend/source-provenance.json` 分开记录原始哈希、导入路由、来源缺口和有意修正。旧后端成员 Hook 已退出，避免 class REPLACE 后再由旧方法覆盖。新增状态与 API 后 HCU 严格目标集合为 363 项。

新增十个 ForwardBatch 字段采用继承官方记录的 dataclass 声明，保留原有方法，使构造、复制及 pickle 保留新增状态。ServerArgs 继续使用固定官方 schema；`--enable-cp-cache-layer-split`（别名 `--hcu-cp-cache-layer-split`）和 `--disable-cp-cache-layer-split` 转为进程选项并由 worker 继承。CLI 解析及完整 ModelConfig 派生分别验证，不将仅解析成功当作运行时配置完整。

正式启用检查复用当前 HCU LayerSplit 约束：DeepSeek V4.1、compression ratio 0/1/2、prefill PD、interleave CP、CP>1、PP1、Mooncake 与 Compressor V2，且 prefill Graph disabled。HiSparse/HiCache、encoder SWA replay、单独 staging transport、Q8 sparse prefill 和不支持的 speculative 组合仍被拒绝。这些是来源的支持边界，不为通过测试删除约束。

本阶段复现并处理以下数据接口问题：

- Torch 候选源在宽度 259 时给出未对齐 FP32 行，原生 selector 拒绝。对齐 carrier 仅在该 caller 准备，低层 selector 的非法 dtype/offset/stride 拒绝保持不变；三种布局的变输入 Graph 与独立 mask/ID oracle 验证实际 copy 和只读合同。
- legacy/interleave 的 Python list/tuple 被当成 Tensor 使用 range 高级索引；interleave 四 rank 的反例均失败，修复为 slice 后通过，Tensor 路径保持来源行为。

BF16 KV 对照还复现旧 compressor Hook 读取环境变量的接口错误：代码导入 `environ` 模块，却按 `envs` 对象访问 descriptor，旧单测又向错误模块添加虚构属性。Hook 改为读取实际 `envs`，两项回归改为注册实际 descriptor 并通过环境变量驱动分支；CPU 两项通过后继续执行真实 BF16 设备复验，见 [配置对象案例](development-methodology.md#22-环境变量对象与测试替身的接口一致性)。

输入 HCU 源码还保留 graph selector 对缺失 `glm5_next` 模块的无条件 HCU 导入，节点工作区与固定 HEAD 均确认文件不存在。候选移除该失效分支并保留可用模型的官方选择行为；该条作为来源缺口记录，不声明迁移了 GLM5Next HCU 优化。

真实 GPU 层链路采用实际 pool、SWA allocator、声明的 ForwardBatch、公共 metadata 初始化与两层随机权重 MQA。验证包括 ratio 1、ratio 2、264-token 候选源/消费者，以及 1030-token 跨页 top-k（从 1025 个历史位置选择 512 项）。确认命中原生 dense logits、候选 logits 与 paged sparse consumer；原生评分对同一 packed K 的独立 FP32 运算，输出对 FP32 评分的参考路径，已执行场景最终输出最大差为 0。该参考不构成独立 FP32 attention 算法或模型准确率验证。

真实四 rank 使用公开模型并行初始化，新增第二个 CP communicator 沿来源分区创建；验证 equal/variable all-to-all、peer 零计数、整 rank 空输入/空接收、两个通信器的双 stream 归约及销毁。`backend/collectives-tp4/rank-*.json` 保存逐 rank 结果；不是模型引擎或 RDMA 传输验收。

| 验证 | 源码形态 | 内嵌形态 | 证据（本轮 artifacts 下） |
|---|---|---|---|
| CPU 基础与合约 | 最后完整回归 425 passed / 1 skipped，147 subtests；配置对象 2 passed；池视图相关 37 passed | 65 passed，9 subtests | `backend/full-cpu-final.log`、`backend/bf16-view-after.log`、`backend/embedded-contracts-final.log` |
| 严格加载 | 363/363 passed | 363/363 passed | `backend/runtime-final.json` |
| GPU / Graph | 37 passed / 1 skipped，13 subtests | 待记录 | `backend/gpu-regression-attempt1.log` |
| 实际 allocator / metadata / MQA | ratio 1/2、候选/paged consumer、跨页 top-k 均通过 | 待记录 | `backend/gpu-public-*.json` |
| CP4 独立通信器与交换 | 四 rank 均通过 | 待记录 | `backend/collectives-tp4/rank-*.json` |
| CP4 完整后端/层链路 | FP8；显式匹配批次参考，四 rank 最大差 0 | BF16；四 rank 最大差 0，实际来源为内嵌目录 | `backend/cp-mqa-tp4-*` |

新增的完整模型配置与四卡层探针暴露了合成 HF fixture 缺失 `sliding_window` / `index_head_dim`；补齐实际 ModelConfig 所需字段后再验证，不修改生产代码绕过配置派生。原始失败与每轮后续结果均保留。

设备验证前按已有授权释放了 `debug_lzg` 在物理 4–7 卡的四个调优进程，核对 cmdline、容器与进程 start-time 后发送 SIGTERM；保留该容器在前四卡的服务。`backend/stopped-tuners.json` 和设备/PID 快照记录实际范围，未将整容器停止记作执行结果。

本阶段仍未宣称完整 decoder/model loop、DSpark streaming source-event/ACK、W4A8/DeepEP、AOT FP4 或完整 HCU fork 等价。最终集成后仍须生成正式输出并重验原 MiniMax 配方。当前使用方法按 [插件开发指南](plugins.md#使用-skill-开发插件能力) 调用开发 Skill，选择 [前向记录](development-methodology.md#19-前向批次-schema-与复制生命周期)、[评分行对齐](development-methodology.md#20-候选评分的行对齐与消费边界) 或 [序列分片](development-methodology.md#21-python-序列分片与-tensor-高级索引) 对应回归，历史案例不构成全部任务的固定检查清单。

候选后端集成提交为 `19ec9e554a0cc27ab112bc143ba205f278692f46`。本次生成包含 532 个插件文件、538 项清单文件，逐项哈希核验通过；生成前后两个来源的索引字节哈希保持不变。wheel 的 532 个插件文件逐字节相符；尚未执行隔离 wheel 运行验收。

CP4 与单卡原生完整批次的对照仍未通过原数值阈值（`rtol=0.02`、`atol=0.003`）：四层最大绝对差约 0.01733。首层 Q 与单卡输入相同，首层注意力输出已有差异；该结果不能写成完整 CP 模型验收通过。关闭参考 compressor 融合后偏差未消失，继续检查实际原生输入与独立数学参考。相关原始失败保存在 `backend/cp-mqa-tp4-*.log`。

网络恢复后，后四卡被其他用户容器的 TP8 MiniMax 服务占用。按已有授权尝试停止容器，但节点权限以创建者不是当前执行用户而拒绝，未绕过权限或记录为停止成功。随后诊断在每卡约 7.6–8.0 GB 剩余显存下执行，探针设置约 3.9 GB 的 PyTorch 单进程分配上限；这不是设备空闲或独占的声明，也不构成性能测试。完整服务复验与需要较大显存的 GPU 回归仍需满足资源条件。

首层原生输入捕获与独立对照已完成：15 个查询的 Q、解码 KV 集合和 FP32 数学输出一致（`backend/cp-native-input-comparison.json`）。单卡重放相同输入时，完整批次与 CP 批次形状原生输出差为 0.0078125，按相同批次形状执行则与 CP 输出差为 0（`backend/native-shape-replay.json`）。验证器新增显式 `--match-reference-native-batch`，用于隔离这一因素；完整批次结果与匹配批次结果分别记录，未修改原阈值或生产注意力实现。见 [数值参考案例](development-methodology.md#23-原生计算批次形状与数值参考)。

BF16 设备复验在修复配置读取后进一步暴露 SWA 视图的单位错误：父池 getter 继承了来源中 packed FP8 的 584 字节宽度，但实际 BF16 分配的最后轴应为 512 个元素。修复前，真实父池及四个 CP rank 的五项 BF16 consumer view 均失败，五项 FP8 对照通过；修复后后端与父池合计 37 项 CPU 测试通过。该来源修正保留 FP8 和 uniform-FP8 行为，单独保存为 `backend/bf16-view-source-comparison.json`，不纳入 42 个后端模块原样映射的声明。修复提交 `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c`，前一个验证器提交为 `6db84b4f27ce0c1da1a4068c6d3633f724b02032`。

最终内嵌 BF16 四卡复验通过：`backend/embedded-cp-mqa-tp4-bf16/rank-*.json` 四项均为 passed，四层 Q、注意力输出和最终层输出与参考逐元素一致，最大差为 0；实际 SGLang/plugin 路径均位于候选内嵌目录。使用显式匹配原生批次形状及 Torch compressor 参考；本次 BF16 逐层完整原生批次与匹配批次的差也均为 0。该结果限定于真实 allocator、metadata、CP 池与小型随机权重 MQA 层链路，不是完整模型引擎或 PD 服务验收。内嵌 CPU 最终回归为 65 passed、9 subtests；源池相关回归为 37 passed。

本检查点最终候选为 `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c`，由该干净提交重新生成内嵌源码并重建 wheel。方法库仅保存文档、Skill 和工具，未放入运行时包；原 v0.3.1 主输出与其 448 项内嵌清单继续保留。候选尚未替换主输出，完整 MiniMax 配方的新候选服务复验仍需可用设备，历史通过结果不继承为本候选通过。

### 2026-10-26 8. 最终交付与需求验收

2026-10-27 在历史验证容器完成本轮验证，并将已验证提交同步到正式输出。运行时代码继续位于方法库之外；方法库的 HEAD 和暂存区保留原状，文档及 Skill 更新保存在工作区。

| 产物 | 正式位置（工作根目录下） | 来源与状态 |
|---|---|---|
| 源码插件 | `output-code/sglang-plugin` | v0.3.2.dev5，干净提交 `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c` |
| 内嵌源码 | `output-code/sglang-das` | official `baa9befa381511cd4d56f83980951e5a45fffc6d` + 上述插件；538 项清单、532 个插件文件逐项核验 |
| wheel | `output-code/dist/sglang_dcu_plugin-0.3.2.dev5-py3-none-any.whl` | SHA256 `c2260d30015e932faf33c09082758dee675113785353717a2750f75775544fac`；独立 CPU 安装/加载已验证 |
| v0.3.1 回退 | 插件分支 `validated/v0.3.1-before-v3-1026`；`output-code/.sglang-das-v0.3.1-validated-backup` | 旧插件 `39038d5` 和原 448 项内嵌文件保留 |

#### 2026-10-26 8.1 最终验证矩阵

下列日志相对 `artifacts/v3-20261026/backend/`。服务原始响应分别位于上级 `backend-source-service-attempt3/` 与 `backend-embedded-service-attempt2/`。

| 验证 | 源码插件 | 内嵌源码 | wheel |
|---|---|---|---|
| 完整 CPU 合约 | 435 passed / 1 skipped，147 subtests | 435 passed / 1 skipped，147 subtests | 435 passed / 1 skipped，147 subtests |
| 严格加载 | 363/363，三个具名 seam | 363/363，屏蔽外部入口 | 363/363，实际模块来自独立安装目录 |
| GPU / Graph | 37 passed / 1 skipped，13 subtests | 37 passed / 1 skipped，13 subtests | 未执行 |
| TP4 native all-reduce Graph | 四 rank × 两套捕获分配 × 三轮重放，通过 | 同范围通过 | 未执行 |
| MiniMax TP4 服务 | 自有 health 200；最终答案 `2`、`2`、`Paris`；正常 stop；清理通过 | 同范围通过 | 未执行 |

CPU 日志为 `final-source-cpu.log`、`final-embedded-cpu.log`、`final-wheel-cpu.log`；GPU 为 `final-source-gpu.log`、`final-embedded-gpu.log`；通信为 `final-*-collective-graph/rank-*.json`。部分 AST 抽取用例读取外部测试仓源码，因此独立加载、启动隔离和实际模块路径另行核对，不由总测试数推断产物来源。

服务仍使用原 `minimax-debug/d.sh` 的用户授权派生配方，仅将 TP8 改 TP4、NUMA 改为 `2 2 3 3`，物理卡为 4–7。保留 FA3、FP8 权重/KV、Graph、context 131072、chunk/prefill 等参数。每轮在待测目录下验证新进程来源，识别自有监听者后调用真实生成并按完整最终答案验收；前导 think 只按本轮明确允许的闭合前缀处理。重复请求不作为 cache hit 证据，也不作性能结论。

服务验证实际执行于候选目录；正式输出具有相同提交与 532 个相同插件文件，并在正式目录另做严格加载和模块来源验证。最终来源报告为 `delivery-source-runtime.json`、`delivery-embedded-runtime.json`；wheel 为 `final-wheel-runtime.json`、`final-wheel-startup.json`。

#### 2026-10-26 8.2 HCU 逻辑对照结论

`delivery-fork-logic-comparison.json` 比较正式内嵌核心与固定 HCU 当前树，列出 1208 个内容不同文件、387 个 HCU 独有文件及 20 个内嵌核心独有文件；这些计数是审阅清单，不直接代表行为相同或不同。生成器以官方源码为核心，HCU 行为经插件注册接入。

532 个插件文件与生成树相同，284 项当前 JIT 来源资源核验通过；完整后端的 42 个模块、482 个顶层节点、9 个桥接节点及十个前向字段有声明映射对照，BF16 SWA 维度修正单独记录。GLM 缺失模块导入、Python 序列切片、候选评分行对齐、FP8 原生载体和 BF16 元素维度均是明确的适配/修正。

**结论：明确迁移的节点及已执行契约有来源和行为证据；没有宣称整个 HCU fork 全逻辑等价。** 完整 DeepSeek 模型引擎、全部 decoder/DSpark streaming source-event/ACK、W4A8/DeepEP shared-expert、AOT FP4 构建导出及所有原生分支仍未形成全范围验收。CP4 的原生完整批次数值差异与显式匹配批次对照分别保留，详见 §7.6，不将对照选项解释为生产精度修复。

#### 2026-10-26 8.3 需求与证据对应

| 原需求 | 实际交付与证据 |
|---|---|
| 1. 使用给定社区/HCU 完善插件 | 正式源码提交 `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c`；各模块及修复见 §2、§7 |
| 2. official/plugin 两输入回溯 | 正式内嵌目录、来源 manifest 与 538 项文件核验 |
| 3. 空闲 GPU 测两种形态 | 指定容器内两份 GPU/Graph 与 TP4 native collective 通过 |
| 4. 原 d.sh 服务 | 两份本轮服务报告、配方哈希、三次生成和自有进程清理 |
| 5. 记录已完成功能模块 | 本文各阶段实现表、当前提交与支持边界 |
| 6. 比较 HCU 逻辑 | 正式核心差异清单、迁移节点映射及明确差异/未验证范围 |
| 6. 迭代 Skill 与文档 | [经验索引](development-methodology.md)、开发 Skill 的状态/原生/数值/来源测试契约及操作文档 |

最终逐项检查保存在 `backend/requirements-completion-audit.json`，包含来源保护、正式输出、测试、服务、文档与未覆盖边界。上述结论限定于本轮固定输入和执行配置；公共 PyPI 发布、TP8、全模型精度和性能不在通过声明内。

## 2026-10-27 V3 方法库导览与统一任务入口实施记录

> **本文用途**：记录 2026-10-27 方法库导航、统一 Skill 和 README 的建设范围、验证方法与限制。此次修改仅涉及方法库，不修改外部插件或内嵌运行时代码。

### 2026-10-27 1. 需求与交付对应

| 需求 | 实现 | 使用入口 |
|---|---|---|
| 修复网页导航与界面使用问题 | 离线文档阅读器、内部文档及章节路由、全文检索与类型筛选、搜索空态、请求复制反馈、窄屏导航状态；区分方法库和外部运行时 | [architecture.html](../architecture.html) |
| 新增总 Skill 检索并管理现有方法 | 新增统一入口，先识别意图并检索当前目录，再选择专项 Skill；利用已有信息，必要时补问，并在输入齐备后继续已授权任务 | [hcu-plugin-assistant](../skills/hcu-plugin-assistant/SKILL.md) |
| 从使用者角度完善 README | 补全文件树、功能及场景、简化调用方式，加入开发、安装和排障案例，说明人工使用、路径配置、维护检查及运行时边界 | [根 README](../README.md) |

统一入口是 Agent 的任务执行指引，不是单独运行的调度服务。它通过读取专项 `SKILL.md` 衔接执行；无需先安装全部 Skill 或修改客户端全局配置。总数为 1 个统一入口和 7 个专项 Skill。

### 2026-10-27 2. 网页构建与维护

网页的布局和交互维护在 [源模板](../templates/architecture.html)，由 [build_guide.py](../tools/build_guide.py) 读取当前共享文本构建。输出只写入 `architecture.html`；运行 `--check` 时不写入，快照过期返回非零。模板与内容不手工双份维护。

页面内嵌文档、Skill、参考和方法工具的阅读内容及检索信息，不依赖外部字体、脚本、CDN 或浏览时的目录请求。操作指南之间的链接转换为阅读器路由，源文件另有明确入口。外部网页链接保留为用户主动访问的链接。相对路径不得越出方法库，代码与原文按文本转义，不能关闭内嵌数据脚本或执行原文中的 HTML。

阅读器支持当前资料使用的标题、链接、代码块、列表和表格；Mermaid 图表以可展开源码展示，图形查看可使用支持 Mermaid 的 Markdown 编辑器。它不声称实现完整的 GitHub Markdown 渲染器。

新增构建工具后，本库共有 10 个方法工具。[runner 配方](../.workflow/l0-gate.yml)、[CI 说明](ci.md) 和 [仓库约定](../AGENTS.md) 已同步快照维护规则。实际平台自动 CI 接入状态保持原有范围。

### 2026-10-27 3. 统一入口的任务选择与补问

统一入口先枚举实际存在的 `skills/*/SKILL.md`，读取文档索引，再按目标、症状、组件和版本词搜索。候选表是初筛提示；实际文件的描述与适用范围是判断依据。根据阶段选择必要输入，避免将所有任务字段一次交给使用者填写。

| 场景 | 要求的处理方式 |
|---|---|
| 只问“怎样开发新插件” | 提供初始化指南和最小案例，不因缺少机器路径而阻断方法咨询 |
| 要求实际创建工程，尚未提供写入位置 | 选择初始化 Skill，询问外部新目录；等待时仍可解释结构、检查已有官方输入 |
| 已提供路径、日志或版本 | 优先读取和核实，不重复索取可得到的信息 |
| pip 或入口发现异常 | 选择安装 Skill，先核对解释器、metadata 和模块来源 |
| 已加载后的 Graph、TP 或数值回归 | 选择诊断 Skill，检索相似经验并定位本次首次分叉 |
| 要求生成 embedded 并对照 HCU | 选择回溯 Skill；HCU 作为独立比较来源，不伪装为生成器的第三输入 |
| 一个请求包含多项交付 | 按真实依赖衔接，不强制顺序执行全部专项 Skill |
| 找不到匹配入口 | 说明候选与差异，询问期望产物，不编造 Skill 或擅自扩展范围 |

上述行为完成了指引和调用资料的一致性审阅。此次没有把这些案例当作已执行的外部工程创建、安装或模型服务任务；Skill 元信息校验也不证明任意 Agent 都能正确理解所有自然语言请求。

### 2026-10-27 4. 验证与证据边界

| 检查 | 本轮结果及范围 |
|---|---|
| L0 | 通过：18 个 Python 文件、8 个 Skill；结构、资源链接和方法库阶段边界 |
| 工具契约测试 | 53 项通过；新增网页构建用例覆盖离线内链、代码与数据转义、表格代码中的管道符、目录边界和只读快照检查 |
| Skill 元信息 | 8 个入口通过 `quick_validate.py`；新入口与对应 UI 提示一致 |
| 文档与网页内链 | 文件、标题锚点、生成阅读器路由及内嵌原文一致性检查通过；计数以证据 JSON 为准 |
| 网页交互 | 无网络 JSDOM 测试通过：任务与章节跳转、跨文档链接、历史返回、错误路由恢复、检索及筛选、复制成功和拒绝反馈、窄屏菜单与 Escape |
| 响应式检查 | 检查样式规则及 390 / 1440 宽度下的 DOM 菜单状态；未测量真实浏览器布局几何 |
| 实际浏览器与截图 | 未执行：内置浏览器安全策略拒绝 `file://` 页面打开；未换用其他浏览器或方式绕过拒绝 |
| 外部插件、GPU 与模型服务 | 本轮未执行，不引用历史运行结果充当本次验证 |

证据位于方法库 ignored 目录 `artifacts/v3-20261027/`：`local-validation.json`、`guide-qa.json`、文档检查结果、远端检查日志及最终需求验收报告。ignored 证据需另行保管，不随 clone 自动分发。

新导航、Skill、文档和工具已落入方法库；原有未提交改动按原状保留。本轮不会自动提交或推送现有工作区。

## 2026-10-28 运行时代码迁入与入口验证

## 2. 迁移与提交台账

| 日期 / 步骤 | 完成内容 | 提交记录 | 验证与边界 |
|---|---|---|---|
| 2026-10-28 / M01 运行时代码迁入 | 525 个来源文件；180 个 Python 文件仅改命名空间；239 个源 Python 文件 AST 对照；286 个原生资源/许可证按原字节保留 | [`4d82a16`](https://github.com/Sherlolo/hcu-sglang-plugin/commit/4d82a1663a54c5d4075c5215acc2d91dcfc66750) | 本次静态迁入对照通过；不把旧 GPU/模型结果记为新包验证 |
| 2026-10-28 / M02 安装与启动入口 | 本仓打包定义、`hcu` 入口、显式 launcher、命名空间可选严格加载工具、阶段门禁及当前使用文档 | 同一代码提交 `4d82a16` | 65 项仓库测试、源码/wheel 各 363/363 项 Hook 和 3 个具名接口通过；完整服务未执行 |
| 2026-10-28 / M03 HCU 发现入口 | entry point 改为 `hcu = hcu_sglang:register`；版本 `0.3.2.dev7`；同步当前安装与开发说明 | [`443eb83`](https://github.com/Sherlolo/hcu-sglang-plugin/commit/443eb83eb56d91ec5a86c1c53989419fb0077667) | 源码/wheel 各 363/363 项 Hook、3 个具名接口；旧 `hcu` 名称隔离通过 |

代码提交：`4d82a1663a54c5d4075c5215acc2d91dcfc66750`，标题 `feat: import HCU runtime as hcu_sglang with isolated packaging`，共 557 个变更文件。范围包括 526 个运行时包文件、根打包定义、迁入清单、当前使用文档与 Skill、结构门禁和运行验证工具/测试；未迁入输出仓的 docs、scripts 或 tests。其后提交 `c8ba45a` 仅回填该代码 SHA 并重建导览。

当前入口在 M03 改为 `hcu`；M01/M02 中的 `hcu` 与 dev6 验证结果保留历史含义。

当前代码来源是已存在实现的迁入，不是宣称整个 DAS fork 已完成转换。模型能力、Native、CP/EP/PD 的历史范围仍以对应来源记录为线索，本命名空间必须建立自己的产物与运行证据。

## 4. 实际验证记录

### M01/M02：首次迁入（dev6 / hcu）

验证日期：2026-10-28。实际环境为 Python 3.10.12 与既有 HCU/DTK 依赖容器；官方源码固定在上述 `baa9befa`。安装、构建和仓库测试使用源码所属用户；真实 Hook 加载使用容器原有的可访问设备的执行身份。没有修改设备权限或共享 site-packages。

| 检查 | 实际结果 | ignored 原始证据（相对仓库根） |
|---|---|---|
| 在线来源核验 | 525 个来源/目标 SHA256 一致；official、fork、来源 plugin 的 HEAD 与干净状态保持 | `artifacts/v3-20261028/provenance.json` |
| 结构及仓库测试 | L0：261 个 Python、8 个 Skill；65 项 unittest 通过 | `artifacts/v3-20261028/l0.log`、`tests.log` |
| 源码隔离安装与冷导入 | distribution `hcu-sglang-plugin==0.3.2.dev6`，唯一 `hcu = hcu_sglang:register`；冷导入未载入 Torch/AITER/旧包 | `artifacts/v3-20261028/source-install.log`、`source-cold-import.log` |
| 源码真实加载 | 363 项目标集合完全匹配且全部应用；3 个具名 resolution 接口；软错误与未应用列表为空；未导入旧包 | `artifacts/v3-20261028/source-hooks.json`、`source-hooks.log` |
| wheel 构建与完整性 | wheel 内 526 个包文件路径/字节与源码一致；未携带外部 docs/tools、旧包或全局 Python 启动文件 | `artifacts/v3-20261028/runtime-validation.json`、`build-wheel.log` |
| 隔离 wheel 安装与冷导入 | 从独立 target、仓库外 cwd 导入；唯一新入口；冷导入无 Torch/AITER/旧包 | `artifacts/v3-20261028/wheel-install.log`、`wheel-cold-import.log` |
| wheel 真实加载 | 未将仓库根加入 PYTHONPATH；实际导入来自 wheel target；363/363 项和 3 个具名接口通过，无加载软错误或旧包导入 | `artifacts/v3-20261028/wheel-hooks.json`、`wheel-hooks.log` |
| 显式 CLI 参数入口 | 源码和 wheel 的 `python3 -m hcu_sglang --help` 均 exit 0；只证明参数入口可用 | `artifacts/v3-20261028/source-cli.log`、`wheel-cli.log` |
| 文档与 Skill | 8 个 Skill 元信息、导览快照及链接/锚点校验通过；迁移计划保持原三方快照并链接当前记录 | `artifacts/v3-20261028/local-validation.json`、`guide-qa.json` |

本次 wheel 为 `hcu_sglang_plugin-0.3.2.dev6-py3-none-any.whl`，SHA256：`dc992c7a420148204cee1e0a9274ca590ea555189fb9bc58d74b606e1f720e8a`。它包含 Python 与原生源码资源，实际运行仍依赖兼容 vendor 库及工具链；没有发布公共 PyPI 版本。

复现构建和隔离 wheel 安装（`ARTIFACTS` 为本次独立可写目录）：

```bash
python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir "$ARTIFACTS/wheels" .
python3 -m pip install --no-deps --no-build-isolation --target "$ARTIFACTS/wheel-site" \
  "$ARTIFACTS/wheels/hcu_sglang_plugin-0.3.2.dev6-py3-none-any.whl"
```

wheel 加载进程在仓库外执行，只将固定官方 `python/` 与本次 `wheel-site/` 加入 PYTHONPATH，并将严格验证工具的 `--plugin-root` 指向 `wheel-site/`。目标合同仍使用本仓迁入清单。

**环境排查记录**：首次用无容器 passwd 记录的普通 UID 验证时，Torch 默认缓存落到不可写目录，AITER 随后尝试写不可用的用户 JIT 目录。配置本次任务的 `XDG_CACHE_HOME`、`TORCHINDUCTOR_CACHE_DIR`、`TRITON_CACHE_DIR`、`AITER_JIT_DIR` 后缓存问题消除；首次 AITER 原生依赖构建完成。随后普通 UID 报设备不可用，而容器原有执行身份可见 8 张设备，真实加载因此使用后者。失败尝试日志保存在 `artifacts/v3-20261028/failed-attempts/`，不能将编译耗时或 loader 捕获异常后的退出视为加载通过。

**未执行范围**：新命名空间的模型生成、GPU 数值/Graph/多卡服务及 embedded 产物未在本轮复验。真实 Hook 加载使用了设备环境，但不能代替这些验收；旧命名空间的历史结果仅作为后续回归线索。

### M03：HCU 插件发现入口（dev7 / hcu）

代码提交：`443eb83eb56d91ec5a86c1c53989419fb0077667`（`fix: discover HCU plugin through the hcu entry point`，13 个文件）。

将根打包元数据入口改为 `hcu = hcu_sglang:register`，开发版本更新为 `0.3.2.dev7`。运行时实现及 363 项目标合同不变；同步当前安装说明、开发指南和 Skill，保留旧产物的历史入口。实际执行环境沿用固定官方 `baa9befa` 与 Python 3.10.12 / HCU 容器。源码与 wheel 均从新的隔离 target 安装；构建/安装使用源码所属用户，设备加载使用容器原有执行身份。

| 检查 | 本次结果 | ignored 证据（目录 `artifacts/hcu-entrypoint-20261028/`） |
|---|---|---|
| 安装 metadata 与冷导入 | dev7 仅提供 `hcu → hcu_sglang:register`；源码及 wheel 冷导入均无 Torch/AITER/旧包 | `source-cold-import.log`、`wheel-cold-import.log` |
| `SGLANG_PLUGINS=hcu` 源码加载 | 363/363 目标全部应用、3 个具名接口，无加载软错误或旧包导入 | `source-hooks.json`、`source-hooks.log` |
| `SGLANG_PLUGINS=hcu` wheel 加载 | 从仓库外 cwd、隔离 wheel target 加载，363/363 和 3 个具名接口通过 | `wheel-hooks.json`、`wheel-hooks.log` |
| 旧名称隔离 | 新进程设置 `SGLANG_PLUGINS=hcu`，0 项 Hook，新包未导入 | `old-name-isolation.log` |
| wheel 完整性 | 526 个包文件路径及字节与源码一致，无旧包或全局启动文件 | `runtime-validation.json` |
| 仓库门禁 | L0 与 65 项 unittest 通过；两个更新的 Skill、链接/锚点和导览快照校验通过 | `l0.log`、`tests.log`、`guide-qa.json` |

本次 wheel：`hcu_sglang_plugin-0.3.2.dev7-py3-none-any.whl`，SHA256：`5d61be21697b03c9c5e21cf26c7ec74a161357f7af15e93b0bf2c2e911740ecd`。新版本更新安装元数据后采用 `export SGLANG_PLUGINS=hcu`；dev6 的 `hcu` 结果不作为 dev7 的实际加载证据。本轮未执行模型服务、数值或 Graph 复验。

构建首次使用设备执行身份时被 Git 所有权校验拒绝，已改用源码所属用户完成构建；保留失败日志，未修改全局 Git 信任配置。

## 6. 2026-10-29 分阶段继续迁移

输入保持 official `baa9befa`、DAS `15c7204b`；本轮起点为本仓 `405b8dc`。输入只读，不推送远程。原固定盘点含 3,010 个非 same 路径，其中运行时范围差异 1,614 个；本轮执行台账将全部非 same 路径纳入，未审核项保持 pending，不能把 450 条历史映射线索当作当前行为覆盖。

| 阶段 | 本轮实施 | 退出条件与当前边界 |
|---|---|---|
| M0 基线与迁移工具 | 当前/旧包识别；清单和处置目标发现；抽取/签名 CLI；v4 embedded；全差异执行台账 | 离线契约与新包隔离加载夹具；当前类 profile 仍可能拒绝漂移 |
| M1 基础与量化 | 延续既有实现，逐项处置尚未完成 | shape/dtype/依赖/回退行为验证后更新人工决定 |
| M2 Attention/KV/模型 | 延续既有实现，逐项处置尚未完成 | 构造、状态与消费者、数值、Graph 和指定模型分别验收 |
| M3 通信/PD/扩展 | 新增 HCU StandardDispatcher 哨兵映射修复 | 公共 dispatcher CPU 回归；真实 EP 多进程、GPU Graph 与模型仍待验收 |
| M4 产物与交付 | 当前包 embedded 生成与核验能力 | 新包实际源码/wheel/embedded 严格加载分别记录；不代表全 DAS 等价 |

### StandardDispatcher 来源与行为

来源 `python/sglang/srt/layers/moe/token_dispatcher/standard.py` 的 helper 与 dispatch 方法迁入 `hcu_sglang/layers/moe/standard_dispatcher.py`。只把官方模块全局名显式限定为 `S.*`，保留来源注释；新增 HCU AROUND 门控与顶层注册。非 HCU 交还传入 original，保留前序包装。没有从 DAS 动态导入。

原始反例：映射 `[0, 1, -1, -1, 2]` 下，输入 `-1` 被 Python 负索引映射为共享专家 `2`；修复后负数与超出全局表的 ID 均保持 `-1`。正常 routed/shared IDs、DP pad 的后置屏蔽、空 batch 和跳过映射分别验证。当前完整目标集合为 [当前合同](hcu-sglang-development.md#历史-json-快照)（`hcu-sglang-targets.json`）；原迁入 manifest 的 363 项保持历史快照。

运行测试要求固定 official python 与本仓在 PYTHONPATH，使用已有兼容 HCU/Torch 容器，从仓库根执行：

```bash
python3 tests/runtime/standard_dispatcher.py
python3 tools/build_migration_execution.py --check
python3 tools/build_guide.py --check
python3 ci/l0_gate.py
python3 -m unittest discover -s tests -v
```

本次原始证据在 `artifacts/v3-20261029/`。逐路径台账的 implemented 只表示对应文件范围的代码接入，不是全领域完成。GLM5-next、Muse Spark、各 KV/Graph/native 分支等仍需逐项迁移与完整验收；本轮不宣称全量 DAS 转换完成。

### 本轮本地提交与实际检查

本地工具提交 `af5c635`：当前包识别、清单/处置/抽取/签名和 v4 内嵌工具，含旧包兼容与新包隔离夹具。运行时提交 `42fe693`：StandardDispatcher 修复、当前 364 项 Hook 合同、公共 dispatcher 回归和全差异执行台账。提交均未推送。

| 检查 | 实际结果 | 本轮证据 |
|---|---|---|
| L0 / 离线契约 | 267 个 Python、8 个 Skill；70 项 unittest 通过 | `l0.log`、`tests.log` |
| 真实公共 dispatcher | 4 项 CPU 张量测试通过，含原行为反例、EP 映射与共享专家、DP pad、空 batch 和跳过映射 | `dispatcher.log` |
| 源码严格加载 | 364/364 项应用、3 个具名 seam、无加载软错误与旧包导入；集合与当前合同一致 | `source-hooks.json`、`source-hooks.log` |
| 隔离 wheel 加载 | 仓库外 cwd、独立 wheel-site；364/364 与 3 个 seam 通过 | `wheel-hooks.json`、`wheel-hooks.log` |
| wheel 构建 | 本地 dev7 产物 SHA256 `d4ed63db35ee5ad9ef42a058e1bde39205bed60dc254c3b314d002c5abec3fea`，与历史同版本产物按 SHA 区分 | `wheel-build.log`、`artifact-checks.json` |
| Skill 与网页 | 3 个修改 Skill 的 validator 通过；重建导览、实际桌面和 390px 窄屏阅读器与导航开关检查 | 本次浏览器观察；快照门禁日志 |

测试环境为已有 HCU/DTK 容器、Python 3.10.12、Torch 2.11.0；公共 dispatcher 测试使用 CPU 张量并只控制设备/并行环境及昂贵通信条件，保留真实官方构造和 dispatch。节点宿主 Python 3.6、本机 Python 3.9 均不作为本项目验证解释器。首次未安装 AITER bootstrap 的测试导入失败，加入与 launcher 相同的兼容启动后通过；失败日志不能当作通过证据。


### 本轮实际 embedded 复验

从干净本地提交 `08b027e3228ace34537a5cf4a3e334f00ef9b747` 与固定官方 `baa9befa` 生成独立 `embedded-hcu` 源码树，迁入 527 个包文件；532 项 manifest 哈希全部通过。禁用所有外部 entry point、仓库外 cwd，仅加入产物 `python/` 后，364/364 项 Hook 和 3 个具名 seam 均通过，官方与插件模块均来自该产物，无旧包导入。证据为 `embedded-build.log`、`embedded-hashes.json`、`embedded-hooks.json` 和 `embedded-hooks.log`。产物位于本任务新建的 sibling `tmp/hcu-embedded-v3-20261029/`，不是原始 DAS fork 的逐文件复刻，也未执行模型/Graph/多进程 EP。

当前来源的方法同步预检明确失败在 `DeepseekV4DecoderLayer`：DAS 当前提供 Base/Cuda/Platform 拆分类。工具退出 2、不写任何输出；本次通过包名适配不等于该旧 profile 可直接执行。下一步须审查现有 `current_v4` 接入与平台类继承、状态和消费者，再决定删除旧抽取目标或补充精确 profile，不能将类名机械替换后覆盖既有适配。原始诊断见 `method-preflight.log`。

本轮结束时 official/DAS 两输入 HEAD 与起点相同、工作区干净。全量迁移总任务保持 in-progress；台账剩余 3,009 个未审核路径不等于 3,009 项缺失实现。

## 7. 2026-10-29 模型服务测试中的 spawn 启动修复

本轮以 cookbook 的 SGLang 0.5.18 单机八卡配方为参考，在已有 HCU 容器中运行本仓插件与固定官方 0.5.22 源码；它不是在容器预装 DAS 0.5.18 上加载插件的兼容声明。用户选定代表模型 Qwen3、MiniMax、GLM/DeepSeek，原始请求、响应、配方和运行记录保存在用户指定测试工作目录。

首次 Qwen3-235B-A22B-Instruct-2507 TP8 服务在加载权重前退出：父进程虽有 AITER shim，`runpy` 将主模块身份换成官方 launcher，spawn worker 在反序列化 scheduler 时先导入 quantization，触发旧 AITER 缺少可选 `gemm.fused` API。此前仅父进程的 Hook apply 不能覆盖此路径。

新增 `hcu_sglang.launch_server` 作为 spawn 可重入入口，在模块导入时仅安装惰性 shim；包 CLI/console 入口将主模块设为该入口，再以不替换主模块身份的方式执行官方 launcher。worker 在解析 scheduler 目标前先恢复相同 bootstrap，不安装全局 sitecustomize/usercustomize，不伪造真实 Quark kernel；真调用缺失 kernel 仍抛出原有 NotImplementedError。

版本更新为 `0.3.2.dev8`。真实 spawn 隔离夹具在旧 dev7 上失败、在修复后通过；71 项 unittest 和 L0（268 个 Python、8 个 Skill）通过。dev8 wheel 含 528 个包文件，逐路径/字节与源码相同，SHA256 `457e064d8d7a57413038ca453330604ef0ede781f79be33f6221eadb8d32f5b5`。同一 Qwen3 TP8 配方复跑已越过原失败点并进入八 rank 权重加载；最终生成结果在后续记录中补充。

测试工作目录内证据：`logs/spawn-regression-original.log`、`logs/spawn-regression-fixed.log`、`logs/repository-tests.log`、`logs/l0.log`、`artifact-dev8.json` 及 `runs/qwen3-235b-instruct-tp8-attempt2/`、`runs/qwen3-235b-instruct-tp8-attempt4/`。前者保留原始启动失败；后者为修复后的独立运行。

## 8. 2026-10-29 量化 CLI 与 forward-context KV 归属修复

GLM-5 INT4 W4A8 的 cookbook 配方首先在参数解析阶段被拒绝：运行时量化 registry 已有 `slimquant_w4a8` 和 `slimquant_w4a8_marlin`，CLI choices 却只加入 `w4a8_int8`。`server_args.py` 现在从同一 registry 常量导出全部三个名称；保持原有选择、幂等性和量化实现的惰性加载。真实 CLI help 已包含这些名称，离线回归检查实际 argparse 接受与不提前导入 Torch/量化实现。

随后用空闲四卡的小型 Qwen3 诊断完整 forward，确认 Torch/HIP 当前设备与张量设备一致，并在 Graph warmup 复现 `ForwardBatch` 不再拥有 `token_to_kv_pool` 的异常。固定官方在 `model_executor.forward_context.get_token_to_kv_pool()` 从当前 backend 解析 KV 池；共享 fused rotary、Qwen2 融合路径和模型公共融合判定均改用此接口，不补写旧字段、不缓存单一全局池。

真实 forward-context 回归在旧 dev9 wheel 上失败、在修复源码上通过：无旧 pool 字段的 batch，在外层/内层/恢复后的 context 中分别写入对应 KV 张量并保留融合 store 标记。昂贵的融合数学 kernel 被替换成可观察的 CPU 写入，官方 context 与实际池选择未 mock。证据为测试工作目录 `logs/kv-context-original.log`、`logs/kv-context-fixed.log` 与 `tests/runtime/fused_kv_context.py`。

当前开发版本为 `0.3.2.dev10`，wheel SHA256 `239e573e98ed5b159d84534a1bccf4ce18d06ef0e550c67bd05034281da71146`。CLI 修复后的 72 项离线 unittest 通过；模型结果另行记录，不将解析或 CPU 写入测试当作整模型精度通过。

## 9. 2026-10-29 四卡代表模型复验与融合门控修复

用户在 GPU 0–3 被其他服务占用后授权先验证四卡服务；使用物理 GPU 4–7、独立 ROCR 可见映射，保留其他服务。Qwen3-30B-A3B-Instruct-2507 BF16 TP4 在 dev10 中完成权重加载，越过 init_cublas 后于 Graph warmup 的模型公共 helper 导入处失败。原始证据为测试工作目录 runs/qwen3-30b-instruct-tp4-attempt1/；不是精度通过。

按失败路径对照固定 DAS models/utils.py，现有迁入 helper 使用已删除的 is_prefill_context_parallel_enabled，且通配导入不会导入私有 _is_cuda/_is_hip 标志。当前改为显式导入和官方 layers.cp.utils.is_cp_active(forward_batch)，保留 DAS 对 HCU 的 CUDA/BF16/非 SWA 门控、当前 CP 与 dcp_kv_mask 限制，其他 HIP 平台沿当前官方门控；池仍由真实 forward context 提供。不运行 DAS 对照服务，不动态导入 DAS 实现。

开发版本 dev11；新增 tests/runtime/model_utils_fusion.py 验证真实 helper 导入、forward-context 池、当前 CP 分支、DCP mask、SWA/dtype 和非 HCU 路径。修复前反例、修复后检查、wheel 与模型复验结果见下文；CPU 和打包检查不代表模型通过。

实际检查：dev10 wheel 在 logs/model-utils-original-attempt2.log 复现已删除 API 的 ImportError；修复后 logs/model-utils-fixed.log 的 3 项真实 helper/forward-context CPU 回归通过。72 项离线 unittest 与 L0 通过（logs/repository-tests-dev11.log、logs/l0-dev11.log）。dev11 wheel SHA256 e7badb052105ea5f06be25f92249427bb2e6fdcfd8931615730352ce4b5e07be，528 个包文件与源码和隔离安装目录逐字节一致，见 artifact-dev11.json。

相同四卡 Qwen3 配方的 dev11 attempt2 在权重和 16K KV 缓存加载后，仍于未修改的 init_cublas 16x16 FP16 矩阵乘法触发 rocBLAS Tensile kernel 缺失；15:53:57 终止，已记录自有进程清理为空、端口无监听。此轮没有请求或模型精度通过；Graph 原失败点的完整模型复验仍 pending。固定 DAS 与官方 attention_backend_setup 文件相同、init_cublas 调用及辅助函数相同，未据此认定环境或插件为最终根因。

## 10. 2026-10-29 按触发路径对齐 DAS 的 AITER 通信逻辑

用户明确要求插件与 DAS 的实现路径对应，不以关闭优化路径代替修复。本轮新增 distributed_aiter.py，迁入固定 DAS custom_all_reduce.py 的三个函数：AITER 注册选项、工作空间大小与完整 dispatcher；现有 HCU dispatch Hook 懒加载该实现，非 HCU 保留传入 original。HCU auto 不再按旧 SGLANG_USE_AITER_AR 默认关闭 AITER，而沿 DAS 的确定性判断与 AITER 导入选择。显式后端参数在 helper 层保留；官方 CLI/GroupCoordinator 的新增后端字段及 strict-Fabric 组合尚未在此轮移植，不宣称该组合已支持。

迁入保留 GraphSafeAiterCustomAllreduce：Graph warmup 运行真实 all-reduce/fused AR RMS、传递 live residual，capture 与 piecewise 路径交还 AITER 父类；IPC 注册受 memory-saver 及 DeepSeek-V3/GLM 特定拓扑约束，Fabric/auto 使用 copy-in。保留 DAS 的环境值校验、显式 aiter 与确定性模式拒绝、auto 下仅 ImportError 回退；不吞 kernel 错误。

来源文件 SHA256 为 2fe4be4c81564948b96d283e66f326f6a37b46ccc13b29fde051c581eb7834f9。只做模块全局名限定、HCU 检测入口及相对导入适配；还原这些适配后三个函数的 AST 与 DAS 完全一致。生产运行不动态导入 DAS。静态核对、修复前后 CPU 回归、wheel 和服务结果分别记录；当前准备的开发版本 dev12 不代表模型或 GPU collective 已通过。

实际执行：dev11 在 logs/aiter-align-original.log 的默认 HCU 后端反例失败；修复后 logs/aiter-align-fixed.log 的 6 项针对性 CPU 回归、72 项仓库 unittest 和 L0（272 个 Python、8 个 Skill）通过。dev12 wheel SHA256 30fbb912c349aed615c8febc8631f4e765b873cec7be40df9e6046162cf8c758，529 个包文件与源码、隔离安装逐字节一致。Qwen3 四卡 attempt4 的四个 rank 均记录 AITER IPC、copy-in，越过 BLAS 初始化后在 Graph 暴露 fused_topk 旧签名不接受 num_fused_shared_experts；没有生成请求或模型精度通过。此过程同时经历节点重启及主机内存恢复，不能据此将先前 rocBLAS 错误单独归因为通信选择。

## 11. 2026-10-29 MoE top-k 当前调用契约与 DAS 分支对齐

Qwen3 TP4 dev12 在 Graph warmup 经真实 select_experts 调用，因旧 REPLACE 的 fused_topk 缺少 num_fused_shared_experts 参数退出。固定 DAS 与官方 fused_topk 函数去除唯一 HCU LightOp 分支后 AST 完全一致；两输入文件哈希已与本次来源核对。插件改为 AROUND，完整保留官方的 11 项参数，非 HCU 及其他分支委托传入 original，保留已有包装。

仅在 HCU/softmax/原有 SGLANG_USE_FUSED_TOPK_SOFTMAX 开关启用、且不优先命中 AITER 分支时执行 DAS 的 LightOp topk_softmax。AITER 优先级、sigmoid、路由缩放、共享专家数、packed 输出与非 padding token 数交还固定官方的相同实现，不吞 kernel 错误，不用忽略新参数规避启动异常。版本 dev13；公共 HookRegistry 分发和 CPU 消费者回归、wheel、模型复验分别记录。

实际回归：dev12 wheel 在 logs/topk-original.log 通过真实 HookRegistry 包装的公共入口复现同一个 num_fused_shared_experts TypeError；修复后 logs/topk-fixed.log 的 6 项公共 Hook/CPU 消费者测试通过。72 项仓库 unittest、L0（273 个 Python、8 个 Skill）通过。dev13 wheel SHA256 8f7f9a0d69e6e96c2d180905f11932a4d9f799022c64d59d3d8491fda2b98b6e，529 个包文件与源码、独立安装一致。来源逻辑核对见测试目录 source-diff/topk-source-alignment.json。

模型复验：runs/qwen3-30b-instruct-tp4-aiter-attempt5/ 使用物理 GPU 4–7、官方 0.5.22、dev13 wheel、DAS 对应 AITER IPC 路径及正常 Graph，完成 batch 1/2/4/8 捕获。六个确定性请求严格 6/6 通过：323、北京、Paris、4、10、重复 323，全部 finish_reason=stop。17:31:25 完成，cleanup_remaining_pids 为空、端口无监听。仅代表该四卡短请求 profile，不外推为 235B 八卡、数据集精度、吞吐或全部 DAS 等价。

## 12. 2026-10-29 GLM/DeepSeek Decoder 退役旧替换并恢复当前 stage

GLM-5 INT4 TP4 在加载权重前被旧 Decoder 构造器拒绝 skip_rope。固定官方与 DAS 的 DeepseekV2DecoderLayer 全部方法已相同；当前实现使用 attn_boundary/ffn_boundary，旧插件仍使用 layer_communicator 且 forward 多出 residual 参数。按实际调用/消费边界成对退役该类的 __init__/forward Hook 和两份不再使用的旧实现，使用当前官方与 DAS 相同的真实类，不复制旧函数后仅添加参数。

当前 Hook 合同去掉这两个目标，变为 362 项；迁入 manifest 及历史 364 项验证保留原义。新增 tests/runtime/deepseek_decoder.py 经真实注册入口验证当前构造、skip_rope/build_stages、NextN NoPE、归约 owner 及 native forward 对 stage 字段的消费。版本 dev14；实际 CPU、wheel 和模型结果在执行后补充。

## 13. 2026-10-29 当前 DeepSeek V2 调用契约与 DAS 数据流同步

GLM-5 INT4 TP4 dev14 仍在加载权重前失败：旧 Attention 构造器给当前 ColumnParallelLinear 传入已删除的 tp_rank。逐项审查后，本轮从固定 DAS 同步现有 24 个注册方法，覆盖 MLP、MoE 与 MLA/MHA/ROCm/CPU attention；保留当前 parallel_group、skip_rope、输入量化载体、非 padding token、共享专家与归约 owner 的完整参数和消费路径。只限定实际 provider 的全局名、显式 super owner 和插件依赖导入；移除文档字符串缩进影响后，24 个方法的规范化 AST 均与 DAS 相同。旧全量同步 profile 仍因 V4 类拆分预检失败，本次没有绕过预检运行它。

DAS 新增但固定官方没有的 fused_rms_quant、GLM5-next 模型类型谓词与 MoE graph 输出桥接放入插件命名空间。运行不导入 DAS 源码。W4A8 线性层同步 supports_prequantized_input 和显式 INT8/FP32 输入检查，显式 q/scale 不再依赖无关 RMS 开关；保持原 GEMM 后端。该谓词不代表完整 GLM5-next attention 后端迁移已完成。

同时核实并修正共享 Qwen/Bailing fused rotary 门控：沿 DAS 的 SGLANG_USE_FUSED_RMS_ROTARY 默认关闭语义，Qwen3-MoE 保留原生融合分支优先级。已有 dev13 Qwen 6/6 属于旧版本结果；当前版本必须另行复验，不能将旧请求记作新版本通过。

开发版本 dev15。tests/runtime/current_v2_dataflow.py 检查量化载体经真实 prepare 到线性层的传递、真实 CPU 数学消费、非法载体拒绝、graph 共享输出所有权与 Qwen 分支门控；仅替换昂贵外部 kernel。实际 CPU/仓库门禁、隔离 wheel 与 GLM/DeepSeek 服务结果记录于测试目录，不以静态同步证明整模型精度。

实际检查：24 项规范化方法 AST 核对通过；5 项当前 V2/W4A8/graph/Qwen CPU 回归、72 项仓库 unittest 与 L0（276 个 Python、8 个 Skill）通过，证据为 current-v2-ast.json、logs/current-v2-dataflow-dev15.log、logs/repository-tests-dev15.log、logs/l0-dev15.log。dev14 GLM attempt2 保留原 ColumnParallelLinear tp_rank 异常；新模型启动与请求结果另记。MiniMax-M2.5 BF16 TP4 dev13 已完成 6/6 严格请求检查，属于该版本四卡短请求结果。

## 14. 2026-10-29 GLM EAGLE NextN 当前 stage 同步

GLM-5 INT4 TP4 dev15 四个 rank 已加载目标模型，在 EAGLE 草稿初始化触发旧 NextN 的 is_dsa_enable_prefill_cp NameError；原实现同时保留已删除的 Decoder CP 参数、residual 参数和三值返回。按固定 DAS 同步现有两项 NextN Hook，使用 layer_stack、skip_rope、当前 residual_batch 边界与两值 Decoder 返回，保持 IndexTopKShareState 更新/发布顺序和 SBO stream 选择，不关闭 EAGLE。

两项规范化方法 AST 与固定 DAS 相同；缺失的 legacy CP helper 复用插件现有 HCU backend provider，CP-v2 可用门控按 DAS HCU 默认关闭语义读取环境值，避免读取官方已废弃的 EnvBool registry 名称。通用 CP/GLM5-next 全路径尚未验证，不能从本轮 attn_cp_size=1 的配方外推。生产运行不导入 DAS 源码。版本 dev16，tests/runtime/nextn_current.py 通过真实公共 Hook 调用核对当前构造器契约、SBO stream 和 forward 的 residual/返回消费顺序；实际门禁、wheel 和模型结果另记。

## 15. 2026-10-29 DSA KV 池与 DSpark 解析路径同步

GLM dev16 目标/草稿加载越过 NextN 后，在 KV 池构造因 index_kpool 参数失败；DeepSeek-V4 dev16 首轮在解析阶段因原始 device=None 被旧包装器读取而失败。本轮分别沿触发边界对照固定 DAS，不运行 DAS 对照服务。

DSA 池迁入当前构造与缓存方法命名空间，34 项方法的规范化 AST 与 DAS 相同。现有 AROUND 转发所有参数，在 HCU 消费完整 index_kpool/compress、tail、request、DCP 和 indexer_layer_ids；非 HCU 委托原函数。缓存 facade 同步紧凑全局层映射，保留当前 FP8/BF16/INT8 模式选择、真实缓存读写/搬移和共享 workspace 所有权。INT8 helpers 保留 gfx936 opt-in 限制，读取环境值以适配官方未提供的新 EnvBool；本次 gfx938 FP8 配方不构成 gfx936 INT8 的整模型验证。DAS 的 _should_allocate_index_layer 在当前来源误带 property，但实际消费者调用该方法；保留官方同一函数体的 callable 绑定。facade 补齐通过现有 GetKAndS 消费的连续 K/scale 接口，避免当前 DAS 方法调用不存在的 facade 名称。新增 host_pool_decls Hook，当前合同变为 363 项，历史 362/364 项结果不改写。

DSpark 同步 DAS 的 _handle_dspark 与 PD/CP 资格判断，读取真实 resolving_view，沿 declare_resolution 写入决策，不再临时改写原始 device。保留 HCU DeepEP target/draft runner、静态 verify 与 CP/PD 限制。其他平台沿传入 original；运行不导入 DAS 源码。完整 PD/CP、INT8 GPU kernel 与长请求仍未验证。

开发版本 dev17。5 项 tests/runtime/dsa_dspark_current.py 通过真实公共 Hook 检查构造参数、page/压缩 tail、紧凑层读写、key/scale 搬移、非 HCU 参数保留和真实解析 overlay；仅替换外部 MLA 分配与昂贵 kernel/配置文件读取。证据为 current-dsa-audit.json、current-dspark-audit.json、logs/dsa-dspark-current-dev17.log。仓库门禁、独立 wheel 与模型复验另记。

已有 dev16 四卡复验：Qwen3-30B-A3B-Instruct-2507 attempt6、MiniMax-M2.5 BF16 attempt2 各 6/6 严格通过，Graph 保持开启，均清理自有进程且端口无监听；当前 GLM/DeepSeek 尚未通过，不将启动进展记作精度结果。

## 16. 2026-10-29 DSA HCU FlashMLA metadata 与 Graph 路径同步

GLM dev17 已加载目标/草稿并分配 KV，在 Graph 初始化误调用 CUDA sgl_kernel.flashmla_ops。固定 DAS 使用 flash_mla.flash_mla_interface 和对象 scheduler metadata。本轮迁入当前完整 DSA backend/multistep 类及 metadata producer/manager/indexer 依赖，HCU 注册两个类替换；其他平台沿原类。六个选择出的定义还原命名空间导入后 AST 与 DAS 相同，实际 package/probe 记录在测试目录。相同的 kpool mixin 与其他依赖复用固定官方，不动态导入 DAS。

保留 HCU metadata wrapper 的创建、刷新、Graph 消费、MLP-sync 实际行裁剪和输出补齐；不将 backend 对象塞入旧 CUDA tensor metadata 结构。补齐两个 EnvBoolWithAlias 字段，类型、默认 false 和旧名称沿 DAS；保留原始用户输入与环境兼容。CP helper 复用现有 HCU provider 和原生 kernel；本轮 TP4 非 CP 不外推 CP/PD 验证。

版本 dev18，当前 Hook 合同变为 365 项。tests/runtime/dsa_flashmla_current.py 的两项 CPU 回归通过：真实 producer 到 attention 消费边界传递新 scheduler 对象/num_splits，真实补齐行还原，HCU 保留 fresh source 对象而 CUDA 保留 tensor 地址；仅替换外部 FlashMLA kernel。证据 logs/dsa-flashmla-current-dev18.log、current-dsa-backend-audit.json；实际 wheel 与服务复验另记。DeepSeek-V4 dev17 尚在旧模型 get_pp_group 构造路径失败，Base/Cuda/Platform 拆分的迁移继续处理中。

## 17. 2026-10-29 DSA backend 实际 entry point 注册修复

dev18 wheel 的 542 个文件通过字节验证，但完整 Hook 合同检查发现两个 DSA 类未注册：新增调用仅放入旧 layers.attention.register_hooks，实际入口 hcu_sglang.register_optimizations 使用显式模块列表。修正在实际入口导入与调用 current_dsa；版本 dev19。保留 dev18 的失败合同证据，不将其 363 项已应用结果称为 365 项通过，也不将 dev18 GLM 复跑当作新后端验证。下一轮先确认完整 wheel 合同通过，再启动服务。

## 18. 2026-10-29 DSA TopK 当前依赖与 paged v2 门控同步

dev19 完整 wheel 的 365/365 Hook 与 3 个 seam 均通过；GLM 实际 Graph 已越过 CUDA FlashMLA 扩展错误，在当前 metadata producer 调用旧 DSATopKBackend.should_use_topk_v2_paged 时失败。同步固定 DAS TopK dependency 模块并让 backend/indexer 使用同一枚举与方法，避免新旧 consumer 混合；补齐 SGLANG_HCU_TOPK_V2_PAGED EnvBool 默认 false，沿 DAS 对 HCU paged v2 的明确 opt-in，保留 ragged v2 的 layout/dtype 条件。版本 dev20，完整服务复验继续；没有请求或模型精度通过。

## 19. 2026-10-29 DSA indexer HCU Hadamard 路径同步

GLM dev20 已越过 FlashMLA metadata 与 TopK 接口，实际 Graph warmup 的 Indexer forward 在 native HIP 旋转导入 fast_hadamard_transform 时失败。固定 DAS 对 HCU 默认使用缓存 dense Hadamard 矩阵及 F.linear，显式 SGLANG_USE_FAST_HADAMARD_TRANSFORM 才选择扩展。本轮迁入该缓存、数学消费和完整 rotate_activation(apply_scale=True) 函数，保留 power-of-two 校验与归一化开关；不以其他旋转或随机矩阵替代。显式 fast 扩展开关缺包仍正常报错。版本 dev21，当前合同 366 项；真实 CPU/GPU 数学与完整服务复验另记。

## 20. 2026-10-30 DSA indexer 当前 HCU producer/consumer 同步

在已触发的 Indexer 路径继续核对，native HIP logits 分支仍断言 page_size=1，而 DAS/HCU 当前池使用 page_size=64。本轮不启动这个已知不匹配的 dev21 全模型组合；dev21 三项 CPU/GPU Hadamard 回归通过，真实 BF16 GPU 最大绝对误差 0.0171714，属于单算子证据。

迁入固定 DAS 的完整 Indexer 类及当前本地 helper，HCU 通过原 module 和 DeepseekV2 的实际构造别名注册同一实现，避免只替换类定义后原 caller 仍持有旧类。保留 HCU FP8/BF16/INT8 cache 读视图、LightOp dense/sparse logits 分发、与 mask TopK 配对的资格检查、head-gate scale 和 logits scratch 预算；既有相同依赖复用官方，Hadamard 复用已验证的 DAS 实现。sparse producer 只有匹配的 consumer/API/布局/门控全部满足才可运行，未命中沿 DAS dense producer。运行不动态导入 DAS。

版本 dev22，合同 368 项；隔离暂存导入已通过，CPU/GPU 消费者、来源 AST、wheel 与完整服务结果继续记录。gfx936 INT8 与 CP/PD 仍未执行整模型验证，不能从 gfx938 TP4 profile 外推。


本轮实际检查：Indexer 与两个 HCU logits helper 规范化 AST 均与固定 DAS 相同；4 项消费者回归通过，涵盖 owner/broadcast 读存储、page64 packed ABI、dense clean-logits 参数、sparse producer/consumer 配对门控以及真实 HCU head-gate Triton 数学。72 项仓库 unittest 通过。依赖字段审计补齐 DAS 保留的 SGLANG_NSA_MQA_LOGITS_MEMORY_BUDGET_GB EnvFloat(None)，无全局预算覆盖；证据 current-dsa-indexer-audit.json、logs/dsa-indexer-current-dev22.log、logs/repository-tests-dev22.log。

## 21. 2026-10-30 HCU paged-MQA 后端选择依赖同步

GLM dev22 在资源检查停止两轮后，使用通过准入的物理 GPU 0/1/6/7 越过前述路径，进入新 Indexer 的 paged logits 分发；其残留原生枚举缺少 is_lightop。迁入 DAS 当前 DSAPagedMQALogitsBackend，并让 Indexer 使用该同一枚举。HCU auto/显式 lightop 选择 page64 LightOp；普通 HIP auto 仍选择 AITER；不支持的显式后端正常拒绝，不把 HCU 当普通 HIP。当前配方使用 auto，不宣称已验证所有显式 CLI choices。版本 dev23，合同仍 368 项，模型复验继续。

## 22. 2026-10-30 四卡 W4A8 Triton producer/consumer 路径同步

GLM dev23 已越过 attention/indexer，在真实 MoE warmup 触发已安装 LightOp Marlin 对 E=256,N=512,K=6144 的 unsupported size。当前 DAS auto 同样优先选择可导入的 LightOp，不能把它记作 plugin 独有精度错误；DAS 另提供显式 SGLANG_W4A8_TPMOE_BACKEND=triton，而旧插件仅 auto/lightop/aiter。本轮同步固定 DAS 的完整 SlimQuant Marlin 配置/后端实现和独立后端规范化/context helper，保留 auto 的原选择，四卡新配方明确选择 Triton。

权重生产端必须在载入前选择后端：Triton 保留原 packed INT4 权重，消费同一 channel scale，沿 DAS 的 chunk、block 适配、align、两次 GEMM、激活、路由/共享输出与缩放；不能把已 Marlin repack 的权重直接送入 Triton。版本 dev24。当前仅选择标准 TP4/EP1 路径；完整 DeepEP W4A8 与 DSpark override 尚未执行模型验证。CPU/GPU 数学、完整 wheel 和实际请求结果另记。


Triton GPU 数学夹具首先复现原生 align 的 HCU 320-thread launch 超过 256-thread bounds，随后非法访存。固定 DAS 的同一 align 明确记录并修复该问题。本轮补齐其 HCU Triton count/prefix/sort、padding sentinel 和无效专家映射，公共入口 AROUND 在其他平台委托原实现；当前合同 369 项。原故障证据 logs/w4a8-triton-current-dev24.log 保留，修复后数学测试另记，不把崩溃夹具称为通过。


实际数学检查：修复公共 HCU align 后，随机 INT4 两层 MoE/SiLU/路由/缩放的 GPU 输出与显式参考完全相同，最大绝对误差 0。另以 E=256、N=512、K=6144、top-k=8 的 GLM 四卡真实权重形状执行 warmup、Graph capture 与 replay，零输入/零权重结果正确；共 3 项检查通过。旧夹具的高/低 nibble 顺序错误已修正，失败日志保留；不是通过修改数学容差或算子绕过。证据 logs/w4a8-triton-current-dev24-graph.log 与 w4a8-source-alignment.json。完整模型请求仍待复验。

## 23. 2026-10-30 DeepSeek V4 当前 Base/Platform 执行接入

DeepSeek dev17 的旧模型替换仍调用已删除的 get_pp_group。按当前 DAS 拆分迁入 DecoderLayerBase/Platform 与 NonCudaDecoderMixin、模型/输出包装和所需 mHC/transport provider。模型、causal class 继续保留官方类身份，经原公共入口调用当前方法；显式 super owner 适配保证原类与派生类实例均只执行一次 nn.Module 初始化。当前 Model/causal 共 13 个实际不同方法沿 DAS 同步，旧 Decoder 三个方法替换退役为完整 Platform 类替换；注册同步更新所有已导入的 NextN 构造别名。

14 个选择定义还原 provider 导入、super owner/base 和未启用 mxscale 的惰性导入后 AST 与 DAS 相同。8 个 mHC 数据/状态类与官方 AST 完全相同，复用其类型身份，使 target、attention、DSpark 生产/消费一致；仅把 _mix_stats_impl/post 两项 HCU 差异函数接入同一当前实现。当前合同 376 项。固定 DAS 的 AITER in-tree helper 与版本化 CSV/JSON 原样打包，保留其 upstream-first 能力检查；当前容器缺少三项 gfx950 tuning API，沿 DAS 原逻辑记录 skip，不将其称为实际 tuning 验证。

HCU 不执行的 CUDA row-bucket replay/tail boundary 依赖改为首次实际使用时导入；当前 HCU maybe_create 按 DAS 返回 None。显式 dense-mxscale 仍需基线实际支持的接口，本轮保留默认 false 并仅在原 opt-in 分支导入该接口，不宣称 gfx950/CUDA tail/CP/PD 已迁移验证。没有为缺失算子制造成功结果或替代数学。

版本 dev25；真实原模型类入口、PP group/五 stream、Platform 状态、NextN 已导入别名和 mHC 状态身份的 CPU 回归、完整 wheel 与 GPU 模型复验分别记录。静态/构造通过不代表 DeepSeek 精度通过。证据 v4-execution-source-audit.json、v4-shared-mhc-types.json、v4-execution-env-specs.json 与 logs/v4-execution-current-dev25.log。


GLM dev24 已越过 Triton MoE，在 EAGLE draft forward 的 CP 资格 helper 读取旧 enable_prefill_context_parallel 字段时失败。固定 DAS parallel_hook 为 legacy CP 从 enable_prefill_cp 与当前 attention backend 派生该 alias；官方 bag 没有此字段。共享 HCU helper 现优先使用真实 legacy 字段，缺失时按同一派生条件读取当前 getter，排除 dsa/dsv4 的 MLA legacy alias；不伪造全局 bag 属性。NextN 公共 forward 回归改为真实经过 CP-disabled helper 分支，避免仅在 enable_cp_v2=True 时跳过原失败点。完整 CP/PD 仍未验证。

## 24. 2026-10-30 EAGLE 当前 KV plan 返回与 hidden-state 规则同步

GLM dev25 完成 target/draft Graph 后，在服务 warmup 的 EAGLE draft 调用触发旧包装器按二值解包当前三值返回。prepare_for_draft 已返回 (ForwardBatch, graph_flag, kv_loc_plan)，runner 位于第五个 positional 参数。包装器现只读取第一项补齐 DAS 的 mHC pre-norm 条件，原返回容器、KV plan 和其他项按身份完整返回；普通 GLM 不改 hidden-state 标志。

同时核对固定 DAS eagle_utils.eagle_prepare_for_verify 与官方均明确 return_hidden_states_before_norm=False；旧 HCU wrapper 强行改 True 已不对应当前路径，退役此一项 Hook 与函数。当前合同 375 项；不通过丢弃 plan、关闭 EAGLE 或伪造返回规避异常。版本 dev26，公共入口/调用者三值消费与完整服务复验另记。


DeepSeek dev25 已加载目标权重并进入 mHC Graph，旧 hcu_sglang.kernels.mhc 仍从官方 DSA utils 导入被移除的 round-robin CP 名称。改为同仓已接入的当前 HCU compat provider，保留当前拓扑与 round-robin/interleave 判定；实际模块导入和服务路径继续复验，不设置虚构 CP 状态。

## 25. 2026-10-30 GLM 精度失败与当前 DAS mHC 同步

GLM-5 INT4 dev26 attempt17 已完成 target/draft Graph、warmup 和 HTTP 请求，但算术、中文、英文三个响应均乱码且耗尽 1024 token，不能作为精度通过。主动终止余下请求，控制器清理完成，残余 PID 为空且端口关闭。实际 checkpoint 的三个 dense INT8 投影与 FP32 INT8 matmul/scale 参考误差均为 0；这只排查局部量化投影，未证明模型正确。

DeepSeek V4 dev26 attempt4 在 Graph 捕获进入旧 mHC 的 TileLang Allocate monkey patch 时失败。对照固定 DAS，当前 mHC 已删除该旧补丁并增加 HCU 小 batch split-K、AITER 物理维度压紧等路径。本次整体同步其 mHC 模块，只将 is_hcu 导入改为 plugin 检测；模块内所有顶层函数/类 AST 与固定 DAS 一致。没有通过关闭 Graph、mHC 或 AITER 避开问题。

真实 HCU 运行 8 token、4×7168 hidden 的 mHC pre，对照同源 _mhc_pre_torch：post 最大绝对误差 1.85e-6，comb 8.8e-7，BF16 activation 0.0078125；Graph 捕获及 replay 通过。初次探针误用了另一历史 mhc_pre_torch 辅助函数（其 post 额外乘 2），失败原日志保留，改用当前实际 dispatch 的 _mhc_pre_torch；生产公式未为测试修改。dev27 整体服务复验另记，GLM 精度仍待修复。Qwen3 dev26 attempt7 六请求再次通过；MiniMax dev26 attempt3 正在复验。

## 26. 2026-10-30 MLA INT8 反量化的 None block size 分支同步

继续排查 GLM 精度时发现旧 post_load_weights 通过 hasattr(weight_block_size) 判断 block quantization。当前 QuantizationConfig 即使 channel INT8 也显式带 weight_block_size=None，因此旧分支跳过 channel scale，w_kc/w_vc 留为原始 INT8。固定 DAS 已改为检查实际 block size 是否非 None；plugin 同步其整个 INT8 分支，分支 AST 与 DAS 一致。该修改保留 MLA 路径和 checkpoint scale，不改变注意力/量化后端配置。

新增以显式 None block size、非均匀 channel scale 的实际 post_load_weights 数值回归，检查拆分出的 w_kc/w_vc 是 BF16 且精确对应反量化值；旧 wheel 失败证据与修复后结果均单独保存。该局部问题修复不等同 GLM 整体精度通过，后续完整六请求复验另记。另补充 256 experts、高位 expert IDs 的真实 HCU W4A8 单算子数值参考通过，最大误差 0。

DeepSeek dev27 attempt5 已越过 mHC 并完成 decode Graph，在 verify Graph 遇到 HIP metadata factory 返回 None，而 HCU FlashMLA consumer 要求 FlashMLASchedMeta 对象。沿用当前 DAS HCU backend 的对象 metadata 路径，新增 HIP factory 的 HCU 定向 Hook，通过同源 get_flashmla_op 访问当前 flash_mla_interface provider；非 HCU 仍调用原 factory。合同增至 376 项。

## 27. 2026-10-30 GLM 六请求通过与 RowParallelLinear 融合能力接入

GLM dev28 attempt18-admission-retry2 使用原 EAGLE、FP8 KV、正常 Graph 和显式 DAS Triton W4A8，在四卡上严格 6/6 通过：323、北京、Paris、4、10、重复 323，均 finish_reason=stop。目标 verify batch 1–8、draft decode/extend Graph 已捕获；04:28:33 清理完成，无残余 PID、端口关闭。一次准入资源竞争只造成未启动的 pending_resources，随后控制器自动重试，没有同时启动两个模型。

DeepSeek dev28 attempt6 越过 HCU FlashMLA metadata 后，在共享 MLP 查询 RowParallelLinear.supports_fused_silu_mul_fp8_quant_input 时失败。同步 DAS 两个能力谓词与完整 Row forward 的融合输入消费实现，仅对 HCU 新量化参数的调用进入该实现；普通调用和非 HCU 交还原 forward。保留环境门控、kernel 可用性和真实 consumer trait 的组合判断，不用恒 False 或关闭配方开关代替缺失 API。新增 FP8 EnvBool 的默认 false 与 DAS 相同；DPSKV4 原有独立融合开关保持原值。

三个方法只限定真实官方全局 provider；能力方法挂到当前真实 RowParallelLinear，新增已有 forward 入口的 AROUND Hook，合同 377 项。回归覆盖能力/consumer 配对、FP8 q/scale/输出 dtype 经当前 Row forward 到 quant method、默认及非 HCU 委托。外部 LightOp 数学在该 CPU 数据流夹具中替换为可观察 quant producer，不能作为其 GPU 精度证据；完整 DeepSeek 服务继续复验。

## 28. 2026-10-30 共享 MLP 构造门控同步

进一步核对已失败的共享 MLP 路径，官方 DeepseekV2MLP.__init__ 将所有 HIP 都设为 use_fused_clamp_act_mul=True，而固定 DAS 明确是 HIP 且非 HCU 且 SGLANG_OPT_USE_FUSED_CLAMP_ACT_MUL。此前只同步 forward 留下初始化分支差异。本次迁入完整当前构造器；除该赋值外两个输入构造器 AST 完全一致，其他初始化、parallel_group 和权重处理没有省略。按 DAS 添加默认 True 的环境字段，HCU 选择 DAS 的后续 clamp 路径；这是源码中的平台资格判断，不是为跑通测试关闭配方开关。

新增构造器 Hook 后合同为 378 项。真实公共类构造回归保留 nn.Module 初始化，替换昂贵线性分配，验证 HCU 与普通 HIP/环境门控；完整 DeepSeek 服务仍需新版本复验。dev29 的第七轮在资源等待中出现多次准入竞争，实际 run_id 以控制器记录为准；未启动的 pending_resources 不计为模型失败或请求结果。

## 29. 2026-10-30 BW1100 DeepEP ABI、buffer 与量化路径同步

DeepSeek dev29 第七轮实际失败在普通 HIP AITER fused-clamp 包不存在，dev30 已按 DAS HCU 构造门控越过该路径。dev30 attempt8 随后在 Graph 的 low-latency dispatch 被 vendor Buffer 拒绝 use_fp8 参数。固定 DAS 已通过签名检测区分 upstream use_fp8 与 BW1100 quant_type/topk_weight ABI；本次迁入完整 DeepEP dispatcher 模块，保留 normal/low-latency、量化生产/消费、combine、event/overlap，以及 target/draft 共用最大布局 buffer 的管理。

仅替换现有 DeepEPBuffer 和 DeepEPDispatcher 两个公开类，其他 AST 相同的数据/配置类型复用官方身份；合同 380 项。HCU DeepGEMM 能力探测复用 DAS 的真实 deepgemm 导入检查，量化布局另要求实际选择 deep_gemm runner，不能因包已安装就切换 AITER/SlimQuant 的 dispatch。没有安装/替换系统 DeepEP，也不通过关闭 DeepEP、speculative 或 Graph 绕过错误。

CPU 回归覆盖 quant_type=0/2/3 与权重/scale contract、不支持 NVFP4 的明确拒绝、backend/能力/SlimQuant 选择、真实公共 dispatcher 模式选择及 target/draft 最大 buffer 布局保留。通信 kernel 由可观察 vendor stub 替代，不能当作真实多卡通信或模型验收；新 wheel 的四卡完整服务继续验证。


同一失败边界继续核对量化消费，当前官方 DeepEPMoE 没有 DAS 的 HCU channel-FP8 grouped GEMM 分支。因此同步其已验证的 legacy channel-FP8 producer/consumer：完整权重 pack/非持久 buffer、普通及 masked FP8 两层 GEMM、LightOp scatter/gather 资格与 fallback、SiLU clamp/FP8 quant。只在 DAS 同一 channel-FP8/SILU/非 OAI/无 local-shared 条件进入；其他模式委托原实现。旧配置类没有新增 local_shared_experts_without_dispatch 字段时读取其 DAS 默认 False，不声明该 opt-in 功能已支持。构造器同步 legacy flags 和 dispatcher quant config；checkpoint 后按 Source 打包并删除 canonical 权重，保留逻辑 shape。两个现有公开方法新增 AROUND 后合同 382 项；量化后处理复用已有合同目标。运行没有动态导入 DAS 源码。

## 30. 2026-10-30 V4 输出收尾 helper 接到真实模型类

DeepSeek dev31 attempt9 的 target verify Graph 已通过完整 body，包括 DeepEP vendor dispatch、HCU channel-FP8 grouped GEMM 和全部模型层，随后在输出收尾缺少 logits_from_body_output。该 helper 已位于当前 DAS 的 owned causal class，但实际公共 causal 类保留官方身份，原注册只覆盖已有方法交集，遗漏了这个新增方法。

HCU 注册现在把同一已迁入的 helper 接到真实官方 causal 类；不更改数学，不伪造隐藏状态。继续保留既有 forward Hook 及 Native 类身份，合同仍为 382 项；只补齐当前基线缺少的能力方法。公共 forward 回归直接走真实模型类到 helper/logits_processor，检查 pre-hc-head、aux/PD capture 和输出 flatten 的实际消费。Source body/helper 已成对同步，Graph 和完整请求须在新 wheel 复验。

## 31. 2026-10-30 normal DeepEP FP8 全局依赖补齐

DeepSeek dev32 attempt10 进入 normal channel-FP8 分发消费，在已迁入的 contiguous GEMM helper 读 get_offloader 时因遗漏导入失败。该 helper 的源模块额外导入当前官方已有的 get_offloader 与 copy_list_to_gpu_no_ce；本次显式接入这两个真实 provider，保留禁用 copy-engine 时的无 copy-engine GPU 计数搬移分支。

新增 normal 两层 GEMM/quant/scatter/gather 的 CPU 数据流回归，使用真实公共 DTO，检查 count、scale、clamp 与输出传递；外部 GPU kernels 由可观察 stub 代替。同步检查该 owned 模块顶层函数的实际引用全局名均存在，防止只跑 masked 路径而遗漏 normal 依赖。没有改精度容差或关闭 normal dispatch；合同仍 382 项，完整服务由下一轮验证。


全局依赖检查又发现 Native ep_moe module 不再 re-export scatter/gather/m_indices；相同检查在缺失时明确失败。对照 Source，scatter/gather 的现行 HCU-aware 实现亦不同于官方，不能只导入名字相似的旧实现。本次把两个入口及其实际函数依赖闭包和 m_indices fallback 放入 owned kernel provider，保留 Source LightOp 资格与原 Triton fallback；只适配 is_hcu 检测导入，所有选中函数 AST 与 Source 同一实现。未改变其他官方调用方。

## 32. 2026-10-30 V4 target/draft backend 分流同步

继续审查启动失败链，发现固定 DAS attention_registry.create_dsv4_backend 明确用 HIP 且非 HCU 才进入 DeepseekV4HipRadixBackend；当前插件遗漏该分流，前述 DeepSeek 轮次实际仍选普通 HIP 类。现在对现有 registry 函数及 draft decode/prefill 工厂增加 HCU AROUND，按 DAS 同一路径进入 V4 factory；其他平台委托原调用。registry dict 保存 callable，不会仅因模块函数被 Hook 自动更新，因此一并接好实际 ATTENTION_BACKENDS['dsv4'] 字典入口，保留原 callable 供非 HCU 委托。

当前配方 auto/FlashMLA 选择已迁入的 owned HCU Base/Multistep backend；不通过临时篡改 is_hip 或平台全局值选路。显式 TRTLLM/Mega、CP/PD 等模式不因本轮而新增验收。合同增至 385 项。前述普通 HIP 分支没有模型通过记录；正确 HCU backend 的真实服务验证另记。

dev33 已完成 6 项 normal/masked/全局依赖/权重别名/DTO/HCU kernel 回归；11 个 owned EP kernel 定义 AST 与固定 DAS 相同。真实 HCU byte scatter、per-token scale 与路由加权 gather 最大绝对误差 0，m_indices 有效行及 -1 padding 精确匹配。这些属于单算子证据，不能代替完整模型请求。

## 33. 2026-10-30 四卡代表模型验证结果

当前运行时代码提交 `86f3f70ac72e673627c816f546eaf647cb045cf9`，wheel `0.3.2.dev34`，SHA256 `45b88367f9fc3f9462eed414d24279fb7176dacb5f9c588733f0e7e0479d415a`；本地提交，未推送。当前源码/安装文件逐字节核验和 385 项完整 Hook 合同均通过，各次实际版本见下表。

| 模型 | 实际 wheel 目录 | 运行 ID | 物理 GPU | 严格请求 |
|---|---|---|---|---|
| qwen3-30b-instruct-tp4 | `site-dev26` | `qwen3-30b-instruct-tp4-aiter-attempt7` | 0,1,6,7 | 6/6 |
| minimax-m25-bf16-tp4 | `site-dev26` | `minimax-m25-bf16-tp4-attempt3` | 0,1,6,7 | 6/6 |
| glm5-int4-tp4 | `site-dev28` | `glm5-int4-tp4-attempt18-admission-retry2` | 0,1,6,7 | 6/6 |
| deepseekv4-fp8-tp4 | `site-dev34` | `deepseekv4-fp8-tp4-attempt12` | 0,1,6,7 | 6/6 |

六个确定性检查为算术、中文、英文、短阅读、代码结果与重复请求；最终答案及 stop 条件均严格通过。各服务结束后自有进程清理为空，端口关闭。Graph、EAGLE/DSPARK 与 DeepEP 按对应配方记录，具体捕获模式和参数偏移见原始运行配置与日志。

环境仍为固定官方 SGLang 0.5.22 加隔离插件，既有容器提供 DAS 0.5.18 依赖，cookbook 0.5.18 仅为配方参考。用户授权四卡 profile；本表不外推完整八卡配方、数据集精度分数、长上下文或性能基准。全量迁移台账保持独立状态。

仅在启动/精度失败处对照固定 DAS 代码，未启动 DAS 对照服务。原始证据在指定 test-model 目录的 README.md、results.json、runs/、artifact-dev 各版本 JSON 和 wheel-hooks-dev 各版本 JSON；历史失败完整保留。

DeepSeek dev34 attempt12 的日志明确记录 HCU owned DeepseekV4AttnBackend，用于 target/draft 的 backend 选择已与 DAS HCU 分流对应。attempt11 在静态比例 0.8 下因 target/draft 权重合计超过 KV 预算停止，框架测得最低可用比例 0.8175；成功轮使用 0.9，并保留 16K KV、8K context、Graph max batch 8、每卡至少 132000 MiB 空闲的准入检查。该变化为测试 profile 的显式容量调整，未更改运行时代码或隐藏原内存检查。

## 34. 2026-10-30 HCU 安装命名与文档一致性核验

当前 `pyproject.toml` 的 distribution、Python 包和 general plugin 入口分别为 `hcu-sglang-plugin`、`hcu_sglang`、`hcu`。本轮从干净代码提交构建 `hcu_sglang_plugin-0.3.2.dev34-py3-none-any.whl`，检查 wheel metadata/entry_points 和包目录，再隔离安装并在源码目录外检查 import 来源。唯一 general plugin 入口为 `hcu = hcu_sglang:register`，console script 为 `hcu-sglang-serve`；无旧包目录或 `hcu` 入口。该 wheel 的 SHA256 为 `79150616eb8c55e984317554addeed9faad82af42fe811d2f76516e8fc012675`，与前文远程历史构建分别记录，不因版本号相同替换历史文件哈希。

修正安装指南的 wheel/editable、metadata、严格加载合同、服务及卸载示例，当前合同显式指定 `--plugin-package hcu_sglang` 并先初始化 AITER 兼容。同步配置、回溯、排障、README 和相关 Skill 中仍将当前实现称作旧包的说明；旧版本实施记录、来源清单、兼容工具接口和历史文件名保持其来源语义。硬件接口 `is_hcu()` 不属于插件入口改名范围。

本轮仓库 72 项离线测试、L0 的 333 个 Python 文件/8 个 Skill、全部 Skill 元信息及 Markdown 链接/标题锚点检查通过。网页从原始文档重新构建，不直接修改生成 HTML。检查证据保存在本次 ignored artifacts 中。本轮未修改运行时行为、未在本机执行 HCU Hook/模型服务；wheel 名称与入口检查不作为模型精度或设备可用性验收。

## 35. 2026-11-01 RMSNorm/Gemma HCU 分支继续迁移

继续执行 plugin-v3-1028，输入仍为固定 official `baa9befa` 与 DAS `15c7204b`。从当前干净 `b97b4d9` 开始，本轮只做本地提交，不推送。dev34 的八卡 Qwen3-235B 验证与本轮 dev35 代码验收分别记录，不把隔离旧 wheel 的结果归给新代码。

原插件 RMSNorm 在 residual 非空时直接调用 native；当前 DAS 明确进入 LightOp `fused_add_rms_norm_opt`，保留实际 residual 原地更新和 post-residual 的独立张量语义。本轮迁入该完整 HIP 方法，并将当前 Gemma HIP 方法、显式 `forward_with_lightop_fp8_quant` 与其 vLLM ABI 辅助函数放入 owned `norm_current`；只限定官方 provider 全局名、函数命名和惰性 vendor 导入。AITER Gemma 优先级、`1 + weight` 缓冲、batch-invariant、FP8 门控与 `(quantized, residual)` 返回结构沿用来源。非 HCU 保留传入 original，不改官方文件，也不动态导入 DAS。

[源码对照](hcu-sglang-development.md#历史-json-快照)（`hcu-norm-source-alignment.json`） 固定源文件哈希和五个选择定义；去除导入限定并规范化抽取 docstring 缩进后 AST 相同。这只证明选择的方法，不能证明整个 layernorm 文件已迁移。全文件处置保持 in_progress；AR/RMSNorm 的通信组与 per-group quant 融合仍需独立处理。

`tests/runtime/norm_current.py` 在指定容器内通过六项 CPU 回归，真实公共 RMSNorm/Gemma 类、注册与 forward 调用保留，昂贵 LightOp kernel 用可观察数学夹具替代。覆盖非连续输入、融合分支命中、residual/post-residual 存储语义、batch-invariant、Gemma 权重加载/AITER 优先级、FP8 carrier/gate 与非 HCU original 委托；不据此宣称 vendor GPU 数学、Graph 或模型精度通过。当前准备版本 dev35、Hook 合同 386 项，产物和完整运行验证结果待本次执行后补充。原始日志位于本次 ignored `artifacts/plugin-v3-1028-20261101/`。


本阶段实际结果：dev35 wheel SHA256 `4dabfc1963c943b625de840a17c70a83eb2aab29e5ceb840ac11cd043839e35a`；575 个包文件与源码及隔离安装相同。隔离 wheel 严格加载 386/386 项 Hook、3 个具名 seam，软错误和旧包导入为空；72 项仓库 unittest、6 项公共入口 CPU 回归通过。

`tests/runtime/norm_current_gpu.py` 使用真实 gfx938/HCU LightOp 数学，3 项通过：BF16/FP16 RMSNorm 的 plain/residual/post-residual/非连续输入与 FP32 参考，BF16 Gemma plain/fused 与 `1 + weight` 参考，以及 RMSNorm residual 的 Graph 捕获和两次不同输入 replay。未使用 kernel stub、未修改生产容差；这些仅证明该测试的形状和分支，不代表 Gemma FP8 opt-in 分支或整模型已通过。完整 MiniMax 八卡服务已用该隔离 dev35 wheel 启动，结果在后续验证记录补充。

## 36. 2026-11-01 原八卡配方 Qwen3-235B 验收

在八张卡均空闲时，按 cookbook 固定 `e91f05fe` 的 Qwen3-235B-A22B-Instruct-2507 / BW1100 / SGLang0.5.18 单机配方运行固定官方0.5.22 + 隔离 plugin dev34。保留 TP8、fa3、page64、内存比例0.85、默认上下文/KV 和默认 Graph，没有沿用四卡 bounded smoke 的容量限制；只使用本地只读权重、任务 loopback 端口及既有离线 Rust/缓存配置。

运行 `qwen3-235b-instruct-tp8-dev34-20261101-attempt1` 于 00:02:07–00:13:40 完成。六个算术/中文/英文/阅读/代码/重复请求的 HTTP200、模型身份、完整最终答案和正常 stop 均通过；默认 Graph 捕获从 batch512 开始。结束时自有进程为空、端口关闭。对提交 `b97b4d9`、实际 dev34 wheel 与隔离安装的574个包文件重新逐字节核验一致。

[可提交验收摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-qwen3-validation-20261101.json`） 保存 cookbook 来源、版本、wheel与响应哈希、每个实际最终答案和清理结果；原始日志/请求/响应继续留在用户指定 test-model/runs 目录。该结果证明 dev34 的八卡短请求配方，不是数据集精度、吞吐、全 DAS 功能等价，也不归因于本阶段 dev35。

## 37. 2026-11-01 归一化通信组与量化 handoff

dev36 将固定 DAS 的两个 communication_op fused norm 入口和两个 layernorm AR/norm/quant helper 接入本仓，按 `tp` / `attn_tp` / `moe_tp` 选择真实 coordinator，不能把 MoE EP 大小当作 MoE TP 大小，也不能在选定 attention/MoE 组后退回全 TP 组。计算 AITER 开关与 communicator 融合开关分开，融合拒绝时仅在来源规定的组上归约一次；post-residual 只加一次。

量化路径保留 FP8 tensor/scale 和可选 BF16 side output、bpreshuffle 的 transpose/materialize 区别及 `None` 拒绝约定。[四个 helper 对照](hcu-sglang-development.md#历史-json-快照)（`hcu-norm-collectives-source-alignment.json`）还原 provider 限定后 AST 与 DAS 相同；两个真实 GroupCoordinator 方法 AST 在官方与 DAS 中完全相同，直接复用。注册通过官方 Hook 传播机制更新已导入别名，未修改两个输入仓。

| 检查 | 本阶段实际结果 |
|---|---|
| CPU 公共入口 | `tests/runtime/norm_collectives.py` 9 项通过：三种组、默认 TP、拒绝未知组、预导入 alias、独立通信策略、decline fallback、Gemma 权重、FP8/BF16 carrier 和 scale 布局 |
| 真实八进程 HCU | `tests/runtime/norm_collectives_gpu.py` 使用真实 AITER IPC fused kernel，TP8 / attention TP4 / MoE TP2 各 rank 的 BF16 输出及 residual 与 RCCL+FP32 参考通过 |
| gfx938 平台拒绝 | 未把 gfx95-only per-group fused quant 伪装成功，三个组均保留 native guard 的 None 返回 |
| 隔离 wheel | SHA256 `7df1214bbc0f347c8b3cacf83ecaf240de8fbf7d360cc11a6e02b28d1383b6b9`；576 个包文件与源码、隔离安装一致 |
| 完整加载合同 | 390/390 Hook、3 个具名 seam，通过；软错误、未应用与旧包导入为空 |

八进程探针显式创建 Gloo 元数据组、NCCL 参考组和真实 AITER communicator，并将其提供给原生 GroupCoordinator 方法；只注入组 getter，未替换通信或归一化数学。该探针不验收完整 coordinator 构造、CLI 后端选择、Fabric 或 collective Graph replay；细节与原始日志哈希见[可提交摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-norm-collectives-gpu-validation-20261101.json`）。模型验收按各自实际 wheel 版本记录。

执行台账新增两个已实现文件处置，当前为3007 pending、3 implemented；这是逐文件审阅状态，不是全量完成率。parallel_state 的新后端选择/strict-Fabric、CP/PD，以及其他领域仍需分阶段继续，不能由本阶段或八卡代表模型结果关闭这些范围。

## 38. 2026-11-01 MiniMax-M2.5 BF16 八卡验收

隔离 dev35（代码 `fefb5fc`）按同一 cookbook 固定版本的完整 MiniMax BF16 单机八卡配方运行，保留 TP8、fa3、FP8 KV、内存比例0.9、131072上下文、prefill16384、max-running-requests512 和默认 Graph。运行 `minimax-m25-bf16-tp8-dev35-20261101-attempt1` 于00:16:58–00:27:48完成，六请求完整最终答案及 stop 全通过，自有进程为空、端口关闭。

对实际 dev35 wheel、隔离安装与已提交源码的575个包文件再次逐字节核验一致。[验收摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-minimax-validation-20261101.json`）保留来源配方、版本、wheel/响应哈希、实际答案和清理结果。该结果属于dev35；不冒充dev36通信组路径、数据集精度、长上下文压力或性能基准。

## 39. 2026-11-01 原配方后端参数与组构造链路

原GLM八卡配方的 `--custom-all-reduce-backend aiter` 在dev36真实parser上被拒绝（exit2）。本阶段新增完整四选项CLI、worker继承与distributed bootstrap桥接，保留来源的disable优先级和legacy auto提升规则。官方0.5.22原始/comm记录无该字段，采用本仓进程配置，不写入冻结字段；程序化调用方式及单进程边界见[配置说明](configuration.md#当前-custom-all-reduce-参数)。同时迁入来源的两个deprecated NSA别名，保持canonical字段、选择及默认语义。

新增 `distributed_backend` owned实现，提取DAS完整GroupCoordinator构造器、world/model group工厂和backend setter/getter；限定现有官方provider后，[五个选择定义](hcu-sglang-development.md#历史-json-快照)（`hcu-custom-allreduce-source-alignment.json`）规范化AST相同。保留原类身份、原有comm属性和进程组创建路径；只在HCU调用owned实现，其他平台保留传入original。strict-Fabric构造失败保留原异常，disabled明确拒绝，QuickAllReduce沿独立门控及transport排除规则。

`tests/runtime/distributed_backend.py` 在指定容器内13项通过，包含真实CLI/原始record、冷子进程继承、公共bootstrap与GroupCoordinator构造，昂贵进程组/communicator分配使用可观察夹具；不将CPU构造测试记为Fabric硬件通过。首次导入发现官方parallel_state没有re-export来源所用get_bool_env_var，已显式引用真实utils provider，原始失败保留。完整GLM八卡参数解析已通过，未删除 `aiter`、EAGLE或旧NSA选项。

当前dev37 wheel SHA256 `c991a6918ea1fff44bada82348477bab133ef0b0366837043cee2504e7e05498`；577个包文件与源码及隔离安装一致，394/394 Hook与3个具名seam严格加载通过，无软错误、未应用或旧包导入。原八卡GLM INT4/EAGLE/FP8 KV配方正在指定容器复验，结果单独补充；没有运行共享节点的全局NUMA sysctl。parallel_state其余CP/拓扑、新namespace字段等仍按台账保持未完，不由选择定义对照推定完整文件或全量DAS完成。

## 40. 2026-11-01 GLM-5 INT4 原八卡配方通过

代码提交 `89ac151`（dev37）完成后，运行 `glm5-int4-tp8-dev37-20261101-attempt1` 于00:59:54–01:08:40通过六个严格请求：HTTP200、正确模型标识、完整最终答案及stop均通过。TP8/DP1/EP1、旧NSA参数、INT4 quantization、EAGLE、FP8 KV、内存比例0.85和默认Graph保留，无四卡smoke的context/KV-token/graph-size覆盖。只使用本地模型路径、loopback端口及任务Rust/cache；未修改主机NUMA sysctl。

八个rank日志均确认 `custom_all_reduce_backend=aiter requested_transport=ipc selected_transport=ipc disabled=False`；W4A8按原配方`auto`解析为LightOp，未设置TPMOE backend覆盖。权重后处理与target/draft Graph完成，服务结束时自有PID为空、端口关闭。实际wheel/隔离安装与已提交577个包文件逐字节一致，[验收摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-glm5-validation-20261101.json`）保存版本、wheel/响应哈希、实际答案与后端日志计数。该结果不外推Fabric、数据集精度或性能。

## 41. 2026-11-01 已有SiLU迁移实现复核

固定DAS的activation模块只有 `SiluAndMul.forward_cuda` Python定义与官方不同，另在module级 `_use_aiter` 选择上排除HCU。已有 `norm_activation` 的CUDA AROUND与构造AFTER分别覆盖同一LightOp调用和HCU构造选择；确定性/native及其他平台的effective original保留。其他Python定义AST与官方相同。具体审阅及环境边界见[记录](hcu-sglang-development.md#历史-json-快照)（`hcu-activation-source-review-20261101.json`）。

`tests/runtime/activation_hcu.py` 四项通过：HCU/AITER/确定性构造选择，真实公开LightOp BF16/FP16输出在三组形状下与FP32 SiLU×up参考比较，变更gate/up输入的Graph replay，以及非HCUoriginal委托。未改运行时数学或容差；本阶段补充验证与处置，不重建相同版本wheel。

最初强制全局 `SGLANG_USE_AITER=1` 的探针在进入activation之前，因容器缺少 `aiter.gemm_a8w8_blockscale` 从fp8_utils导入失败；固定DAS与官方该开启表达式/导入要求相同。改用原配方默认的global-AITER关闭状态验证HCU LightOp，原失败日志保留；不增加假函数或篡改来源门控，全局AITER部署仍未验收。现有代表模型配方未要求该全局开关。

DeepSeek-V4原八卡配方正在复验，保留TP8/DP8/EP8、DSPARK、DeepEP、context32768/graph32和64线程加载。两个网卡xxx替换为实际ens14f0，显式DeepEP配置文件由已安装vendor对8-rank的推荐dispatch/combine参数导出，并构造真实Config对象验证字段；不是未执行的调优或性能结论。该任务仍有全量文件审阅与其他功能待完成。

## 42. 2026-11-01 DeepSeek-V4 原八卡 DSPARK/DeepEP 验收

隔离dev37（代码 `89ac151`）运行 `deepseekv4-fp8-tp8-dev37-20261101-attempt1`，01:26:47–01:34:20完成六请求，HTTP200、模型身份、完整最终答案及stop全部通过。保留TP8/DP8/EP8、DSPARK、多stream/fast sampling、DeepEP、max-running32、context32768、graph32、内存0.90与64线程加载，无四卡KV/context/graph缩减。八个DP rank日志明确命中owned HCU `DeepseekV4AttnBackend`；结束时自有进程为空、端口关闭。

[验收摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-validation-20261101.json`）保存实际版本、577个包文件的源码/wheel/隔离安装核验、响应哈希、backend路径/ranks与DeepEP配置哈希。网卡/模型/draft/profiler/config占位符按节点实际值填写；EP8推荐配置由实际vendor工厂读取数值、重建真实Config验证，并保留独立来源哈希。首个日志提取器误用了历史文案，修正为实际logger的 `HCU dsv4 backend:` 模式后确认八rank；没有重启或改变通过的模型运行来修补证据。

## 43. 2026-11-01 完整 Python AST 文件复核

新增 `audit_python_equivalence.py`，在固定两输入的Git tree/blob上完整比对，保留docstring/type comments/常量/imports/guards/decorators；权限、对象、shebang、输出重叠、dirty输入和并发决定修改均保护。5项实际Git fixture检查通过，包含布尔/数值差异、docstring、无效语法、入口/权限、人工决定保留与陈旧stat cache下index字节保留。

本次覆盖1534个双边changed Python路径：26个AST一致、1494个实际不同、14个权限/类型漂移。复核26个匹配路径后，只添加尚无人工记录的代码级upstream，未覆盖已有决定；两个来源HEAD/index不变。当前台账为2975 pending、26代码级upstream、5 in_progress、4 implemented。报告保留[逐文件哈希](../plans/sglang-das-python-equivalence.json)，代码复用与行为/依赖验收分层，不据此计算全功能完成率。

## 44. 2026-11-01 线性层预量化消费者与 fused RMS 契约

从干净 `5082647` 继续M1。旧Column包装器没有 `input_quant_args`，公开接口在dev37上实际抛出unexpected keyword；同时它对rms_weight执行单独归一化，没有来源的fused INT8消费契约。本阶段迁入来源的Replicated/Column/Merged forward、Column构造/partition、Merged构造和attn_cp group解析，保留原生类身份和准确super owner。完整helper原已在本仓与来源同字节，本次将其接入实际投影消费者，不另写数学fallback。

显式q/scale不依赖legacy开关，unsupported消费者按来源拒绝；保留epsilon、input/residual更新、Merged fused分支五值返回和普通分支二值返回。Row的来源位置参数及swiglu_limit也接到已有完整forward；普通无扩展调用与旧位置output buffer保持effective original。[对照记录](hcu-sglang-development.md#历史-json-快照)（`hcu-linear-source-alignment.json`）包含全部十个改变/新增Python定义，去除provider限定和显式owner适配后AST相同，helper原始字节相同。

| 验证 | 实际结果与范围 |
|---|---|
| dev37公开反例 | 显式prequantized Column调用因unexpected keyword失败，原始日志保留 |
| CPU公共入口 | 10项通过，含真实SlimQuant消费者与post-load布局、三个投影类、QKV继承、fused输入/residual/epsilon、Merged五值、组选择、unsupported拒绝及Row扩展/旧调用 |
| 真实HCU | dev38隔离wheel的3项通过：BF16/FP16 fused RMS量化/残差；真实INT8 GEMM的显式/融合路径及Graph replay；packed MLA q/kv归一化、q量化与rope保持 |
| dev38产物 | 578包文件源码/wheel/site一致，SHA256 `a8c00a5ff80060ecbafa8484a2f3c27e1b1bb20fd0967a2af0aa71da56ecab5c`；400/400 Hook与3 seam通过 |
| dev39产物 | 只补Row调用路由；578包文件一致，SHA256 `b6c970994bb9b8c490174ea28bf5eae8fbc25520bf80b5860341dca62d2be2f3`；400/400 Hook与3 seam通过；模型结果按实际版本另记 |

GPU首轮的投影夹具错放为连续[K,N]，不符合真实INT8 GEMM post-load[N,K]布局。修正为调用真实process_weights_after_loading后通过；生产GEMM、来源数学和测试容差未改。3项GPU结果只归于dev38；dev39路由由新增CPU公共调用覆盖，不将其冒充新GPU执行。原始证据在ignored `artifacts/plugin-v3-1028-linear-20261101/`。

[原八卡Qwen3-235B的dev38运行摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-qwen3-dev38-validation-20261101.json`）记录六请求通过、进程/端口已清理；该中间artifact来自独立build-dev38源码快照，不伪装为含后续Row修复的dev39提交。后续按最终代码版本记录完整服务。

本阶段77项仓库unittest、L0、执行台账快照、Skill校验及实际宽/窄屏检索/阅读/导航检查通过。GPU证据与源码快照按版本分开，未推送远程。

## 45. 2026-11-01 channel-FP8 fallback 来源路径同步

从本地 `9976f34` 继续M1。旧FP8包装器在HCU上固定调用LightOp channelwise GEMM，未覆盖当前DAS的torch._scaled_mm探测、逐形状BLASLt失败缓存与选择性Triton分支。本阶段迁入完整 `_apply_fallback_scaled_mm`：仅RuntimeError且文本包含blaslt才记录失败形状；其他异常保留原对象抛出。key包含device type/index、M/N/K、输入/输出dtype，成功probe不污染cache。

缩放、bias、padding裁剪、旧tuple输出和dtype转换按来源执行；非HCU保留effective original。[源码证据](hcu-sglang-development.md#历史-json-快照)（`hcu-fp8-fallback-source-alignment.json`）去除两个provider限定后AST相同，复用的Triton wrapper、kernel、scale适配与weak-contiguous检查四个定义在固定DAS/官方中AST完全相同。没有用另一个数学实现伪装fallback成功。

`tests/runtime/fp8_fallback.py` 6项CPU公共入口回归通过：成功路径scale/bias一次、BLASLt失败只probe一次、dtype cache隔离、padding/tuple输出、非BLASLt与非Runtime异常原样传播，以及非HCU委托。昂贵backend在CPU夹具中可观察替代，GPU数学/Graph另测。

当前dev40 wheel SHA256 `577e634e1b5bd16dc761e4c3681124bf9e0308615f6c90e797a038e2f7322b6f`；579个包文件源码/wheel/site逐字节一致；400/400 Hook和3个具名seam严格加载，无软错误或未应用。dev39 MiniMax原八卡服务仍在本次运行中，不因等待另启同一模型；dev40的真实fallback GPU验证按实际执行另补。fp8_utils其他block/mxscale/channelwise opt-in差异仍未关闭，全量目标保持进行中。

## 46. 2026-11-01 FP8 多维scale与Triton混合bias控制适配

dev40真实GPU探针暴露两个固定source/native共有的问题：先把[M,N]恢复为多维输出后再乘per-token[M,1] scale，广播失败；原Triton kernel的 `c += bias` 在FP16输出与FP32 bias下导致分支SSA类型不一致、无法编译。dev41保持二维行域完成scale/bias后最终reshape；在同源优化kernel的bias分支显式执行原store本应完成的输出类型转换，保持先转换accumulator再加bias的顺序，不替换成普通矩阵乘法或外部逐元素fallback。

迁入wrapper时provider别名N又被局部输出维度N遮蔽，已改成native_fp8_kernel，并在方法抽取工具加入replacement/owner目标命名与局部绑定的预检及两个实际执行/拒绝回归。原始失败保留。更新后的[对照](hcu-sglang-development.md#历史-json-快照)（`hcu-fp8-fallback-source-alignment.json`）明确这两处有意适配，反向归一后恢复source fallback、wrapper、kernel AST，不再声称逐字源相同。

源树真实GPU三项已通过：输出[2,14,128]的per-token scale/bias/padding与BF16参考；M17/K6144/N768数值与变更scale的Graph replay；显式预置已知cache条目的M19真实Triton FP16+FP32 bias。当前设备对M17的真实hipBLASLt调用成功，不能冒充自然触发的失败/cache；选择/异常策略由公共CPU夹具独立验证。没有改变测试容差。

当前dev41 wheel SHA256 `774751e5a3c49382e8b3eadc5cbbf9e8801e18260911ad3fc9b4a97b6c195dd2`，580包文件与源码/site一致。隔离wheel的完整CPU/GPU/Hook与门禁结果按本次实际执行补记；尚不据此关闭fp8_utils其余定义、所有profile或全量DAS目标。

## 47. 2026-11-01 dev39 MiniMax 八卡复验

本地代码 `9976f34` 的隔离dev39按原MiniMax BF16八卡配方，02:52:18–03:02:41完成六个严格请求，正常stop、自有进程为空、端口关闭。TP8、FP8 KV、context131072、max-running512与默认Graph保留；578包文件与该已提交代码/实际wheel/site逐字节相同。[运行摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-minimax-dev39-validation-20261101.json`）保存实际artifact与响应哈希，不归到后续dev41。

隔离dev41实际结果：7项CPU、3项GPU数值/Graph与400/400 Hook、3seam通过；79项仓库unittest、L0、Skill和实际宽/窄屏检索/阅读/导航检查通过。真实M17与M32均未进入失败cache，已知cache夹具用真实同源优化Triton kernel。仅当前选择范围通过，其余fp8_utils定义及完整模型dev41验收不由此关闭。

## 48. 2026-11-01 通用HCU FP8 apply与条件量化生产端

本阶段从本地 `1951b9e` 延续FP8迁移。来源 `apply_fp8_linear` 在HCU上使用真实DeepGEMM接口并支持预量化输入，原插件没有该路径接入。现提取完整来源函数并沿公开Hook更新已导入alias；原生 `Fp8LinearMethod.apply` AST与DAS相同，直接保留其consumer，验证从原生类入口到新apply，而非旁路私有quant_method。

[源码/边界证据](hcu-sglang-development.md#历史-json-快照)（`hcu-fp8-linear-source-alignment.json`）明确两处受控buffer适配：直接carrier统一flatten token维及scale、还原输出leading维；output按实际q行数（包括padding）分配，截取真实token行后还原shape。源函数此前用未补齐行数分配GEMM输出，CPU严格backend夹具在q17/out2时失败，未在GPU上冒险执行越界配置。

真实GPU进一步发现Native ROCm scaled_fp8_quant在input8行/scale17行copy失败。DAS对此已有实际条件HCU分支：LightOp dynamic per-token返回未补齐且匹配的q/scale，tensor/static helper只写真实行。已将该分支及三个配套helper原样提取到owned fp8_quant，沿公开函数Hook传播到所有早期alias；保留Source capability/dtype/shape和错误边界，不只修consumer掩盖producer缺口。

12项CPU公共入口/原生consumer回归通过；源树4项真实DeepGEMM GPU检查通过，覆盖BF16/FP16输出、直接多维carrier、dynamic/compressed生产端及变更scale的Graph replay。fixture起初错误要求view后的Tensor对象identity，已改为storage指针与数值/参数检查；并未放宽数学参考或容差。dev42中间artifact保留，其compressed GPU失败指向原生产端；后续隔离dev43结果按实际执行补记。

fp8_utils、quantization/fp8_kernel的其余dispatch、post-load、block/mxscale及其他定义仍分别待审。此处完整选择函数、4个producer定义及未改原生consumer的来源对照不能代表整个文件或全DAS完成。

## 49. 2026-11-01 dev41 DeepSeek-V4 八卡复验

代码 `1951b9e` 的隔离dev41运行 `deepseekv4-fp8-tp8-dev41-20261101-attempt1`，六请求严格通过，自有进程与端口清理。保留原TP8/DP8/EP8、DSPARK/DeepEP、context32768、graph32及64线程加载；580包文件与源码commit/wheel/site逐字节核验。[独立验收摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev41-validation-20261101.json`）保存实际版本、响应/配置哈希和HCU backend日志，不归因于后续dev43。

本阶段隔离dev43实际结果：12项公开CPU producer/consumer、4项真实DeepGEMM GPU数值/Graph、401/401 Hook与3seam通过；582包文件与源码/site一致，wheel SHA256 `d04f014e94f3dbd32784902d9ac53805a0570587baeeea72674185affcd3a2e3`。79项仓库unittest、L0、执行台账、Skill及实际宽/窄屏检索/阅读/导航检查通过。测试只证明选择路径，block/mxscale/post-load及全量目标仍未完。

完整加载复核确认scaled_fp8_quant早已有kernel_compat动态分支hook，因此去重后的总target为401而非推算的402。旧局部wrapper与registration已退役，当前完整Source量化函数统一拥有入口；两层同时注册的中间dev43保留各自artifact，不以重复wrapper作为新能力数。最终dev44重新构建/隔离核验。

最终dev44隔离验证完成：12项CPU、4项真实GPU数值/Graph、401/401 Hook（errors与unapplied均为空）、79项仓库unittest通过；582包文件与源码及实际安装内容一致。wheel SHA256 `d9987b88b04eeb87c2321dc34fcf19414af8cf22fb1cfb5b7a0e633741aea6f7`。完整模型记录仍分别归属于其实际版本，不将dev41的八卡结果归到dev44；其余DAS迁移范围保持未完成。

## 50. 2026-11-01 HCU block-FP8 hipBLASLt 与加载布局

从已提交 `469976c` 继续迁移。DAS在DeepGEMM、FlashInfer、AITER优先项均不命中时，HCU auto选择LightOp hipBLASLt；对应加载方法把checkpoint [N,K]及[N/128,K/128]转成连续[K,N]与[K/128,N/128]。缺任一端都会形成错误consumer契约。

完整提取auto dispatcher、两个量化/GEMM helper和block post-load方法，[来源对照](hcu-sglang-development.md#历史-json-快照)（`hcu-block-fp8-source-alignment.json`）对4个函数归一native命名空间及本仓owner导入后AST相等。post-load Hook仅在实际选择本仓hipBLASLt callable时使用该来源方法；其他backend沿用有效原生方法，不提前替换其权重布局。计算仍使用来源LightOp per-token-group quant与hipblaslt blockwise GEMM，不替换优化backend。

源树7项CPU覆盖优先级、早期alias、原生加载/消费、参数identity、权重与非方形scale转置、backend参数和异常；3项真实GPU覆盖M1/17/32、BF16/FP16与bias、多维token输入，以及改变输入的Graph replay，全部通过。数学参考将真实量化activation与逐128块scale解码，再独立FP32乘积；容差rtol/atol均0.025。保留原来源忽略传入input_scale而重新量化的接口语义；不扩大为tuple carrier支持。

当前源码dev45；完整隔离wheel、全部Hook及仓库门禁验证按本阶段实际执行补记。AITER mxscale生产/消费、其他post-load分支及全量DAS仍未完成，原配方八卡dev44/dev45尚无新模型运行证据。

隔离dev45实际结果：7项CPU、3项真实GPU数值/Graph与403/403 Hook、3seam通过，errors/unapplied为空；583包文件与源码/site逐字节一致，wheel SHA256 `7041059d9e6f164fb6db763c7e91b7a9b93f8feceaa3d8a07c3e5e695b68af8c`。79项仓库unittest、L0、执行台账、Skill元信息及实际宽/窄屏浏览器检索/阅读/导航检查通过。八卡dev45服务另行执行并记录，不用局部测试代替。

## 51. 2026-11-01 dev45 DeepSeek-V4 八卡复验

本地 `0995491` 的隔离dev45从04:02:42至04:08:13完成原TP8/DP8/EP8、DSPARK/DeepEP、Graph32、context32768及64线程加载配方；六项严格请求均以stop结束，自有进程为空、端口关闭。583包文件与提交/wheel/site逐字节相同，[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev45-validation-20261101.json`）保存版本、配置和响应哈希。仅六项确定性生成检查，不是数据集精度或吞吐指标，亦不归到后续dev46。

## 52. 2026-11-01 dense mxscale 准备、生产与消费

从已提交 `0995491` 继续：模型已有dense_mxscale_ready/set_dense_mxscale调用，却缺少公共符号和融合RMSNorm/e8m0来源模块。现迁入8个公共/配套helper、AITER完整consumer及加载后准备方法，绑定原生模块/类的缺失符号并检测冲突；block post-load仅对实际hipBLASLt或AITER consumer采用完整来源方法。保持capability检查、gfx95 bpreshuffle条件、shape白名单和正normal二次幂scale的无损判定；先shuffle权重，再附加e8m0 scale副本，共享生产端要求全部consumer ready。

模型融合生产端改用本仓byte-identical RMSNorm/e8m0 kernel；不假装官方存在缺失模块。[源码对照](hcu-sglang-development.md#历史-json-快照)（`hcu-dense-mxscale-source-alignment.json`）归一native owner后10个函数AST相等，模型helper仅重绑定import，融合kernel整文件逐字节相同。源树11项CPU路由/数据契约、7项block回归、3项真实gfx938融合GPU/Graph通过，覆盖权重/scale准备、早期alias、混合scale保留fp32 backend、缺失scale拒绝、空rank不launch、M1/17/8、K256/384/7168、零输入与改变input/weight的Graph replay。数值参考独立计算RMSNorm及ceil(log2(amax/448))量化，实际scale字节按column-major storage解释。

实际SGLANG_USE_AITER=1导入发现安装的vendor缺少gemm_a8w8_blockscale，来源/原生fp8_utils也无法导入，原始失败保留。CPU测试在实际可导入环境验证数据流并显式模拟gfx95条件，不能证明优化GEMM数学或性能；gfx938上的融合Triton生产端是真实执行，不能替代gfx95 mxscale bpreshuffle GEMM。未升级或替换vendor，未禁用优化分支掩盖失败。

测试夹具早期误把view要求为同一Python对象、缺少条件native import的fixture及解包错误已修正，原始日志保留。运行时guard使用getattr保留AITER关闭时原生模块没有该alias的正常情况；未修改来源数学或放宽容差。当前dev46隔离wheel/全Hook/门禁按实际结果补记；其余DAS与完整dev46模型未完成。

隔离dev46实际结果：11项mxscale CPU、7项block回归、3项真实融合GPU/Graph及404/404 Hook、3seam通过；585包文件与源码/site一致，wheel SHA256 `b2760cbc92f83cc260f6d50d186922303bba616ef9e85890d7b01c36ebf5ecdc`。本地79项仓库unittest、L0、执行台账、Skill元信息和实际宽/窄屏检索/阅读/导航检查通过。此结论不包含gfx95 mxscale GEMM、性能或完整dev46模型；本阶段dev45八卡验证单独归属 `0995491`。

## 53. 2026-11-01 dev46 DeepSeek-V4 八卡复验

提交 `5ba8e96` 的隔离dev46，04:17:30至04:23:01按原TP8/DP8/EP8、DSPARK/DeepEP、Graph32、context32768、64线程加载完成六项严格请求，全部正常stop、自有进程为空、端口关闭。585包文件与提交/wheel/site逐字节相同。[运行摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev46-validation-20261101.json`）保存实际版本、配置与响应哈希，不归到dev47，不代表数据集精度或性能。

## 54. 2026-11-01 compressed FP8 的完整调用与加载契约

从本地 `5ba8e96` 继续。进程启动前显式设置 `SGLANG_USE_LIGHTOP_CHANNEL_FP8=1` 启用该路径，缺省关闭；它只选择compressed CHANNEL路径，不能代替block/tensor配置或全模型兼容验收。LightOp channelwise opt-in要求OCP E4M3权重[N,K]，不能继续使用普通HCU转置/FNUZ布局。完整提取来源helper、scheme capability/post-load/apply与上层CompressedTensorsLinearMethod.apply；复用来源与官方AST相等的6个native构造/建参/入口方法。公开capability限定HCU、非AITER、dynamic channel、非block；channel opt-in保留连续OCP权重，动态调用真实LightOp quantizer或直接复用tuple，并保留来源RuntimeError→KME重试及其他错误传播。

[来源对照](hcu-sglang-development.md#历史-json-快照)（`hcu-compressed-fp8-source-alignment.json`）归一namespace、property绑定和抽取方法名后恢复5个函数AST，并显式列出受控适配。AROUND保留legacy函数反射签名，为owned scheme提供来源参数元信息，其他scheme仍按来源inspect/cache过滤参数；不mock掉上层native wrapper入口。

真实CPU反例发现两个来源逻辑缺口：compressed caller已传pre_quant_output_dtype，但generic HCU tuple分支固定BF16；compressed block后端选择来源hipBLASLt时，加载后权重/scale未做其要求的成对转置。现分别令allocation遵守forwarded dtype（缺省仍BF16），仅对该block consumer转换weight及scale，保留Parameter对象与其他backend布局。[generic对照](hcu-sglang-development.md#历史-json-快照)（`hcu-fp8-linear-source-alignment.json`）追加dtype适配，不再把这处修复称作逐字源相同。

源树14项CPU从实际native构造/建参/加载/包装apply到backend验证通过：compat kwargs、其他scheme旧签名、capability矩阵、动态/预量化/FNUZ、opt-out DeepGEMM dtype、KME复用同一operand、错误传播和非方形block布局。初始fixture缺tp_group已补TP1上下文；原始dtype与布局失败保留。GPU初次bias使用FP32违反LightOp的bias.dtype==out_dtype契约，已改为实际模型对应dtype，未改运行时规避检查或放宽容差。真实GPU/Graph、隔离wheel和全Hook按实际结果补记；不据此关闭所有compressed schemes或全部DAS。

源树6项真实GPU测试全部通过：opt-in BF16/FP16预量化+bias、动态多维量化、opt-out tuple输出dtype、KME实际计算、更新activation/weight scale的Graph，以及native compressed block加载到实际hipBLASLt的非方形scale数学对照。自然预量化两种dtype均为primary1/KME0；KME数学由明确的primary miss夹具触发（KME1），不称为自然硬件失败。统一rtol/atol=0.025，未放宽容差；block反例由[256,512]原错误layout转为[512,256]后才通过。

隔离dev47实际结果：14项CPU、6项真实GPU/Graph、407/407 Hook与3seam通过，errors/unapplied为空；586包文件与源码/site一致，wheel SHA256 `3f9ba712b33e60c70e063b6e7b136175eadf56989988f7a60aac7111572331b2`。79项仓库unittest、L0、执行台账、Skill与实际宽/窄屏检索/阅读/导航检查通过，固定official/fork输入工作区复核干净。该artifact的原配方完整八卡模型验证尚待独立运行；dev46六请求证据仍只归属 `5ba8e96`。

## 55. 2026-11-01 dev47 两组原配方八卡复验

提交 `d2dc00d` 的隔离dev47先后完成[DeepSeek-V4](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev47-validation-20261101.json`）与[GLM-5 INT4](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-glm5-dev47-validation-20261101.json`）八卡配方，各六项严格请求均正常stop，已清理自有进程与监听端口。两次586包文件与提交/wheel/site逐字节相同，分别04:41:59–04:47:30、04:48:50–04:57:01，均只归到dev47。

DeepSeek保留TP8/DP8/EP8、DSPARK/DeepEP、Graph32/context32768和64线程加载。GLM保留原TP8、INT4、EAGLE、FP8KV、默认Graph、memory0.85和显式AITER custom-AR；八rank IPC选择记录与multimem all-gather运行时capability降级日志独立计数，没有手动取消优化参数。GLM SlimQuant使用独立LinearMethod，属于组合回归，不能用它替代所有compressed schemes的验证。

## 56. 2026-11-01 FP8 utility 剩余定义与递归盘点

从本地 `d2dc00d` 继续，迁入完整显式backend选择函数与公开torch_fp8_block_gemm参考函数。HCU请求不可用AITER时，错误现在按来源指出HCU设备及SGLANG_USE_AITER=1，其他backend返回有效native/已hooked consumer；非HCU保留有效原分派。公开参考函数按来源使用per-row/per-K-block及per-K/N-block scale，不把参考Torch矩阵乘法接入优化计算fallback。

[定义/global对照](hcu-sglang-development.md#历史-json-快照)（`hcu-fp8-utils-source-alignment.json`）覆盖17个变化的顶层定义与62个AST未变的native定义，并核对7个顶层global及conditional import归属。Native CK表保留来源2行并额外保留(4096,2048):2048与(4608,4096):512两条上游ROCm7.0 NaN防护，不能为了消掉旧DAS差异删除这些已存在保护。其gfx95性能/数学依然未经当前gfx938验证。

递归FunctionDef检查发现顶层盘点外的CUDA-only _fp8_blockwise_scaled_mm_abstract，另有CUDA raw FlashInfer BMM从eager变LazyImport的来源差异，均尚未迁入。它们不会在当前HCU条件命中，但仍在本文件处置范围内，因此fp8_utils继续in_progress，不把顶层17定义核对称为整个模块完成。

源树6项CPU通过实际公开/早期alias检查显式backend身份及诊断，并用独立scalar double乘积验证ragged K/N block、三种输出dtype、输入不变和空token。2项真实HCU GPU/Graph将reference函数与CPU double按block求和对照，覆盖K257/N129尾块和更新scale的Graph；它是新公开debug/reference helper的设备验证，不代表优化GEMM新增或性能结论。

另外2项CPU使用真实CompressedTensorsLinearMethod及native W8A8 INT8静态/动态scheme，实际构造、建参、post-load、signature cache/kwargs过滤到apply均验证；仅量化和scaled-MM计算mock，证明旧签名路由不回归，不能声称INT8 HCU kernel验证，也不替代后续INT8 scheme来源迁移。隔离wheel、全部Hook及门禁按实际执行补记。

隔离dev48实际结果：6项utility CPU、2项真实native INT8 scheme旧签名路由CPU、2项reference GPU/Graph及408/408 Hook、3seam通过，errors/unapplied为空；586包文件与源码/site逐字节一致，wheel SHA256 `3269c3b34ed33d5e3358417b9af53f05e454526b22a2b83ce6d17c4675c08be8`。79项仓库unittest、L0、执行台账、Skill与实际宽/窄屏检索/阅读/导航通过。Generic来源反向比较新增精确受控分支AST检查，先验证允许的buffer/dtype适配再恢复Source，避免笼统替换整个分支掩盖额外修改。dev47两次八卡结果不能自动记为dev48；其余CUDA条件分支和DAS范围仍未完成。

## 57. 2026-11-01 dev48 DeepSeek-V4 原配方八卡复验

本地 `b0eafcf` 的隔离dev48从05:13:31至05:19:13按原TP8/DP8/EP8、DSPARK/DeepEP、Graph32/context32768及64线程加载完成六个严格请求，正常stop，自有进程为空、端口关闭。586包文件与提交/wheel/site逐字节相同，[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev48-validation-20261101.json`）保存实际版本、响应与配置哈希，不归到后续dev49，也不是数据集精度或吞吐指标。

## 58. 2026-11-01 FP8 MoE 建参、HCU分派与加载布局

从本地 `b0eafcf` 继续。来源HCU在FP4建参保留int8 packed bytes与FP32 scale，不使用generic ROCm的e8m0 scale初始化；generic runner-core AiterMoeQuantInfo在HCU返回None；standalone/grouped MegaMoE有各自格式边界；HCU AITER/ASM的block-FP8加载消费b8布局，不能套用generic16x16 shuffle或MXFP4路径。上述4个变化方法需配套迁移，不能只改单个dtype。

完整来源方法迁入owned fp8_moe，四个HCU-only AROUND沿原生Fp8MoEMethod绑定；非HCU保留有效original。Native HookRegistry保留static factory的staticmethod descriptor。既有MegaMoE注册先发布HCU runtime/builder，再注册quantization，避免导入未发布helper。ASM使用来源aiter.ops.shuffle.asm_shuffle_weight_b8 provider，在实际分支调用时加载，未伪造vendor接口。初值SGLANG_USE_AITER_FP8_ASM_MOE=False及gfx95 AND NOT HCU的MXFP4 shuffle条件保留来源语义。

[来源对照](hcu-sglang-development.md#历史-json-快照)（`hcu-fp8-moe-source-alignment.json`）覆盖Fp8MoEMethod全部4个变化定义，归一native P后AST相等；16个AST未变方法继续复用native，包括真实constructor、create、apply和FP4转换helper。来源的声明、注释和许可保留，说明放本记录；方法抽取阶段误带第二个future import，已在执行测试前修正header，未改来源计算语义。

10项CPU通过实际Native Fp8Config/Fp8MoEMethod构造、create_weights到static factory，验证FP4在AITer/e8m0 override请求下仍生成FP32 scale；验证HCU quant-info在no_combine assert之前返回、standalone非法格式先拒绝、grouped block转交、ASM顺序1/2且不走generic shuffle、packed FP4 int8保留、dequant早退与non-HCU original。MegaMoE builder及CPU ASM计算采用明确spy/mock，不作数学精度结论。

2项真实GPU均从native完整process_weights_after_loading入口执行：FP8 E2/H256/I256在真实已安装ASM b8 provider后，用独立逆排列逐字节恢复w13/w2，保留Parameter及storage；packed FP4用全nibble值与正二次幂scale模式，经真实native FP4->FP8转换后按128-block scale解码，与独立E2M1解码逐值相等，保留Parameter对象。它们证明布局/转换，不证明完整MoE expert routing、GEMM或性能；未把权重初始化反复Graph replay当作forward Graph验收。

当前容器全局SGLANG_USE_AITER=1的FP8 import仍缺vendor API；方法分支内的实际b8 provider可用，不等于整个AITer FP8配置已可部署。CPU patch分派标志与真实GPU provider证据分开记录。dev49隔离wheel、全Hook及完整模型按实际执行补记；全DAS、其余global/conditional/provider组合仍未完成。

追加非serialized BF16 expert的完整加载检查：从native create到owned HCU quant producer量化w13/w2，得到FP8权重与FP32 per-expert scale，解码后按FP8舍入界限rtol0.075/atol0.005比较原BF16值；不是放宽既有lossless FP4或byte布局断言。初次fixture缺实际FusedMoE的num_local_experts上下文，补齐2个expert后3项GPU通过。最终隔离用例另断言两组权重、2expert共4次调用确实进入owned producer，避免只凭数值推断alias已切换。

进一步补查真实routed FP8 consumer：以完整插件入口注册、真实ServerArgs发布exec上下文，TP1用公开parallel override显式声明无通信group；模型配置只读位置由HCU_TEST_MODEL_PATH提供。E2/H256/I256、M16/topk1实际native create_moe_runner→apply选择Triton，HCU alignment计数1，两次真实GEMM各自记录中间结果供独立Torch检查，未替换任何计算kernel。

最初仅登记FP8相关方法的测试漏掉了生产入口已有的HCU alignment，原生小batch CUDA alignment触发320线程超过256 launch bound并使后续fused kernel VM fault。GPU进程退出后设备内存恢复，未重置设备或修改来源；改用完整register复验不再触发故障。该失败属于不完整测试注册，不能描述成修改kernel后修好了生产故障。

直接BF16 dense参考漏掉两次activation FP8量化、中间BF16舍入，产生31/4096差异；补齐后单点量化阈值仍放大首层rounding差异。现分别核对gate/up GEMM与activation（rtol/atol0.025）、用实际中间值作down阶段输入再独立按128-group量化/分块Torch计算down及route，最终out维持原rtol0.08/atol0.03，不放宽门限。两次GEMM和SiLU均独立检查，实际forward结果未被参考值替换。源码环境单用例通过；最终隔离四用例复验按实际结果补记。

最终隔离dev49四项GPU用例全部通过（完整register、HCU_TEST_MODEL_PATH真实只读配置上下文、Native routed forward）；owned producer4次及HCU alignment1次断言通过。10项CPU、412/412 Hook、3seam和79项仓库unittest通过；587包文件与源码/site一致，wheel SHA256 `ad42689f38b850b6bdc6c53d8860e512ea9a2369c9a486386068bb94b7442f9f`。L0、执行台账、Skill及实际宽/窄屏检索/阅读/导航检查通过。完整dev49八卡服务仍待独立运行，不把dev48结果归到它；其他MoE后端/拓扑、global AITER配置与全DAS仍未关闭。

## 59. 2026-11-01 dev49 DeepSeek-V4 原配方八卡复验

提交 `dc59cd0` 的隔离dev49在06:11:43–06:17:19按原TP8/DP8/EP8、DSPARK/DeepEP、Graph32/context32768及64线程加载完成六项严格请求，均正常stop，自有进程为空、端口关闭。587包文件与提交/wheel/site逐字节相同。[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev49-validation-20261101.json`）保存实际版本、配置及响应哈希；不归到dev50，不代表所有MoE profile或数据集精度/性能。

## 60. 2026-11-01 compressed INT8 strategy 与预量化入口

从本地 `dc59cd0` 继续。来源使用W8A8_SUPPORT_METHODS选择1 Triton、2 cutlass入口（HCU实际委托rocBLAS NN）、3 hipBLASLt、其他值rocBLAS；不同strategy的权重布局必须与其provider一起迁入。默认值1；变量应在进程启动前设置。完整迁入constructor、static scale normalization、post-load与apply，使用既有owned quant_ops/cache_config，来源gfx928 KME predicate逐字保留。普通gfx936/gfx938 strategy3保留连续[N,K]；gfx928按来源[K,N]、stride(1,K)打包，当前设备无该硬件验证。

BF16 checkpoint scale在加载前转FP32；strategy1按已有cache reader读取指定TRITON_JSON_DIR，只有提供的config才预热，LMSLIM_TUNING_JSON显式存在时才导出model.json。未向输入仓库或模型目录写文件。SGLANG_USE_LEGACY_FUSED_RMS_QUANT与SGLANG_USE_FUSED_SILU_MUL_QUANT仍缺省关闭，启用后消费对应input_quant_args/silu_quant_args，RMS优先。Native linear wrapper读取owned scheme参数元信息，避免AROUND旧反射签名吞掉新增参数。

[源码对照](hcu-sglang-development.md#历史-json-快照)（`hcu-compressed-int8-source-alignment.json`）归一native N后4个变化方法及架构predicate AST相等，create_weights/get_min_capability复用AST未变native方法；沿用已存在的owned cache兼容类，不把其实现称为本阶段逐字Source迁入。Non-HCU保留有效original。Legacy旧签名测试只登记wrapper，保留真实未迁入native INT8类作接口兼容单元测试；生产GPU用例登记完整插件，这与forward测试漏登记alignment的错误情形不同。

严格public CPU反例确认共享legacy hipBLASLt helper接收bias却直接返回不含bias的GEMM结果，6/6元素全部丢失。现仅在该返回路径用in-place add加入非None bias并保留输出dtype；精确移除该AST增量后恢复完整来源helper。未替换kernel、未禁用hipBLASLt；gfx928 KME原有bias处理保持其provider接口。

13项CPU覆盖真实native constructor/create/post-load/wrapper apply、BF16 scale、gfx928 stride/AZP元信息、tensor scale、cache预热/m bucket、两种预量化参数优先级、四provider分派及异常、bias反例和non-HCU委托。4项真实GPU覆盖四种strategy实际backend、普通动态INT8量化、RMS pair进入真实hipBLASLt、SiLU pair进入真实Triton、以及更改scale/bias后的Graph重放；数学参考使用实际量化结果及独立FP32矩阵乘法，rtol/atol0.025。Static/asymmetric仅核对来源metadata，不能冒称AZP kernel数值通过；gfx928也仅CPU契约通过。隔离wheel、全Hook与模型结果按实际执行补记。

隔离dev50实际结果：13项INT8 CPU、4项真实GPU/Graph、2项指定wrapper旧签名兼容单元测试、415/415 Hook与3seam通过，errors/unapplied为空；588包文件与源码/site逐字节相同，wheel SHA256 `2d897fad205952c99b6c66058937d2918da0bed1c7222bb55359c6b466b94997`。本地79项仓库unittest、L0、执行台账、Skill及实际宽/窄屏检索/阅读/导航检查通过。新增helper bias按独立AST增量记录；完整dev50模型未由这些局部检查替代，dev49模型结果仍仅归属dc59cd0。

## 61. 2026-11-01 dev50 GLM-5 INT4 原配方八卡复验

提交 `814792e` 的隔离dev50于06:31:46–06:37:17完成原TP8、INT4、EAGLE、FP8 KV、memory0.85、默认Graph和显式AITER custom-AR配方，六项严格请求均正常stop，自有进程为空、端口关闭。588包文件与提交/wheel/site逐字节相同，[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-glm5-dev50-validation-20261101.json`）记录实际响应、配置和backend日志计数。GLM走独立SlimQuant，本次是共享helper/组合回归，不作为新compressed INT8 scheme的全模型证明，也不归到dev51。

## 62. 2026-11-01 small-M ASM 与量化 target 父类匹配

从本地 `814792e` 审查相邻pending文件发现：既有quant_ops虽已迁入大部分helper，却缺来源small-M ASM fast path及其致命binary缺失架构保护；Native matching只看当前类名，不能识别不含Linear命名的自定义子类。现迁入完整来源7个helper和匹配函数，保留默认Triton/显式config/bias/layout/shape gates，以及来源异常后Triton fallback。m/n的slide1/9选择、完整K256 tile要求、gfx928禁用与未知arch probe失败均按来源保留，未为了通过改掉kernel选择。

当前官方utils没有is_hcu导出，owned module将该导入绑定到既有detection.is_hcu；未给官方模块伪造符号。Source quant_ops全部函数与module其余AST均核对，除这处真实provider重绑定及dev50精确bias AST增量外相同。[全文件对照](hcu-sglang-development.md#历史-json-快照)（`hcu-quant-ops-targets-source-alignment.json`）还证明target utility唯一变化函数归一native N后相同，其余完整module AST与native相同，因此这两项代码处置改为implemented；不据此把全部DAS或所有profile改为完成。

Target按exact/regex layer path、Python MRO中的leaf/parent、fused全组件匹配的顺序选择，跳过object，支持空字符串显式匹配。公开Hook传播到早期alias，并保留non-HCU有效original；不因自定义layer名称改变而漏配quantization。

6项CPU验证ASM架构缓存、source slide与provider完整参数、bias/tuned config/partial K/large M回到原Triton、gfx928不调用会exit的provider、exception和non-HCU边界；6项CPU验证target inherited-only、leaf/regex/path/MRO优先级、fused-all/object/空值。3项真实gfx938 GPU验证自然small-M命中ASM1/Triton0、带bias命中ASM0/Triton1及实际ASM Graph读取scale更新，独立FP32数学参考rtol/atol0.025。gfx928只测保护契约，不作硬件执行结论；没有性能A/B结论。隔离artifact、全Hook和消费者回归按实际结果补记。

追加真实native CompressedTensorsConfig.get_scheme_dict消费者检查：自定义HiddenProjection仅通过父类LinearStem命中，返回实际QuantizationArgs payload；用于验证native config模块内的早期import alias也更新。该用例将target CPU检查增至7项，隔离执行结果单独记录。

隔离dev51实际结果：6项分派CPU、7项target CPU（含真实native config消费者）、3项ASM/Triton GPU/Graph和4项实际compressed INT8 backend回归均通过；416/416 Hook、3seam与79项仓库unittest通过。589包文件与源码/site逐字节相同，wheel SHA256 `8d2c37a703cafbcea4f148a34a6a79492c59c2337715ff42273d059020535635`。L0、执行台账、Skill及实际宽/窄屏检索/阅读/导航检查通过。Source helper7函数完整覆盖与config utility单一变化覆盖只支持对应两项implemented处置，未关闭其他文件或全DAS。另有106个pending来源路径与owned文件字节相同，仅作为后续编译/注册/caller审查线索，未凭此更改状态；完整dev51模型尚未运行。

## 63. 2026-11-01 dev51 GLM 原配方八卡复验

本地 `701fce6` 的隔离dev51于06:49:40–06:55:01按原GLM INT4 TP8/EAGLE/FP8 KV/默认Graph/memory0.85和AITER custom-AR配方完成六项严格请求，全部正常stop，已清理自有进程与监听端口。589包文件与提交/wheel/site逐字节相同，[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-glm5-dev51-validation-20261101.json`）保存实际配置、响应与backend日志计数；不归到dev52，不是数据集精度或性能分数。

## 64. 2026-11-01 JIT 资源的可重复编译闭包审查

前阶段106个pending同字节路径只是线索。本阶段新增只读工具[资源审计](../tools/audit_jit_resources.py)，从固定fork Git tree核对plan OID、owned文件mode/bytes及wheel成员，并验证历史sgl_deps.json的deps-key、库文件存在、历史安装与当前owned内容摘要。清单与路径被当作数据，不执行来源代码或载入.so；输出不修改处置状态。工具6项单元测试覆盖来源HEAD/OID漂移、缓存清单篡改、库缺失、历史/当前内容漂移、wheel遗漏、外部安装路径与来源删除，并核对Git index/状态未改变。

本机历史review为14个recorded compiler dependencies、74个仅同字节owner、4个来源已删除；[覆盖摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-jit-resource-coverage.json`）保留代表性manifest及哈希。历史编译记录不是本次执行、更不是所有模板数值通过。这14项先转in_progress审查，74项和删除项不自动完成；输出目录由显式变量指定，未改写源仓库或全局Git配置。

4项native load_jit/BuildSpec检查从早期import alias进入真实loader，在compute_build_key前截获不可变spec，核对owned cuda_files、include首位、外部绝对路径与native未覆盖路径、重复注册和调用方list不变。Header-only的实际include保存在translation_units().source，而非ninja文本的cuda.cu文件名中；原测试断言位置已修正，未更改loader来迎合测试。

冷缓存实际调用owned candidate_blocks.amax8_varlen暴露block_amax.cuh的HIP编译错误：__launch_bounds__中带<typename, typename>的表达式逗号被宏解析，后续报BlockAmaxKernel不存在并不代表Python模板实参顺序错误。现只给两个常量表达式加括号，线程512、occupancy4及算法不变，[精确反向字节对照](hcu-sglang-development.md#历史-json-快照)（`hcu-jit-block-amax-source-alignment.json`）证明除这处适配外保持来源。没有换Torch/Triton fallback或修改输入树。

新任务cache目录强制经历实际JIT编译，2项gfx938 GPU检查覆盖FP32/BF16 input/output四组合、空/partial/skip rows、最新块+inf、未写区域保留，以及改变score/length的实际Graph replay，逐值一致。四个新编译清单也被审计；该header被正确标成owner_differs而非伪称同字节。[Source/覆盖记录](hcu-sglang-development.md#历史-json-快照)（`hcu-jit-resource-coverage.json`）将受控适配、当前实际GPU测试与历史编译分开。对应block-amax实现完成；其余资源的模板、caller与profile仍待审，不能把14份历史记录视为整仓完成。

工具复跑：将FORK、PLUGIN、WHEEL、CACHE、INSTALL_BASE、REPORT分别设为只读fork、插件目录、待核wheel、JIT cache、历史隔离安装父目录和新输出文件，然后执行：

```bash
python3 tools/audit_jit_resources.py --plan "$PLUGIN/plans/sglang-das-execution.json" --fork "$FORK" --plugin-root "$PLUGIN" --wheel "$WHEEL" --cache-root "$CACHE" --install-base "$INSTALL_BASE" --output "$REPORT"
```

运行工具需要Python3.10+和当前用户对上述只读输入的权限；cache可省略，此时不产生编译证据。GPU用例使用显式SGLANG_JIT_CACHE_DIR独立空目录做冷编译；后续隔离wheel/门禁与完整模型按实际执行补记。

隔离dev52实际结果：6项audit工具单元、4项native loader/BuildSpec CPU契约、2项实际GPU/Graph（四dtype编译变体）全部通过；416/416 Hook、3seam与85项仓库unittest通过。589包文件与源码/site一致，wheel SHA256 `4cfeeee574672743e8cb9424207abfb6413cd07504edc3856c7c182c088bf47b`。隔离cache的4份新依赖清单与库文件及当前owner摘要核对，owner_differs状态按语法适配保留。L0、执行台账、Skill和实际宽/窄屏检索/阅读/导航检查通过；当前完整dev52八卡服务仍待执行，不能归用dev51模型结果。14个历史资源保持in_progress，74个只同字节与4个删除项不作执行/完成声明。

## 65. 2026-11-01 dev52 DeepSeek 原配方八卡复验

本地 `b2302c3` 的隔离dev52于07:23:57–07:29:23按原TP8/DP8/EP8、DSPARK/DeepEP、Graph32/context32768及64线程加载完成六项严格请求，均正常stop，自有进程为空、端口关闭。589包文件与提交/wheel/site逐字节相同，[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev52-validation-20261101.json`）记录版本、配置及响应哈希；不归到dev53，不是所有kernel/profile或数据集精度证明。

## 66. 2026-11-01 BF16 short-row radix top-k 的HCU接入

从本地 `b2302c3` 继续选择仅有字节存在证据的topk_bf16_small.cuh，通过既有公开topk_transform_bf16_small冷编译。原Source在gfx938报5类编译错误：CUDA shuffle、signed dot打包与BF16 pair mask intrinsic不可用，原始日志保留。pivot原为固定32lane/每lane8个bins；后续dev54核对确认当前Source使用逻辑warp32，不能把硬件wave64等同于该逻辑宽度。

现使用既有portable shuffle/broadcast；每lane管理256/deviceWarpThreads个bins；当前Source utils.cuh固定逻辑warp32，CUDA与HIP均为8，不能按硬件wave64推断为4，对应vector load、signed-MSB顺序和suffix pivot按实际宽度分配。HIP pair比较直接使用BF16原始位序，显式处理NaN与signed-zero相等；8个半字比较mask用整型shift打包，CUDA保留原intrinsic。线程512、occupancy3、两阶段radix、top-k quota和page-table接口保持，未强制wave32，也未换用torch topk计算返回值。来源已有说明保留，CUDA特定lane注释按原文保留；HIP差异由本记录说明。

[Source对照](hcu-sglang-development.md#历史-json-快照)（`hcu-bf16-topk-source-alignment.json`）列出全部适配；当时保存的差异补丁经临时副本反向 apply 后与固定 Source 逐字节相同，补丁现可从 Git 历史读取。首轮helper类型限定误将全局bf16x2_t写成device成员，已修正后重新冷编译；未伪造vendor API。

5项公开入口真实GPU检查通过：256宽下空/partial行与任意返回顺序下的精确page slot集合；来源最大width16384/k2048与positive/negative BF16 bins；重复值与signed-zero quota128、唯一且有效索引；有限subnormal位序；Graph中同时改变score、length和page table。普通/最大值参考使用独立topk/page公式，subnormal参考使用整型BF16 rank，均为exact比较而非放宽浮点容差。

Subnormal首轮64/64差异不能归因为GPU：默认CPU float top-k给出129开头索引，关闭CPU flush-denormal后与整数位序同为63开头，kernel整数参考结果正确。初次将该失败直接归因HIP浮点比较的判断不充分，原probe/log保留；最终HIP比较按BF16位语义实现并避免任何flush依赖，不把参考环境修正冒称生产GPU bugfix。NaN沿来源接口不支持；CUDA32仅静态保持原intrinsic/8bin specialization，未在NVIDIA实测。

隔离dev53实际结果：独立空JIT cache冷编译并通过5项真实GPU/Graph检查；416/416 Hook、3个resolution seam、0 errors与85项仓库unittest通过。589包文件与源码/site逐字节相同，wheel SHA256 `3de6fde9d43a45cb9a153a742fd1a30b73766577e63c4ea7ead2411f7b559d78`。资源审计记录新cache的4项编译依赖，受控patch仍标记owner_differs，不混同字节一致与执行证据。完整dev53模型服务尚未执行；该单资源实现完成不关闭其余JIT资源和全DAS范围。

## 67. 2026-11-01 dev54 MoE hash routing 与空 pre-dispatch 接入

核对固定DAS的kernels/ops/moe/dsv4.py与官方差异：HCU应选择已有owned hash_topk JIT而非通用HIP Triton；空输入的非标准stride需在mega-MoE pre-dispatch之前恢复标准空布局。本阶段保留两个完整Source函数与既有native JIT factory，新增HCU范围AROUND并保留非HCU有效原调用链。入口、pre-import alias、真实GPU/Graph、数值与隔离产物验证待实际执行，当前不作完成声明。

## 68. 2026-11-01 dev53 DeepSeek 原配方八卡复验

隔离dev53对应提交 `acb7ae6`，按原TP8/DP8/EP8、DSPARK/DeepEP、Graph32/context32768与64线程加载完成六项严格请求，均正常stop，自有进程清空、端口关闭。589包文件与提交/wheel/site逐字节相同；[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev53-validation-20261101.json`）记录时间、版本、配方与响应哈希。该结论属于dev53，不归用到新增MoE ops Hook的dev54，也不是数据集精度或全DAS迁移完成证明。


dev54边界复核：owned utils.cuh的device::kWarpThreads固定32，hash接口保持topk_fused<=32；首轮64路测试超出Source契约，拒绝属于预期边界，不能据此扩大接口。dev53的top-k constexpr仍按实际逻辑32选择8bins，真实5项GPU结果有效，但先前“HIP64为4bins”的解释不实，已更正。未改全局warp常量或强制底层执行模式。


dev54隔离GPU的7项边界检查中6项通过，零token hash触发Source的零grid HIP launch错误。dev55在hash与同header的mask host入口完成参数核验后、发射前对num_tokens==0直接返回；非空数学、warp、PDL、路由与page布局不变。当时保存的差异补丁保留 Source 差异，现可从 Git 历史读取；失败dev54 wheel/cache/log不覆盖。dev55冷编译与隔离重验待实际完成。


隔离dev55结果：7项真实GPU/Graph检查全部通过；4项CPU契约及85项仓库测试已在本阶段执行，当前418/418 Hook、3个resolution seam、0 errors。590包文件与源码/site逐字节相同，wheel SHA256 `8e25ca64434c064581b177ee7282fe7b75c401bc8b23320ff66d93fc10c11fac`。hash/mask两份新JIT清单均引用当前owned header，精确patch反向恢复固定Source；资源审计仍将受控差异标为owner_differs。完整dev55原配方服务待执行，不挪用dev53结果；其余DAS待办保持原范围。

## 69. 2026-11-01 HCU BF16→FP32 GEMM 的Source与执行核验

继续使用已冻结dev55验证既有owned实现，不把证据补齐伪装成新运行时版本。HCU helper全部4个函数AST与Source相同；gemm dispatcher全部17个函数在仅重定位函数内owned helper import后AST相同。平台检测与模块路径由owned命名空间承接。[对照记录](hcu-sglang-development.md#历史-json-快照)（`hcu-bf16-fp32-source-alignment.json`）保留两份来源/owner摘要。7项DAS原始dispatch CPU测试迁入本仓执行环境并通过。

Source自动调优profile限定gfx936，当前机器gfx938应保留原策略，不能为让原始GPU测试中的SGL断言通过而伪造profile；另通过显式sgl选择检查gfx938真实内核。待执行普通及split-K、M16/M8边界、最大M64、输出清零、独立CPU double参考、真实公共Hook和Graph重放；不预先声明GPU或性能结果。


dev55八卡于Graph捕获失败：HashTopK收到BF16 router_logits，Source JIT只收FP32，自有进程和端口已清理。沿producer定位Source MoEGate.forward在HCU DSV4通过linear_bf16_fp32生成FP32，而当前native gate走非CUDA的F.linear产生BF16。dev56迁入完整Source gate构造与forward，保留router配置dtype、输入weight dtype转换、CP边界、tiny/AITER/auto优先级及已有条件；不在hash consumer强制cast。所有Source说明与许可保留，super绑定到原MoEGate的父类，非HCU保留原链。CPU/GPU producer→consumer和原TP8配方复验待执行。


Source工作树执行结果：gate的5项CPU与2项真实producer→hash GPU/Graph通过，GEMM的7项原始CPU方法和6项GPU通过（10个直接kernel shape，含普通/split8/split4、M16和M8边界、M64、重复清零及split-K Graph）。GEMM普通/显式路径用CPU double参考，rtol2e-4/atol2e-5；精确整数fixture为0/0。初次auto参考错误地假设FP32累加输出，Source在HCU明确先BF16输出再.float()，现按相同Source输出契约将CPU double结果round到BF16再FP32，exact通过；未改运行时后端或放宽显式kernel阈值。

[dev55失败摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev55-failure-20261101.json`）单独保留原配方、提交、590包文件一致性、实际dtype错误与清理，不将失败运行计为通过。新gate的[Source对照](hcu-sglang-development.md#历史-json-快照)（`hcu-deepseek-gate-source-alignment.json`）明确仅两方法完成，deepseek_v2.py其余差异仍待审计。dev56隔离wheel/完整原配方验证按实际结果补记。


隔离dev56实际结果：gate 2项GPU/Graph与5项CPU、GEMM 6项GPU与7项原始CPU均通过；420/420 Hook、3个resolution seam、0 errors与85项仓库unittest通过。591包文件与源码/site逐字节相同，wheel SHA256 `ace94cd214bf842056eb11fb102b9b9e0c47f60a3feffccbcf9f396dc84859a5`。新cache依赖清单中的linear header与固定Source/owner/wheel同字节，真实执行由GPU用例独立证明。当前只证明gate两方法和已核对的GEMM实现，完整dev56原配方与其他DAS差异继续待验。

## 70. 2026-11-01 dev56 原配方八卡验证与MoE路由全量接入

隔离dev56/提交34ff4e9于08:33:06–08:38:46按原TP8/DP8/EP8、DSPARK/DeepEP与Graph32/context32768完成六项请求，均正常stop、自有进程清空与端口关闭。591包文件与提交/wheel/site相同，[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev56-validation-20261101.json`）固定产物与响应摘要。证明gate producer修复越过dev55原故障，不代表所有DAS功能完成。

继续迁入完整TopK与HashTopK来源模块及合成EP均衡helper；已确认相同AST的9个共享类型复用native身份，方法super显式绑定原类父级，保留来源许可/注释和条件custom-op注册。替换旧单分支softmax注册入口，HCU范围接入全部7个变化helper、TopK的3个变化方法与HashTopK的2个变化方法。模拟EP均衡按Source默认关闭，仅作合成性能基准，不用于精度结论。Source对照、导入/契约/GPU与隔离/原配方验证待实际执行。


首次真实LightOP GPU检查3/5通过；V4.1 int64/padding融合两项在Source遗留函数名_eplb_remap_enabled处NameError，该文件实际定义的是is_expert_remapping_enabled。仅将该调用修正到实际Source谓词，保留所有融合条件和开关，并增加EPLB开启必须拒绝融合的策略回归；不通过关闭融合规避错误。


Source路由的6项CPU契约与EP helper的5项独立Python oracle检查通过，原class与owned factory均能构造，9个共享类型身份一致。隔离dev57的7项GPU/Graph suite与1项额外DeepEP共享槽位检查通过：真实LightOP softmax、sqrtsoftplus、V4.1 int64/padding融合及改变logits/有效行数的Graph；TopK/HashTopK合成模式保留weights/shared列；共享槽位只重映射一次，并在最后屏蔽padded routes。595包文件与源码/site相同，wheel SHA256 `8eddc8e9a4ed2d877fc8765f2a0e6ac4f5060e83b3f9ae18ff20592c60f1edd3`。

[完整Source对照与patch](hcu-sglang-development.md#历史-json-快照)（`hcu-topk-routing-source-alignment.json`）可反向恢复固定Source字节。EPLB负向单元初次未发布server args，按Source契约代表“无实际mapping”；补齐发布状态fixture后，EPLB开启正确拒绝融合，没有修改predicate本体或禁用优化。TopK/HashTopK文件继续in_progress：当前native LPLBSolver.solve只收topk_ids，Source新增load-aware solver接收有效token数，实际solver/collective尚待迁入验证；全局AITER环境仍未完成兼容，不从LightOP结果外推。完整dev57原八卡配方继续待验。


隔离dev57收尾：431/431 Hook、3个resolution seam、0 errors与85项仓库unittest通过；8项实际GPU/Graph和11项CPU证据按上述范围记录。L0/Skill/guide门禁及实际宽窄屏检索、阅读与菜单检查通过。原配方八卡服务仍需以本次产物另跑，不归用dev56模型结果；全部DAS范围保持未完成。

## 71. 2026-11-01 load-aware replica solver 接入

迁入Source Triton count/water-fill/sample、完整solver模块与dispatch函数，复用未改变字段的native dispatch-info类身份；接通初始化工厂、ModelRunner初始化门控和重平衡后的solver重建。CLI只扩展现有action的choices，验证沿原解析链追加Source load-aware的A2A/ROCm/冗余expert约束，不改冻结seed字段或Struct定义。Source工厂错误地将静态LP专用kwargs传给load-aware构造器；现按算法传参，并拒绝以静态LP概率配置load-aware。Source核验、CLI/worker往返、边界、实际GPU与Graph均待验证，不预先声明功能完成。


Source GPU边界6项初测3项失败：drop/shared ID会越过counts切片并修改测试私有前后哨兵；非连续routed列在count与sample中被按dense展平读取。现count对logical ID加0..num_logical范围mask，dispatch对无效ID返回-1并屏蔽table读取，两处公共入口先连续化真实routed布局；保留water-fill算法、采样、有效行数设备指针和无collective设计。原失败日志与哨兵fixture保留，正确性/Graph重验待实际完成。CPU静态LP的原始全宽slice是共享存储view，测试改为核对内部缓存身份及原tensor相同data_ptr，没有改LP运行时。


Producer→dispatch集成检查发现metadata并非仅Literal变化：Source还新增layer_id/source_rank采样种子字段，native dataclass不能接收该kwargs。现用完整Source dataclass继承native，保留isinstance兼容并通过原init_new classmethod产生完整owned实例；legacy手工info缺少种子字段时采用Source默认0。Native类型对象保留，返回类型有意是兼容子类，不再声称两者身份相同；新增factory→solver→TopK/Hash consumer检查待执行。


抽取ModelRunner方法的集成模块补齐继承自Source model_runner.py的原版权/Apache声明；dev58已冻结，不覆盖其wheel，以dev59保存该许可声明并重验当前产物。该项不添加说明性runtime注释或改变函数计算逻辑。


隔离dev58实际结果：8项真实kernel GPU检查（含256x264完整padded replica表）、2项原生factory→metadata→solver→TopK/Hash pipeline GPU、7项CPU与85项仓库测试通过；完整CLI采用原八卡配方加load-aware/冗余expert参数，正常ServerArgs post-init/resolution与pickle往返均保持算法和值，不跳过解析阶段。只表示配置与单设备数学/调用链成立，未启动load-aware模型服务。

Source load-aware本来就不做每层EP collective，不能把它描述成尚待补all-reduce；仍需以实际8卡dispatch/model变体验证组合。原dev57原配方已于09:02:55–09:08:39六项请求全过、自有进程为空、端口关闭，[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-dev57-validation-20261101.json`）固定595包文件与8af8abf/wheel/site同字节。该结果不归用到当前新EPLB产物。


隔离dev59收尾：600包文件与源码/site逐字节一致，wheel SHA256 `7676b5ab0ef16aba4edf831119e7e8298ddfb60644fc944543f401f687ed9a00`。当前artifact的2项public GPU pipeline、7项CPU、437/437 Hook、3个resolution seam、0 errors及85项仓库测试通过；完整CLI/resolution/pickle再次通过。与dev58相比，唯一package差异是integration.py继承许可头，函数AST相同，8项kernel GPU测试对应的全部代码字节相同；保留分产物证据，不把历史运行冒称当前重跑。L0/Skill/guide与实际宽窄屏验收通过。后续以原八卡配方加显式load-aware/冗余8专家参数运行功能变体，原参数保留；尚未声明模型精度、跨rank负载或性能。

## 72. 2026-11-01 完整replica元数据与静态locality-fair策略

原生_compute_logical_to_all_physical_map会对所有A2A算法折叠nearest副本，Source只允许static+nearest折叠；这会使load-aware/LP看不到完整候选。迁入Source全部4个变化/新增helper，复用其余native元数据类与helper；仅在原生尚无ep_static_dispatch_policy字段时通过进程配置读取Source默认nearest与locality_fair，标准CLI参数由插件sidecar管理并在seed解析前传给workers，不写未声明的冻结字段。新增策略仅允许static，保留GPU→node→remote局部性优先与静态source-rank公平，不伪称动态token负载均衡。当前dev59服务保持其冻结产物，新的元数据验证与后续8卡复验待执行。


Source CPU 8项通过：load-aware/LP/dynamic全候选、static nearest与旧函数相同、locality_fair优先GPU/node与静态公平、非法topology、CLI/worker进程环境继承、规则与非HCU回退，以及原生metadata builder→真实solver保留复制专家计数。初次对照把原函数存成模块变量，被正式alias propagation一起替换，修正为容器保存原callable后，旧函数确实只保留1候选，新函数保留2。该fixture修正没有改变生产逻辑。原dev59变体09:43:15–09:53:14六项请求通过并清理，[摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-load-aware-dev59-validation-20261101.json`）明确旧候选折叠限制，不能据此关闭均衡功能验证。


隔离dev60的2项GPU检查通过原生metadata工厂→真实solver→原生公开dispatch，完整候选使num_replicated为4而非退化；GPU static/locality_fair元数据表也保持全部候选及正确device/dtype。8项CPU通过并验证sidecar值在worker子进程中继承；完整正常CLI/ServerArgs post-init/resolution/pickle对static/locality_fair+冗余8参数通过。601包文件与源码/site一致，wheel SHA256 `43e2eb0f125976ee6b29a008bcf068ba08a9be66cb8680d72c19b4ca7d304c99`。模型尚待重跑，单设备rank-metadata fixture不等于实际跨rank负载或性能结论。


隔离dev60收尾：439/439 Hook、3个resolution seam、0 errors及85项仓库unittest通过；L0、Skill元信息、guide快照与实际宽窄屏检索/阅读/菜单检查通过。增加启动日志汇总每层实际复制逻辑专家数，后续完整候选八卡运行需看到非零replica计数，不能仅以六个回答通过判断均衡链已生效。全DAS与实际负载/性能范围继续未完成。

## 73. 2026-11-01 EPLB dispatch统计与固定形状int64归约

dev60实际八卡启动日志已在全部8个rank记录43层load-aware solver、每层复制逻辑专家数[8]，与旧nearest折叠不同；服务仍运行，未提前记录请求通过。继续迁入expert_distribution.py全部变化：capture期间不collect；DeepEP auto从normal/low-latency实际dispatch累加，避免padded top-k id误计expert0；窗口先本地sum成固定形状int64再all_reduce，支持rank步数不同与大计数。复用native recorder/gatherer/accumulator类和其余方法，新增auto类继承native低延迟gatherer；保留Source许可/注释。CPU、真实Gloo归约和GPU检查按实际结果补记。


Source统计器验证：5项CPU与2项真实GPU/Graph通过，DeepEP auto normal+low-latency累加真实buffer，伪top-k padded IDs不增加计数；实际CUDAGraph capture跳过collect/host append。真实2-rank Gloo的窗口分别为1/3步和0/2步，归约shape固定[1,2]、dtype int64，输出[40,48]与[2**32,6]精确。该证据是CPU Gloo，不冒称GPU collective或性能；隔离8-rank NCCL/RCCL正在执行。

[dev60八卡摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-load-aware-dev60-validation-20261101.json`）记录10:02:15–10:07:53六项请求正常stop、全部8rank各43层solver与复制专家[8]、601包文件与13497b5/wheel/site相同以及清理。它验证完整候选功能变体的短生成，不是数据集准确率或实际dispatch负载均衡量化结果；新的统计器后续另验。


隔离dev61：602包文件与源码/site逐字节一致，wheel SHA256 `5267f9717f0e1db1c0827900dca68f95692d234ab5fab28021473f8cba7cc3f0`。5项CPU与2项GPU/Graph重验通过；真实8-rank NCCL/RCCL在各rank本地窗口1..8步时全rank得到[480,216]，七个空窗口与rank7两个大计数窗口全rank得到[2**32,6]，shape[1,2]/int64保持，process group销毁正常。[独立collective摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-expert-recording-collective-validation-20261101.json`）固定各rank结果哈希与范围；这是合成counter correctness，不是模型dispatch负载或通信吞吐。

后续模型验证使用单独的任务controller，在本人loopback服务ready后调用公开start/stop/dump recorder端点并保存响应，原始controller不改；模型参数沿dev60完整候选load-aware变体，仅追加stat recorder与明确dump目录。API/实际dump与模型结果未完成前不声明recording服务集成通过。


隔离dev61最终门禁443/443 Hook、3个resolution seam、0 errors及85项仓库测试通过，L0/Skill/guide与实际宽窄屏交互验收通过。原始Source变更函数/class AST经native globals和staticmethod搬移规范化相同；新服务与EPLB真实模型统计仍保持待验，全DAS范围未关闭。

## 74. 2026-11-01 DeepEP Graph capture/replay layout 所有权

Source runner_utils/deepep_adapter.py完整4方法新增capture_end保存LL layout、replay先恢复layout再设置dispatch mode；Source decode runner.capture_one_shape在backend实际capture完成后调用capture_end。当前owned DeepEPBuffer已有layout API但缺少这对调用链，现迁入完整Source adapter（继承原native class以保留isinstance）及完整Source producer方法，仅将未变native globals显式限定。生产者和消费者同时接通，不只修改replay或假设上一模型布局仍正确。来源注释/版权/docstring保留，不新增说明性runtime注释。CPU顺序/跨模型状态与实际Graph/缓冲区/八卡复验待执行。


实际dev61录制服务六项生成与start/stop成功，8rank均记录after all_reduce；rank0在file dump分支因搬移时遗漏time全局限定而NameError，API连接断开，服务判failed并清理。这是本轮实现缺口，保留原失败证据；修正为native N.time并新增原生_StatAccumulator.dump(file)端到端落盘测试，而不是只测_dump_to_file helper。新Graph producer测试目标类名纠正为当前DecodeCudaGraphRunner；不以Hook注册日志替代实际入口断言。


[dev61真实服务失败摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-recording-dev61-failure-20261101.json`）记录602包文件与184ebf1/wheel/site相同、六项严格生成通过、start/stop200及全部8rank after-all_reduce；rank0最终time NameError导致dump失败/连接断开，进程与端口已清理。此结果保持失败，不因生成或归约成功冒称录制集成通过。修正后的Source完整file branch与free-global检查通过；隔离dev62的5项Graph protocol CPU通过。当前603包文件与源码/site一致，wheel SHA256 `ea5404376812f4e2c93a43222067123ad4ce2075116850f0aa00404774cf86d7`，最后门禁与同profile真实服务重验继续待完成。


隔离dev62最终结果：5项Graph CPU protocol与7项recorder CPU（含public dump(file)、free globals）通过；445/445 Hook、3个resolution seam、0 errors及85项仓库测试通过，603包文件同源码/site。L0/Skill/guide与实际宽窄屏检索/阅读/菜单验收通过。新的layout控制无伪造GPU/RDMA结果，仍须同profile实际八卡Graph与start/stop/dump重跑确认；全DAS、数据集精度、实际dispatch负载/性能范围继续未完成。

## 75. 2026-11-01 dev62 录制服务结果与固定数学子集方法

dev62于10:39:39–10:45:24完成六项严格生成及本人loopback start/stop/dump，三端点均200，自有进程清空、端口关闭。603包文件与56043f0/wheel/site逐字节相同，全部8rank各43层solver计数[8]；[独立摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-load-aware-recording-dev62-validation-20261101.json`）固定版本与dump。rank0文件为int64[43,256]、非负、43层全有计数，总和4548352，SHA256为9aaf3d3b872b1471a2a40ea57dcbf8c6b72a34bcbd12448d037bbd8e35d22cd4。它是dispatch计算工作，可能含expert对齐/padding，不当成唯一语义token或性能改善。旧dev61失败记录保留。

新增一次请求、保存逐题原始数据的runner与离线scorer；模型身份/正常stop在数学提取前检查，错误和截断留在分母。冻结官方GSM8K commit与SHA256，按当前SGLang协议使用前5题prefix及后128题评分且互斥，共享库不复制整份数据。目前仅完成方法/协议测试与数据准备，真实128题和同配置DAS基线待执行，不预先声明精度通过。操作与范围见[验证指南](validation.md#固定-gsm8k-子集与原始响应验收)。

新增工具的6项契约测试与本地/容器完整91项仓库测试通过：分离数学错误与传输无效、拒绝截断/错误模型、prefix隔离与索引检查、缺失样本留在分母，以及真实loopback测试HTTP服务验证逐题原始数据和500错误不重试。L0/Skill/guide与实际宽窄屏验收通过。此次只改方法/工具/证据，hcu_sglang运行时与冻结dev62的603文件不变；数据集采集仍待执行，不将工具测试或旧六题作为128题结果。

## 76. 2026-11-01 固定128题测量与匹配对照结果

冻结dev62的load-aware/redundant8/stat功能变体完成128题：127正确、0无效、准确率0.9921875；独立离线复算一致，128份request/response/transport均齐全。[证据摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-gsm8k128-feature-dev62-validation-20261101.json`）固定603包文件、56043f0/runtime与f773a25/method、数据SHA、ordered raw hashes和清理。唯一oracle不一致sample119原样保留：expected95200、predicted99077，正常stop且有效回答；不重试、改标签或从分母剔除。该子集未建立通用准确率或DAS回归结论。

基础配方匹配对照已完成：DAS 127/128（99.21875%），plugin 126/128（98.4375%），两端均0无效响应；相差1题、0.78125个百分点。126题双方正确，sample12仅DAS正确，sample119双方错误；原始答案、模型回答及一次请求记录保留，不重试或改标签。用户明确“无需完全对齐，128题大致一样即可”，本次结果按此接受，不据此推导未来统一阈值。见[成对证据](hcu-sglang-development.md#历史-json-快照)（`hcu-gsm8k128-paired-controls-20261101.json`）。

二者使用同一基础8卡配方、canonical decode graph flag、优化环境、权重路径、数据、prefix、采样和并发；原始DAS只读且跳过general Hook，两端六项短回答检查通过、128份request/response/transport齐全、进程与端口完成清理。比较范围是声明参数与环境；另存的正常启动配置重建并非服务当时的live server_info，schema差异保留，不宣称全部有效配置相同。功能变体127/128单独报告，不混入基础匹配对照。

新增compare工具独立复算并拒绝配置不匹配，4项比较契约覆盖相同总分但不同错题、功能标志与优化环境差异及允许的输出路径差异。精度子集按用户标准接受；全DAS迁移和性能验证继续按各自范围推进。

## 77. 2026-11-01 dLLM 独立上下文编码的 Graph 批大小限制

固定DAS的base_cuda_graph_runner仅在get_batch_sizes_to_capture新增DllmConfig请求上限；现迁入完整Source函数至graph_batch，沿聚合register注册HCU AROUND，非HCU返回当前有效original。配置由原生DllmConfig.from_server_args解析，只有requires_separate_context_encoding时取pool size与配置max_running_requests的较小值；保留原先ceil alignment、captured_req_width、TBO和compile列表逻辑，不能改成简单截断既有结果。来源许可、函数注释及docstring保留，不新增说明性runtime注释。

新增真实原生配置解析与预导入decode/runner公共别名测试：Gemma4Renoise单独上下文路径、默认1请求、pool上限、alignment/width/TBO、普通decode与非separate dLLM不变、非HCU保留已有wrapper。模型配置读取用fixture隔离权重下载，实际算法和DllmConfig保留；这是CPU控制流验证，不能宣称Gemma模型GPU Graph或生成精度通过。隔离dev63产物与正式Hook/仓库门禁按实际结果补记。

隔离dev63门禁完成：604包文件与源码/site逐字节相同，wheel SHA256 b28a3dd1befcfb80e715defa8bfae0875c0907032164b44c2cb21293e0ba8f4a。6项批大小CPU测试、5项既有DeepEP Graph协议测试、446/446 Hook与3个resolution seam均通过且0 errors；本地与容器95项仓库测试、L0、Skill元信息、guide快照和实际宽窄屏交互通过。当前按原八卡基础配方启动独立服务验证普通模型回归，不重复追求已接受的128题完全对齐；服务结果另记。未验证的实际扩散模型路径保持未验证，全DAS迁移未完成。

[dev63八卡回归摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-graph-batch-dev63-validation-20261101.json`）：11:38:15–11:43:57按原基础配方完成六项严格生成，全部HTTP200、完整最终答案正确、正常stop，自有进程为空且端口关闭；604包文件与84f961e/wheel/site相同。DSPARK/DeepEP/Graph及原优化环境保持。该模型为普通解码，验证新函数在此路径没有阻断实际Graph捕获和服务；不能据此宣称Gemma4Renoise GPU分支、吞吐或全DAS迁移完成。已接受的dev62 GSM8K128对照不归用为dev63重跑结果。

## 78. 2026-11-01 Graph 初始化的 PP 预算、混合层与 H16 缓存释放

迁入固定Source cuda_graph_setup全部6个变化/新增函数：混合attention/MoE局部覆盖检查、保留尾部纯MoE的全局索引、H16捕获后同步释放、PP内存预算、完整capture_cuda_graphs与capture_prefill_graph调用方。复用native记录类型和不变helper；预算initialized状态仍写原生模块，不产生插件副本。现有4个入口为HCU AROUND，非HCU保留当前有效original；新增2个helper由owned调用方使用。Source完整函数AST在native globals限定、模块状态写入及相同HCU检测映射后一致，原注释/docstring保留，不新增说明性runtime注释。

待验：真实调用方/预导入别名、尾部纯MoE、洞/重复层、PP组与native预算状态、capture/finalize/cache释放顺序、非HCU回退。CPU fixture控制流与实际collective/GPU/模型服务分别记录，不把模拟内存查询或runner构造当作模型Graph通过。

Source控制流9项通过，包括完整public prefill/capture入口。真实4-rank Gloo在PP2时rank0/1执行本组sum=3，rank2/3执行末级TP内存MIN并由native setter得到2684354560字节；切换PP1后全rank得到3489660928字节，[证据](hcu-sglang-development.md#历史-json-快照)（`hcu-graph-setup-collective-validation-20261101.json`）固定原始输出哈希。CUDA显存查询是fixture，归约与process group真实，不冒称GPU collective或实际PP模型。隔离dev64的1项真实GPU测试通过：实际CUDAGraph捕获后调用native platform synchronize/empty_cache，两次变输入replay精确；这是allocator/小型Graph安全性，不是H16 kernel或模型验证。

[隔离dev64门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-graph-setup-gates-validation-20261101.json`）完成：605包文件与源码/site相同，wheel SHA256 b3d040c862457b000406b574096bc2cf52ac6980045343f5c0fb7dfc9091cd58；450/450 Hook、3个resolution seam、0 errors，9项新控制流、6项既有批大小与本地/容器95项仓库测试通过。L0、Skill元信息、guide快照及实际宽窄屏检索/阅读/菜单验收通过。实际八卡原配方服务仍按独立run继续，未将其提前标为通过；PP/hybrid/H16模型及全DAS迁移范围保持未完成。

[dev64八卡服务摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-graph-setup-dev64-validation-20261101.json`）：11:52:23–11:58:00按原DSPARK/DeepEP/Graph基础配方完成六项严格生成，全部HTTP200、完整答案正确且正常stop，自有进程为空、端口关闭。605包文件与4873fea/wheel/site逐字节相同；实际预算日志24条，owned HCU backend正常加载。该PP1普通模型证明其预算与捕获流程可完成，不外推PP/hybrid/H16模型；也未重复计为128题或性能测量。全DAS迁移保持进行中。

## 79. 2026-11-01 预填充 runner 与 replay 记录契约

迁入固定Source预填充runner全部12个变化/新增方法，owned类继承native以保留类型关系；完整Source构造器显式从native父类的base初始化，不重复调用旧prefill构造器。runner-owned ContextVar scope完整迁入，capture与replay上下文成对接入；原生LogitsProcessorOutput的dataclass子类追加Source三项draft字段，保留官方已有字段与复制生命周期。ForwardBatch沿已有owned schema使用encoder replay元数据。旧官方AttentionBackend缺少的可选prefill_cuda_graph_default_max_seq_len通过getattr(...,None)读取，对应Source基类声明默认值，不写冻结配置或动态补全seed。

变化包括后端默认max_seq_len与本地/DP共用准入、DSA的BCG前缀例外、预热AITER先于backend构造、capture前model prepare、load_batch保留encoder replay字段、V4.1 replay input IDs、MegaMoE valid-token刷新、trim保留draft top-k，以及Full replay等待KV层加载。保留来源注释/许可/docstring。构造、元数据、公开execute、scope生命周期与实际GPU/模型验证按本次证据补记，不以类替换日志代替行为验证。

配对迁入Source完整调度_local_prefill_cuda_graph_vote，使backend vote与max_seq_len在schedule与forward两端读取一致；MegaMoE仅迁入runtime selector和graph token setter，沿native对称buffer registry更新，未迁入其余standalone backend/kernel。Logits、MegaMoE及DP文件仍标in_progress。13项CPU测试通过，构造夹具首轮漏attn_tp_sequence_sharded修正后完整构造完成；GPU夹具首轮漏prefill_backend_name，补齐后隔离dev65的真实primitive Graph经公开load/execute/body/trim两次变输入通过，encoder metadata、prepared IDs、hidden states与draft top-k均验证。两次夹具失败原始日志保留，不是生产补丁或虚假模型结果。实际八卡显式BCG prefill功能变体继续执行，保留原decode/DSPARK/DeepEP优化，不提前声明服务或模型replay hit通过。

[隔离dev65门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-prefill-runner-gates-validation-20261101.json`）完成：610包文件与源码/site一致，wheel SHA256 469b7a6fee61cada806a89ac9d7167e79be67aab31f91a19c80d6a5b22c33079；453/453 Hook、3个resolution seam、0 errors。隔离13项prefill CPU、9项既有Graph setup、1项真实GPU公开execute与本地/容器95项仓库测试通过，L0/Skill/guide及实际宽窄屏交互通过。原始8卡基础配方仅新增显式breakable prefill、64/128 token buckets、固定context/sequence上限256的功能变体正在执行；它不是“原配方未变”的基线，也不能将六题服务结果推为128题重跑或DP模型Graph hit。

## 80. 2026-11-01 真实 BCG 捕获通过后的 MLP 同步投票缺口

[dev65失败证据](hcu-sglang-development.md#历史-json-快照)（`hcu-prefill-dev65-failure-20261101.json`）固定b017aeb的610包文件与wheel/site一致：八个rank均完成BCG prefill64/128捕获，但scheduler启动时旧prepare_mlp_sync_batch_raw未传Source vote新必填num_tokens，TypeError导致ready前退出，自有进程和端口已清理。这是迁移调用链缺口，旧13项CPU直调vote未覆盖生产入口；不能以捕获完成或primitive GPU测试称服务成功。

修正迁入完整Source MLP producer、同步记录的12列协议/归约、batch metadata更新与elastic拓扑校验；配套完整CPU-only CP token geometry、mixed-replica归一化和encoder replay计数helper。ScheduleBatch追加声明的global_cp_num_tokens以保留dataclass复制；MLPSyncBatchInfo继承native并迁入全部方法。HCU CP-v2环境字段保留Source默认False，不写冻结seed；缺少可选backend folds属性按Source基类默认False读取。真实公共prepare入口新增修复前反例，Source同步组/CP行宽与消费路径按CPU/真实Gloo/模型分别验收。此阶段不宣称剩余CP模型/worker carrier与整个DP文件完成。

Source的8个完整同步函数/方法与7个完整计数/拓扑helper经native globals、owned import及相同默认属性读取规范化AST一致。新增6项CPU通过，保留旧公共producer缺参反例，新公共producer将3+128=131折叠行传给vote与batch metadata；记录12列、ScheduleBatch复制/pickle通过。[真实四进程Gloo](hcu-sglang-development.md#历史-json-快照)（`hcu-prefill-sync-collective-validation-20261101.json`）覆盖CPU gather+broadcast、world all-reduce、DP2/CP2混合布局归一化及子组[1,2,3]的global root1。dev66的612包文件与源码/site相同；同一八卡BCG配置正重跑，失败dev65证据不改写，GPU primitive旧dev65结果保持其artifact归属。

[隔离dev66门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-prefill-sync-gates-validation-20261101.json`）：612包文件与源码/site相同，wheel SHA256 0f720444c16cd2ab081c8eee768c29ba950c99dc5d118893fc3d25cbdbc702e2；458/458 Hook、3个resolution seam、0 errors，6项同步CPU、13项prefill CPU及本地/容器95项仓库测试通过。Source Gloo证据字节与本产物相同，旧dev65 primitive GPU结果没有改记为dev66重跑。L0/Skill/guide与实际宽窄屏检索/阅读/菜单通过；相同BCG feature服务结果仍待完成。

[dev66同配置八卡BCG服务摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-prefill-bcg-dev66-validation-20261101.json`）：12:35:06–12:40:58完整通过，全部8rank完成目标prefill breakable64/128捕获，随后decode/verify捕获与六项严格生成通过，HTTP200、完整最终答案和正常stop均满足；自有进程清空、端口关闭。612包文件与c61fb5c/wheel/site相同。与dev65失败相比仅运行时修正及输出标识/端口变化，配方和BCG feature参数一致；没有关闭prefill、DSPARK、DeepEP或原优化来回避问题。该证据证明实际模型capture初始化与服务回归，不证明请求实际命中BCG replay、DP混合请求、CP/elastic模型、128题重跑或性能。旧dev65失败保留，全DAS迁移未完成。

## 81. 2026-11-01 Encoder SWA replay 完整折叠与 worker 生命周期

迁入Source完整encoder_swa_replay模块（FoldedExtend、批量legacy replay、fold/drop、window reset、Engram history与执行计数），并迁入完整TpModelWorker.forward_batch_generation方法，将新return_hidden_states_before_norm参数、折叠后FB生产、late-tail行号回映射及返回原调度KV plan成对接通。复用native未变GenerationBatchResult与其余worker方法；HCU AROUND，非HCU保留effective original。Engram history_row完整Source方法以显式receiver helper复用，避免给native类注入未声明方法；其余Engram变化仍待迁移。原计数模块导入唯一owned replay_len/folded_replay_rows，避免重复owner。保留来源注释/docstring/版权许可；旧backend缺可选folds属性按Source基类默认False。

本阶段验收从公开worker入口检查折叠行生产、隐藏态/drop消费、return-kv plan身份和legacy先重播再extend顺序。ForwardBatch初始化/模型计算的昂贵fixture与真实张量/GPU/模型分开记录，不把源AST或Hook计数解释为cache-hit模型精度。

8项CPU通过：137行fold回5个scheduled rows、batch未变、2个legacy replay合成1次forward、dense/late-tail行回映射、clamped Engram history、公开worker fold与非fold两种分发，以及原调度plan身份。隔离dev67的真实GPU张量与primitive CUDAGraph在变输入后dense/late-tail映射精确。源码14个encoder符号、完整worker方法与Engram history_row经native globals/owned import/receiver/docstring缩进规范化一致。调度PrefillAdder预算和模型feature validator仍有Source差异，不将本阶段声明为整个encoder feature链已完成。

预检发现旧snapshot_control_resolution脚本只创建seed/读resolved_view，未调用resolve_once；已修正历史GSM8K共享分析范围，保留原始种子差异及分数，不把它解释为effective/live配置。新的显式resolve_once对当前0731 Flash配置拒绝encoder replay：requires DeepSeek-V4.1。model-query另找到匹配model_type=deepseek_v41的V4.1-Flash，配置与容器可见性已读验；完整模型feature仍待其合法参数与预算迁移验证。当前八卡BCG回归使用原0731模型且未启用encoder flag，不冒称encoder cache-hit服务验证。

[dev67分范围门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-encoder-swa-gates-validation-20261101.json`）：614包文件与源码/site一致，wheel SHA256 b741efaa6649b133f4961d6147a47c206735bf8a222da56914044eead37b268e；460/460 Hook、3个resolution seam、0 errors，8项encoder CPU、6项同步CPU、1项真实GPU映射Graph及本地/容器95项仓库测试通过，L0/Skill/guide与实际宽窄屏验收通过。实际resolve_once对V4.1-Flash进一步拒绝当前硬件（CUDA/gfx950限制），与Source保留条件相同；没有绕过硬件/DP约束。当前HCU的完整encoder feature服务仍未验证，普通BCG服务回归另记，不将算法GPU primitive通过外推为该模型能力。

[dev67普通八卡BCG回归](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-prefill-bcg-dev67-validation-20261101.json`）：13:01:45–13:07:31通过，8rank完成prefill捕获、六项严格生成全部HTTP200/正确完整答案/正常stop，自有进程为空、端口关闭。614包文件与cb69307/wheel/site相同。此run明确encoder flag为False，只验证新worker普通生成路径与既有BCG回归，不称encoder cache-hit或兼容硬件模型验证；V4.1预检限制和下一阶段PrefillAdder预算/feature validator差异保留。旧128题分数与其artifact不归用到dev67。

## 82. 2026-11-01 Prefill replay 预算与 feature 校验

迁入Source全部11个变化/新增PrefillAdder方法及冻结admission记录，复用native构造与不变helper；compute预算按page-ceiled或exact原始extend加replay行，KV预算只收page-ceiled extend及headroom。首次命中chunk先扣replay，idle容量失败才defer staged，retraction完成storage admission accounting。配套三个memory budget的完整Source check_prefill接口与Req辅助SWA属性/needs-load-back读出；固定native staged plan仅有SWA，无AUX窗口，明确按0 AUX解释，保留原plan构造ABI，窗口plan生产/传输继续pending。

Source完整features校验和最小128+page预算迁入，保留CUDA/gfx950/HIP/DP等限制及decoder auto声明。源码证明handle_model_capability_adjustments除末尾预算校验外与native AST相同，沿现有具名seam先调用previous再追加Source尾部，并保留INSTANCE早退；不复制旧分支覆盖其他合法适配，不修改冻结seed。实际CPU预算、公开admission与resolve_once配置验证按本轮证据记录，未将gfx938模型encoder feature限制绕过。

隔离dev68的8项预算与5项校验CPU测试通过：旧公开adder将150预算扣成-10、提交25行extend的反例保留；新实现只提交16行extend并收128 replay，chunk余6，KV仅32。143预算不提交，144预算提交首个16-token页并耗尽chunk；exact与DLLM compute分别扣原始/ceiled extend加replay。三种native预算新接口、idle deferral、retraction storage完成及legacy/AUX读出通过。实际具名resolution dispatcher确认previous先执行、重复注册不叠加以及128+page拒绝逻辑；未mock掉预算reader。18个完整Source方法/函数及admission记录AST规范化一致，native capability body除Source尾部外一致。

[dev68分范围门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-prefill-admission-gates-validation-20261101.json`）：619包文件与源码/site一致，wheel SHA256 458c34789eab6969fd1d6b91969dc50dd5218670e6c7d387523438afc548c4c9；466/466 Hook、4个具名resolution seam、0 errors；8项预算、5项校验和本地/容器95项仓库测试通过，L0/Skill/guide与实际宽窄屏通过。实际V4.1 resolve_once在encoder-off下成功，decoder auto=False、per-rank chunk4096、attention DP8/replica DP1；encoder-on仍按Source硬件约束拒绝。另捕获未迁移的V4.1 defaults覆盖显式sparse-prefill问题，不将该预检声称为优化配置交付。普通8GPU BCG回归结果另记；当前改动为CPU调度/接口/配置，未冒称新的GPU kernel数值验收。

[dev68普通八卡BCG回归](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-prefill-bcg-dev68-validation-20261101.json`）：13:46:10–13:52:01通过，全部8rank完成prefill捕获，六项严格生成HTTP200/完整答案正确/正常stop，自有进程为空且端口关闭；619包文件与dd88288/wheel/site相同。encoder flag保持False；该结果验证本次Req/adder/memory-budget接口安装后的普通服务，不证明encoder cache-hit、AUX窗口传输、请求级BCG命中或128题重跑。下一阶段继续迁移V4 defaults/CP和窗口plan生产/传输，全部DAS目标未完成。

## 83. 2026-11-01 V4 attention 默认值与 HCU CP 策略

迁入默认attention后端选择与apply_deepseek_v4_defaults完整Source实现，统一既有Source CP校验所有权；与已有features/budget校验合并覆盖本文件全部变化。显式开启的HCU sparse prefill保留，未显式开启仍按Source HIP默认关闭；auto后端与max-running经声明发布，mega/TRT约束不放宽。CP使用Source HCU规则与统一拓扑声明，保留非HCU/zigzag/TP范围及A2A约束。原有model-specific兼容层保留其余tilelang/FP8开关职责，未以改种子或关闭优化代替迁移。GPU内核未改，Source策略CPU、实际resolve_once环境值与普通八卡回归按本轮证据分开记录。

收敛注册审查：CP校验此前已由current_v4/hcu_backend/arg_groups迁入并注册REPLACE。本轮dev69原型新增AROUND与其同语义叠加，正式报告467个唯一目标揭示合同列表有重复；不将重复计为新能力。最终dev70保留原REPLACE入口，legacy模块只重导出统一v4_defaults函数，去除快照globals副本与新增AROUND，并去重合同。Source策略算法不变，新增别名/单一注册断言；dev69隔离数学run仍使用其冻结原型，结果与最终dev70产物分开记录。

实际解析对照：dev68的0731输入sparse env=1、resolve_once后为False；dev69对0731及V4.1输入/输出均为1。因此旧服务配置中“原优化env保留”只代表声明参数，不能证明此开关有效开启；旧原始分数/日志保持原样。新稀疏路径在原始plain8卡配方上重新固定128题，旧DAS对照与dataset不变且不选择性重试。dev69原型无对应runtime提交，使用不可变build snapshot/wheel/site证据；最终dev70集中CP所有权，实际CLI/门禁/服务分别记录。

[固定128题新稀疏路径测量](hcu-sglang-development.md#历史-json-快照)（`hcu-gsm8k128-sparse-default-dev69-validation-20261101.json`）：冻结dev69原型127/128、0 invalid，384份逐题request/response/transport齐全；独立scorer复算与保存的plain DAS控制组逐题结果相同（127双方正确、1双方错误），没有选择性重试或改标签。比较范围仍为声明参数与env，不声称完整effective配置或权重哈希一致。原型620文件与build snapshot/wheel/site相同，但runtime无对应Git提交，保留其准确归属；最终dev70策略函数AST相同，差异为CP注册收敛，未冒称dev70重跑128题。

[最终dev70门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-v4-defaults-gates-validation-20261101.json`）：620包文件与源码/site相同，wheel SHA256 be336433b63eee297ee126cfcf111d758fbb2076f4301d22bf3ad74b6bc16d33；467/467唯一Hook、4个seam、0 errors。9项默认/CP/alias测试（纠正REPLACE函数由框架包装，不能断言公开wrapper identity）、5项既有feature校验和95项仓库测试通过；0731与V4.1真实resolve_once前/后sparse均1。L0/Skill/guide与实际宽窄屏交互通过。CP/SM100/mega实际模型仍未验证，最终dev70八卡普通回归继续，窗口plan生产/传输等全量DAS范围未关闭。

[最终dev70普通八卡服务](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-sparse-default-dev70-validation-20261101.json`）：14:15:35–14:21:18通过，六项严格生成全部HTTP200/完整答案正确/正常stop，自有进程为空、端口关闭。620包文件与32fb304/wheel/site相同，实际server日志确认保留显式sparse开关；原plain配方的DSPARK/DeepEP/decodeGraph/并行与权重加载保留，未加CP/BCG变体参数。此结果属于最终dev70普通集成，128题127/128仍准确归属于隔离dev69原型；不是CP/SM100/mega模型或吞吐验收。全DAS窗口plan生产/传输及其他pending模块继续未完成。

## 84. 2026-11-01 双窗口计划与树修复后端

作为窗口pipeline迁移前置，迁入完整WindowLoadBackPlan/StagedPrefetchPlan owned记录，暂不替换固定native staged plan的swa_tokens构造ABI；已有Req AUX reader可消费owned windows。迁入Python TreeCore全部3个窗口查询/attach/segment变化方法，按component_type分别识别SWA/AUX tombstone、拆分真实树节点、设置独立数据与LRU/ledger。Rust两项窗口方法按Source仅允许SWA，AUX明确拒绝；沿现有get_tree_core_factory扩展点保留原factory，在实际选择并创建native Rust实例后绑定新方法，不在插件加载时强制编译可选Rust。自定义工厂/子类保留其原行为，非HCU effective original保留。

此阶段不改变pipeline producer与H2D发布，避免新plan先替换旧构造字段而破坏现有caller。测试保留真实TreeCore构造、native组件分片、节点arena与LRU，不mock _split_node或ledger setter；storage/H2D/完整窗口cache模型范围继续pending。

8项CPU通过，保留旧API不接受component_type的反例；使用真实TreeCore构造/native组件split/LRU与账本，验证AUX[2,6)拆成2/4/2三个节点、FULL/SWA字节不变、AUX缺失[0,2)/[6,8)、protected/evictable归属以及pending backup split action。两轮夹具分别补齐只读allocator内部字段与cache session开关，未修改生产算法。隔离dev71的1项真实GPU确认clone设备/存储分离及独立SWA数据；实际native Rust CPU经工厂选中、insert/局部attach/补齐/SWA ledger8通过，AUX明确拒绝。Rust采用原始受锁定Cargo/native来源，输入sglang Git工作树检查仍干净。stage/producer/H2D端尚未迁入，因此owned记录暂未替换native schema，也不声明完整窗口缓存模型通过。

[dev71分范围门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-window-tree-gates-validation-20261101.json`）：623包文件与源码/site一致，wheel SHA256 28b13d4a7d31e45f6b0ac8418cb81fe2f8439e40751ba3f844cc3654e7779344；471/471 Hook、4个seam、0 errors。8项窗口CPU、8项既有预算、本地/容器95项仓库测试通过；1项真实GPU独立数据/clone测试及实际Rust CPU工厂+binding测试通过。5个窗口方法与2个完整Source记录AST规范化一致，Rust加载延迟仅为适配方式变化，不改其SWA算法或AUX guard。L0/Skill/guide与实际宽窄屏检索/阅读/菜单通过。普通八卡回归继续，不将无hicache参数的服务用于证明storage/H2D或窗口cache-hit。

[dev71普通八卡回归](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-sparse-default-dev71-validation-20261101.json`）：14:53:32–14:59:14通过，六项严格生成全部HTTP200/完整答案正确/正常stop，自有进程为空且端口关闭；623包文件与2cb0f1b/wheel/site相同，显式sparse保留。该run未加入hicache/buffer/storage参数，只是新树/factory接口安装后的普通集成，不证明window plan生产、H2D发布或缓存命中；这些下一阶段范围仍pending。原型dev69的128题保持其独立产物归属，不挪作dev71测量；全部DAS目标仍未完成。

## 85. 2026-11-01 多窗口 staged plan 与 H2D 发布

迁入Source BufferModePipeline全部16个变化/新增方法、完整buffer-only双窗口host fence，以及TreeComponent工厂与UnifiedRadixCache.init_hicache生产caller，使用owned windows记录替换该生产/消费闭环中的旧swa_tokens语义。Source未变allocation/queue/ACK核心保留native；缺失区间按组件切host bounce，仅为缺失行排队H2D，resident FULL范围修复树，tail随普通insert发布；完整原始bounce保留到ack。Tail-only backup沿Source False默认，keys和device rows一起裁剪并推进backend prefix context。

固定native SWA组件通过只读view补齐pool_name与完整Source restore mapping helper；已有custom字段/callback原样保留。新Source控制器prefix helper在旧基类缺方法时以显式receiver调用，有现有callback则保留其行为。Root cache retention/readonly-match、其他SWA组件与L3/linker变化仍按独立file scope待迁移，不将generic pipeline迁入解释为所有cache模型完成。测试按真实树修复/接收者与显式传输fixture分层记录。

配套新增PoolName.AUXILIARY_SWA完整Source枚举；捕获的legacy成员保持str/hash键相等，通过copyreg转换后pickle回owned成员，未修改内部Enum表。9项CPU通过：prepare生成SWA[(2,4),(6,8)]、AUX[(4,6),(6,8)]，各仅4行H2D，原三份8行bounce到ACK才释放；分配失败context回滚并保留staging，发布错误保留pending状态并fail-stop，covered窗口移除其派生sidecar。实际TreeCore repair/match保留，cache publication/allocator facade显式作为fixture。public init_hicache与base-component factory、tail keys/rows/prefix、双窗口共享arena fence通过。修正夹具固定prefix长度与原生日志/配置字段后无生产算法改动。隔离dev72 GPU fixture controller经真实pinned host→CUDA stream/event复制，逐池字节oracle及ACK释放顺序通过；不冒称native HiCacheController线程、L3或完整模型cache-hit。

[dev72分范围门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-window-pipeline-gates-validation-20261101.json`）：627包文件与源码/site相同，wheel SHA256 348a661f96d463f063aa3a4eaa41479fbe8da872f487dde9923b24408fcbfe63；476/476 Hook、4个seam、0 errors，9项pipeline CPU、8项树CPU、本地/容器95项仓库测试通过。1项GPU pinned host/stream/event H2D逐池字节与ACK顺序通过，但physical queue/allocator/cache insert facade明确为fixture，不称native controller线程或L3模型。21个完整Source方法/函数及完整PoolName枚举AST规范化一致，legacy字符串键/pickle兼容通过。L0/Skill/guide及实际宽窄屏通过；普通八卡回归继续，其他cache retention/linker/FileIO与全量DAS目标未完成。

[dev72普通八卡集成](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-sparse-default-dev72-validation-20261101.json`）：15:32:59–15:38:41通过，六项严格请求HTTP200/完整答案正确/正常stop，显式sparse保留，自有进程为空、端口关闭；627包文件与8e01b89/wheel/site相同。该run未加hicache/buffer/storage参数，只验证新枚举和pipeline接口安装后的普通集成，不是L3/cache-hit模型验证；GPU H2D使用明确的fixture controller范围，二者不合并成native controller/storage通过。后续retention、readonly匹配、cache/linker/FileIO等Source差异及完整缓存模型继续pending，全量DAS未完成。

## 86. 2026-11-01 FileIO 的 TP 状态与共享 MLA 文件归属

复现旧版两个反例：未限容量时不同 TP rank 的同一 Mamba key 读回首个 rank 的数据；开启容量限制后 MLA 非零 rank 拒绝自己的状态写入。按固定 Source 迁入完整 LRUFileEvictor 类及 HiCacheFile 的 constructor、suffix、metadata warmup 三个完整方法，Mamba 用 TP suffix，MLA KV 继续共享；每个 rank 管自己的 TP 文件，rank 0 额外管理共享文件。公共 lazy factory 解析 owned HiCacheFile 子类，15个未变化 native 方法逐一 AST 比对后继承，读写/临时文件原子替换/commit/abort/batch-v2 保留原生数据路径。

[Source 对齐记录](hcu-sglang-development.md#历史-json-快照)（`hcu-file-backend-source-alignment.json`） 检查完整类和三个方法，不靠局部语句匹配。`tests/runtime/file_backend_cpu.py` 从原生 StorageBackendFactory 构造，真实临时文件和 CPU tensor 读写，覆盖两个反例、其他 rank 文件不被淘汰、metadata 回调/重启扫描、rank 0 双 suffix 启动淘汰、PP/CP、pending 写入保护、失败回滚，以及公共 batch-v2 状态读写与 prefix 命中。batch 的 host pool 为数据夹具，backend 和磁盘 IO 均为真实实现；不将该项解释为控制器线程或完整模型 L3 命中。

精度接受条件按用户补充：128题整体与基线大致一致即可，允许少量差异；无效输出/运行错误仍需修复。原型dev69的127/128和DAS127/128保留原有产物归属，不挪作dev73的测量；本轮 FileIO 不改变模型数学路径，也不重复跑无存储参数的普通服务来证明缓存行为。其余 retention/readonly-match/controller/linker 和全量 DAS 目标继续执行。

[dev73分范围门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-file-backend-gates-validation-20261101.json`）：wheel SHA256 `33dddedb42fb0751d1e66e4c4890ef09b4a61f3a810bebe85ff6d1260ac47970`，629包文件与本轮源码及隔离site逐字节一致。10项FileIO测试、478/478 Hook、4个seam、0 errors，本地/容器95项仓库测试通过；旧实现两个反例均按预期失败，固定来源SHA/clean和文件哈希复核通过。台账将完整FileIO变化按逐文件实现记录，保留未验证的native controller线程、分布式rank进程协调和完整模型L3命中范围；不能从该记录推断全量DAS迁移完成。

## 87. 2026-11-01 Python KV 事件的 LoRA namespace

继续梳理 retention 参数时发现 CacheInitParams 新默认表依赖 KvEventLoraNames，先闭合其 Python 生产/消费。完整迁入 Source events 模块的 LoRANameTable、KvEventLoraNames、KVCacheEventRecorder 三个类；默认事件对已登记的最终 namespace 附加 hash seed，store/remove/parent 链使用同一命名，queue 合并要求 lora_name 相等。保留 publisher 生命周期内的名称，允许 GPU removal 和请求释放后发布 host store；Dynamo 格式仍走原有 adapter ID fallback 与 namespace seed。

[Source 对齐](hcu-sglang-development.md#历史-json-快照)（`hcu-kv-events-source-alignment.json`） 检查完整模块前缀字节、三类及两份完整 scheduler 方法，六个导入 hash helper 与 fixed native AST 相同。原生 public generate/embedding handler 和 elastic namespace 方法保留 original；ContextVar 在外层保存可选 IPC lora_name，namespace seam 在 original 完成后调用完整 Source 登记 helper，finally 恢复上下文。旧 IPC 无名称字段时用 Source 已登记 adapter ID 回退；不改写消息 dataclass。这里仅完成 scheduler 的 KV 名称片段，其余变化继续 in_progress。

测试保留两个 native 反例：默认不同 namespace 的 LoRA ID 冲突、不同名称事件被合并。直接构造和 take 真实 Python recorder，保留普通/Dynamo 兼容，store/remove/parent、请求消失后 host store、disabled 和 context 异常/嵌套。public handler 使用原生 ExitStack 入口与真实注册 namespace seam，昂贵内部 admission 和 elastic namespace 场景显式用数据夹具；不是完整 LoRA 模型部署。原生 UnifiedTreeCore constructor/insert/leaf eviction 的实际事件出口单独验证。修正夹具的 keyword-only 参数、session 配置及 FULL 应走 leaf eviction 的调用后，未改生产算法来迎合断言。Rust tagged binding、发布 socket/router、CacheInitParams/retention 与其余全量 DAS 目标保持待做。

[dev74分范围门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-kv-events-gates-validation-20261101.json`）：12项CPU测试、484/484 Hook、4个seam、0 errors、本地/容器95项仓库测试通过。隔离wheel SHA256 `317d4ba6d6982be22b7b75da952bef5b7ab74c2307e2821585e9a1d4c965de6f`，631包文件与本轮源码及site相同；原生Python树实际insert/leaf eviction产生的store/remove hashes一致。初轮夹具错误日志保留，最终门禁引用修正后的attempt2。后续普通八卡只用于原生scheduler请求路径集成，不替代LoRA事件、Rust或L3模型验证。128题接受“大致一致”的标准保持，本轮不将旧产物分数挪给新wheel。

[dev74普通八卡回归](hcu-sglang-development.md#历史-json-快照)（`hcu-tp8-deepseekv4-sparse-default-dev74-validation-20261101.json`）：16:09:20–16:14:58通过，六项严格请求全部HTTP200/完整答案正确/正常stop，显式sparse保留，自有进程为空、端口关闭；631包文件与14abc7e/wheel/site相同。本轮运行实际经过原生scheduler新增包装后的请求入口，未启用LoRA/KV事件/hicache/storage参数；不将普通生成称为事件路由、Rust binding或完整L3/cache-hit验证。全量DAS、retention及其依赖继续pending，128题历史结果保持各自产物归属。

## 88. 2026-11-01 Owned Rust binding 与宽 token hash

延续 Python KV 事件迁移，完整打包固定 Source 的 Rust crate（22文件，含 Cargo.lock 和测试），按完整 Source converter/recorder 消费新增带 extra_key 的 block_removed tuple。沿原生 public Rust loader 指定 owned workspace 和独立 production/inspection 模块名，保留 mode/features/cache 及显式 custom provider；实际选中原生工厂后校验 binding 身份并接入 Source 事件 recorder，旧 binding 不能与新 ABI 混用。Native capability/custom fallback 保留，只有源码/工具链可用性才允许由插件 workspace 满足；部署不依赖输入 DAS 路径或修改官方树。

Source Rust 同时扩大非负 token 到 int64 域，因此成对迁入完整 Source C++ hash 和 Python 输入转换；普通 u32 页面编码保持原样，宽页面使用0xff tag及little-endian u64，前缀hash与bigram按同一规则连接。C++扩展独立命名，native get_native_hash caller/alias 经正式 Hook 接入。Torch兼容header与rpath使用插件资产，额外编译flags保留，加载可选Rust时才构建；Cargo继续使用--locked。详见 [Source 对齐](hcu-sglang-development.md#历史-json-快照)（`hcu-rust-events-source-alignment.json`）。

dev75为冻结原型：659包文件，真实Rust store/remove与Python recorder及宽hash对照已执行，但未完成owned workspace的backend预检；原型测试还需消费eviction返回的CPU索引。dev76补齐预检后独立冻结，旧原型不挪为最终wheel证明。初轮函数对象identity断言不适用于native REPLACE包装，已改为实际caller alias及数值/事件行为；传输AppleDouble元数据单独清除，未用其解析错误改生产逻辑。CPU夹具会断言并消费返回的FULL索引，原有Python树测试也补齐同一ownership要求。

本轮验证范围是Rust CPU binding、事件/hash与源码/包装契约；没有运行新的八卡模型、Rust GPU/L3/LoRA路由/retention测试，不沿用旧模型分数充当本轮产物测量。用户要求本轮任务收尾后停止目标运行；完成当前验证和本地提交后暂停，完整DAS目标的未完成项保留。

[dev76分范围门禁](hcu-sglang-development.md#历史-json-快照)（`hcu-rust-events-gates-validation-20261101.json`）：659包文件与源码/wheel/site相同，wheel SHA256 `234671c432573204f83d94558dd348f5dd0dc48f51b173fbf1d64a9c6d82cfe7`。487/487 Hook、4个seam、0 errors；10项provider、12项KV事件、本地/容器95项仓库测试通过。真实Rust auto及never缓存复用均通过6组default/Dynamo/namespace事件、12组Rust/C++/hashlib hash、4次负数拒绝及8行SWA恢复；两次binding SHA相同，缺失官方manifest时仍选中owned Rust，返回FULL索引已断言并消费，无ignored destructor错误。完整crate执行`cargo test --locked --release --lib`为930 passed/0 failed/1 ignored；ignored为Source原有不可构造的colliding-page panic用例，未新增skip。固定输入SHA/clean和全部Source资产哈希通过。没有新增八卡服务或数据集测量；本轮收尾后按用户要求暂停目标，剩余DAS工作不标记完成。

提交前的diff检查去除owned events.py末尾多余空行，最终另行冻结dev77：wheel SHA256 `b989298e06102ef3410d1f8836a5ad42cc64bebb4c4239e15ddcba5d0b2aee39`，659文件与源码/site一致。与已测dev76逐文件比较仅该EOF换行不同，其他658包文件完全相同，pyproject仅变版本；dev76门禁保留原归属。dev77另行通过487 Hook及真实never模式全部事件/hash/负数/SWA合同，binding SHA与dev76相同；没有将空行修正描述成算法变更，也没有覆盖原冻结产物。最终本地提交完成后暂停目标。

## 仓库命名与文档清理（2026-11-01）

按 plugin-v3-1101 需求统一需提交文本中的 HCU 命名及大小写，重命名迁移 Skill，移除已由 HCU MLA 模块替代的旧兼容文件。Git 忽略的来源树、环境、构建产物和原始日志不参与替换。

删除 docs 下 95 份不被工具直接消费的 JSON 历史报告和清单，文档索引转到历史快照说明；plans 中的历史证据采用 `git:<提交>:<路径>` 定位。当前工具输入与目标合同继续保存在 plans、templates 与 tests/contracts。重建两份网页；计划签名随纯命名替换刷新，仍保留原采集基线与历史验证语义，本次没有重新采集来源或重做 GPU 验证。

### 历史 JSON 快照

删除的 JSON 可从清理前提交读取，不依赖远端是否已推送。例如在仓库根目录执行：

```bash
git show de390225d615381d3fd02b8d3eed639e5ef529b5:docs/hcu-sglang-targets.json
```

其他历史报告以引用处给出的文件名替换命令中的文件名。清理前工作区备份和本次检查日志位于 ignored `artifacts/plugin-v3-1101-cleanup/`。本次验证结果在提交前记录；历史模型、GPU 和性能结果不视为本次复验。

### 本次检查结果

在指定容器、Python 3.10.12 下执行：L0 通过（474 个 Python、8 个 Skill）；95 项 unittest 通过；229 处 Markdown 锚点核对通过；网页导览、迁移计划快照和执行台账一致性检查通过；需提交文件的旧硬件命名残留为零，docs 下 JSON 为零，Git diff 空白检查通过。

容器默认 site-packages 中的 SGLang 缺少 `ReplicatedParallelGroup`，不满足本仓固定接口；显式使用固定官方 `baa9befa38` 的 python 目录后，10 项 current_linear CPU 回归通过，真实入口加载注册并应用 487/487 个 Hook、4 个具名 resolution seam，加载错误为零。未安装、更新或修改官方源码与容器依赖。

本轮为命名及文档清理，没有执行模型、GPU 数值或性能复验。界面控制权限未开启，网页实际浏览器截图、交互和窄屏视觉验收未执行；生成快照与 Markdown 路径/锚点检查不能替代该视觉验收。

## DAS 分支逻辑对齐（2026-11-02）

需求为 plugin-v3-fix，本节汇总固定来源的模块处置、验证范围与未完成条件；原独立对齐记录已整合到本开发记录。提交按模块组成同一变更系列，存在模块间依赖，使用完整系列进行集成验收。来源只读；仅本地提交。

### 公共运行时与调度

修正 HIP 检测及 FP8/provider 分支，补齐环境变量与 ServerArgs 派生选项；迁入 Req/ScheduleBatch/ForwardBatch 生命周期、CP 和 DSA helper，显式接通冻结配置、元数据和 hidden state 消费者。核心单元 9 项、生命周期 7 项、Forward/CP 13 项及既有 CPU 调度组 64 项通过。未执行完整模型服务；具体 GPU/配置边界见本轮记录与测试。

### 计算与量化

迁入 MegaMoE、DeepEP INT8、Marlin 与 DP 数据流并修正 routed scaling、SlimQuant 返回契约。29 个 runtime 脚本 197 项、13 项针对性单元和 GPU W4A8 3 项通过；deepgemm.mega 完整数值/Graph 未验证。

### Attention 与模型

补齐 MLA/FLA/GDN、Qwen/Bailing/GLM 与多模态调用，保留上游接管及正确 KV/通信归属。CPU 15 项、FLA GPU 4 项通过；完整模型与所有 provider 未验证。

### 缓存与通信拓扑

修复容量、LightOP、SWA/DSV4/MLA、cache builder/retention、DP/CP group 和 prefix 协议；接入完整 LayerSplit page plan producer/consumer。allocator 11、topology 10、prefix 6、LayerSplit CPU 2 项以及真实 Gloo 进程组通过；GPU scatter/gather 在 LayerSplit 测试中用 CPU 数学夹具。

### KDA 配置、池与执行

成对迁入配置生产、原生类型身份、Mamba slot/ring 语义、KDA backend、状态 kernel 与 verify/commit/rollback。配置 7 项、池 CPU 4 项、HIPGraph replay 及 KDA GPU 5 项通过；全模型尚未测试。

### PD 与 DSpark

统一 hidden state 所有权，补齐 prefill/decode/scheduler/ACK/drain/清理、描述符传输与 DSpark worker/model/injector。transport 8、frontend 7、HiCache 5、DSpark 6 项通过，覆盖真实 HTTP/ZMQ 与受控 CPU 字节复制；完整多进程 PD 服务和 TP GPU 传输未验证。

### 聚合、打包与最终门禁

根注册链接通配置、CP/DSA/ForwardBatch、KDA 和 DSpark，package-data 包含 HIP 状态索引源文件；当前 target 合同迁到 tests/contracts，修复安装与 Skill 中失效的旧 JSON 路径。117 项仓库单元全部执行通过；L0 556 Python/8 Skill、226 个锚点及三份生成页面/台账快照检查通过。源码加载 597/597 Hook、4 seam、0 errors；721 个 wheel 包文件与最终源文件相同。具体 SHA 与验证边界见本轮记录。未部署、未推送，完整模型与缺失 vendor provider 的 lane 保持未验证；网页实际视觉检查受界面权限限制未执行。

## 2026-11-02 当前插件内嵌源码生成

按 hcu-plugin-backtrack Skill，将当前插件与固定官方 `baa9befa381511cd4d56f83980951e5a45fffc6d` 组合为 `output-backtrack` 独立 Git 源码树，输出分支为 `embedded-hcu`。原插件存在未提交的文档整理，故在 ignored `artifacts/backtrack-20261102/plugin-input` 保存完整工作区快照；生成输入提交为 `00a9be2c935d0b237fbd7f06ed53f57e40ffa6e0`，运行时文件逐字节等于原插件。没有在原仓库自动提交或清理用户变更，原两输入 HEAD/索引哈希不变。

生成器 v5 补齐 KDA HIP、KV C++、Rust crate 和 AITER 配置的 package-data。输出复制 721 个插件文件，726 项清单 SHA256 核验通过；314 个非 Python 包资源均被打包规则覆盖。上游受跟踪修改只有 bootstrap、插件 loader 与 pyproject，来源/许可另由生成器清单记录。

在仓库外、仅输出 python 的 PYTHONPATH 下屏蔽全部外部 entry point，并清空 SGLANG_PLUGINS，验证 `sglang` 和 `hcu_sglang` 均来自内嵌输出；597/597 Hook 全部应用，4 个具名 seam、0 errors。工具仓库 117 项单元测试通过，其中 13 项需显式开启的层运行时测试未执行。本轮没有构建输出 wheel、启动模型服务或进行数值/性能验收；只验证源码生成、资源规则和隔离加载。原始证据保存在 `artifacts/backtrack-20261102/`。

## 附录：布局、操作方式与汇总

以下为布局说明、使用方法、维护规范及跨阶段验证汇总，不作为单独时间事件排序；表内版本和历史结果保留原有含义。

## 1. 当前布局与接口

| 对象 | 当前定义 |
|---|---|
| 代码目录 / Python 包 | `hcu_sglang/` / `hcu_sglang` |
| distribution / 开发版本 | `hcu-sglang-plugin` / `0.3.2.dev62` |
| general plugin entry point | `hcu = hcu_sglang:register` |
| 显式服务入口 | `python3 -m hcu_sglang` 或 `hcu-sglang-serve`，参数传给官方 launcher |
| 来源插件版本 | `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c`，原包 `sglang_dcu_plugin` |
| 官方接口基线 | `baa9befa381511cd4d56f83980951e5a45fffc6d`，SGLang 0.5.22 |
| 来源清单 | [逐文件迁入快照](hcu-sglang-development.md#历史-json-快照)（`hcu-sglang-import-manifest.json`），用于本次迁入审计，不是未来文件完整性锁 |

复制来源仅为输出仓的运行时包。没有复制外部 docs、tools、tests 或其他生成脚本。包内 7 份不参与运行的来源说明 JSON 未放入运行时目录；源路径、哈希与排除理由保存在迁入清单。保留原生资源、继承的注释/docstring、版权与 vendor LICENSE。

原来的 `usercustomize.py` 位于来源包之外，未作为全局 Python 启动文件迁入。新增包内显式 launcher 先安装同一惰性 AITER shim，再转发官方服务；普通 Python 子进程及冷导入不自动加载 Torch/AITER。

## 3. 当前安装与验证方式

先准备匹配的官方 SGLang、Torch/HCU、DTK 和 vendor 依赖；本次 wheel 不代替该环境。按 [配置管理](configuration.md#1-当前工具的路径配置) 提供真实本机路径，安装时使用隔离环境或 task target，避免同名旧入口重复加载：

```bash
# 从本仓根目录执行；仅安装当前插件，不补齐 vendor/HCU 依赖。
python3 -m pip install -e . --no-deps --no-build-isolation
export SGLANG_PLUGINS=hcu
# 追加本次模型、设备、TP、backend、量化与 Graph 等完整配方。
python3 -m hcu_sglang --help
```

严格加载工具默认保留旧 Python API/命令兼容；验证当前包时显式选择命名空间：

```bash
: "${OFFICIAL:?请设置固定官方源码路径}"
: "${ARTIFACTS:?请设置本次证据目录}"
REPO="$(pwd -P)"
python3 tools/validate_plugin_runtime.py \
  --official-python "$OFFICIAL/python" --plugin-root "$REPO" \
  --plugin-package hcu_sglang \
  --expected-targets tests/contracts/hcu_source_alignment_targets.json --output "$ARTIFACTS/hcu-runtime.json"
```

记录实际解释器、entry point、模块文件来源、目标集合与具名 seam、加载软错误，以及是否误导入旧包。当前 entry point 名为 `hcu`，由固定官方 loader 对 `SGLANG_PLUGINS` 中的名称精确匹配；不是 Python 包名后缀自动扫描，也不保留 `hcu` 别名。升级后须重新安装本仓或新 wheel 以刷新 metadata，再设置 `SGLANG_PLUGINS=hcu`。安装其他插件时同样核对实际白名单和模块来源，不能凭入口存在或目标数量认定通过。源码、隔离 wheel 和完整服务分别验收。

## 5. 后续代码开发与记录规范

每次代码改动追加台账，写明完成模块/行为、源码或前置提交、提交 SHA/标题及文件范围、实际执行的验证和原始证据、未验证模式、已知限制与下一步。提交后的代码 SHA 可在后续记录提交中回填，避免用文档自身未产生的 SHA 伪装已提交。

当前方法抽取、差异清单、处置、签名检查和 embedded 生成已支持当前命名空间。方法 profile 与实际类拆分仍单独预检；不因包名支持而自动执行会覆盖既有适配的方法同步。

## 当前八卡验证汇总

本轮四类代表服务已按各自原八卡配方完成六个严格请求；每行属于对应隔离wheel，不能外推到其他版本或全量DAS。汇总哈希与运行ID见[结构化摘要](hcu-sglang-development.md#历史-json-快照)（`hcu-eight-gpu-validation-20261101.json`），原始请求/响应/日志保留在指定test-model目录。

| 服务 | 实际版本 | 并行与关键模式 | 请求/清理 |
|---|---|---|---|
| Qwen3-235B-A22B-Instruct BF16 | dev34 | TP8、fa3、内存0.85、默认Graph | 6/6 stop，自有进程为空、端口关闭 |
| MiniMax-M2.5 BF16 | dev35 | TP8、FP8 KV、context131072、默认Graph | 6/6 stop，自有进程为空、端口关闭 |
| GLM-5 INT4 | dev37 | TP8、EAGLE、FP8 KV、AITER IPC、auto→LightOp | 6/6 stop，自有进程为空、端口关闭 |
| DeepSeek-V4 FP8 | dev37 | TP8/DP8/EP8、DSPARK、DeepEP、context32768、graph32 | 6/6 stop，自有进程为空、端口关闭 |

全量代码/依赖审阅仍在进行，不把四类代表服务或文件标签数记作整个DAS已完成。

## 2026-11-02 PR #4 审核修复与目标分支整合

恢复历史 wheel 的 `sglang_dcu_plugin` 文件名、原入口及版本归属，保留历史 SHA256，不把当前 `hcu` 命名追溯为旧产物重建。修正 `group_concurrent_contiguous` 的返回注解为两组 Python 整数列表，保留 `.tolist()`、空输入与长度校验。失效的独立对齐记录链接改到本记录已有章节。

合入目标 develop 的既有提交，保留其命名调整和 adapter 姊妹仓文档；冲突处采用本 PR 较新的实现，保留已整合或明确删除的旧文档状态，重新生成导览。目标分支运行时改动已核对仅为命名替换；本轮不回退 PR 的新增运行时功能。

已核实的误报不作修改：盘点 JSON 实际受 Git 跟踪；hidden chunk 异常由 decode 队列捕获并标记单请求失败。本轮修改不声称完整模型或 GPU/PD 服务重新通过。验证结果随本次 PR 修复提交记录。

本次复验：L0 的 556 个 Python 文件和 8 个 Skill 通过；117 项仓库测试中 104 项通过、13 项需专用运行环境而跳过；243 个 Markdown 锚点、导览/迁移规划/执行台账快照及 diff 空白检查通过。已检查实际桌面、390px 窄屏渲染和文档跳转。未执行专用层运行时、GPU 或完整 PD/模型服务复验。

## 2026-11-02 fa3 混合线性模型 paged KV 池判定修复

完成模块/行为：`hcu_sglang/layers/attention/flashattention_backend.py` 的 `around_get_paged_mha_kv_cache` 钩子补齐混合线性注意力模型形态。此类模型（qwen3_5 等）的 `backend.token_to_kv_pool` 是上游 `HybridLinearKVPool` 聚合池，HCU page-major 池挂在其 `full_kv_pool` 属性上；原判定只认裸 `HCUMHATokenToKVPool`，混合形态下判定不成立、放行上游 `get_paged_mha_kv_cache`，上游按 HND 校验拒绝 HCU 转置 V 布局，prefill 图捕获首个全注意力层 warmup 即断言退出。改动后直接形态或 `full_kv_pool` 内池形态均可认出；取数仍走聚合池 `get_kv_buffer`，保留其全局层号到全注意力稠密层号的映射，不能用内池直接取数。非 HCU、纯 MHA 模型、`SGLANG_KV_LAYOUT_HCU_FA=false`（内池为上游布局）三种场景判定与放行行为不变。

问题来源：dev34 期即发现（B4 崩于 `paged_kv_view` HND 断言）；当前 develop 复现同签名，属一直未覆盖的混合形态，与仓库内既有 `full_kv_pool` 解包写法（disagg/utils.py、disagg/prefill.py、mem_cache/cache_builder.py 等）对齐。

实际验证（Qwen3.6-27B，TP2 gfx936，fa3，page_size=64，develop + current_dsa 导入修复，本机日志 dev77-fix1/fix2/fix3 对照）：HND 断言由每次启动 3 次降为 0 次；prefill 图捕获 58 个形状全部完成（修复前死于首个形状 warmup 后约 2 秒），decode 图捕获 bs=1/2/4/8/12 全部完成；修复后服务推进至下一独立死点（disagg 初始化调用上游已移除的 `Scheduler._init_waiting_queue_prefetch_runtime`，AttributeError），与本修复无关，待单独处理。

未验证模式：纯 MHA 模型回归（判定为等价改写）、SWAKVPool 聚合形态、PD 分离模式、`SGLANG_KV_LAYOUT_HCU_FA=false` 显式组合、八卡代表服务复验。

已知限制与下一步：disagg/scheduler.py:93 对上游 Scheduler 方法名的假设需加守卫或对齐新名；混合模型状态池并发上限（mamba 槽位）与 KV 配比未调优。

## 2026-11-02 disagg 调度钩子对官方 sglang 缺失方法的 setdefault 兜底

完成模块/行为：`hcu_sglang/disagg/scheduler.py` 的 `register_hooks` 注册期对 `Scheduler` 做方法级 setdefault。b4ae852 引入的 `init_disaggregation` 整体替换钩子在一切 disaggregation_mode 检查之前无条件调用 `self._init_waiting_queue_prefetch_runtime()`；该方法本仓无定义、官方 sglang 无此 API，仅内部 sglang fork 提供。上游 `Scheduler.__init__` 对 NULL 模式也会调用 `init_disaggregation`，REPLACE 钩子又使上游原版不再执行，导致官方 sglang 上任意模型、任意模式启动即在 scheduler 初始化处 AttributeError。

兜底方式与 `layers/quantization/registry.py` 的 `install_quantization_method` 同范式（探针、填坑、宿主优先，极性反转）：`hasattr(Scheduler, ...)` 为真（内部 fork 或未来官方实现）时不做任何修改，为假时安装 no-op 默认实现并记录一条 warning。调用点不改；提供该方法的 sglang 构建上行为逐字节不变。

实际验证（Qwen3.6-27B，TP2 gfx936，官方 sglang v0.5.22 线，NULL 模式，本机日志 dev77-fix3 与 dev77-fix4-disagg-guard 对照）：修复前双 rank 在 `init_disaggregation` 内 AttributeError、服务不可启动；修复后双 rank 各一条 warning、初始化通过，Uvicorn 监听后输出 ready，冒烟请求 health 200、两道确定性问答回答正确（temperature=0）。门禁：L0、仓库测试与导览快照检查通过（见本记录门禁口径）。

未验证模式：PD 模式在官方 sglang 上（decode/prefill 集成可能仍依赖更多内部 fork API，需同范式逐个探测或声明仅支持内部 fork）；内部 fork 环境回归（hasattr 短路，理论零扰动）；纯 MHA 等其他模型逐一回归（本问题与模型无关，崩点在 scheduler 初始化）。

已知限制与下一步：b4ae852 的 PD 集成面向内部 fork API 面，建议对官方 sglang 的兼容矩阵单独盘查；非 PD 服务路径经本修复在官方 v0.5.22 线端到端恢复。


## 2026-11-03 投机首启崩溃修复（draft_swa_window 垫值）

官方基线投机路径首跑（TP2、NEXTN、fa3、page64、Qwen3.6-27B）在 Scheduler.__init__ → build_kv_cache 的 AROUND 钩子崩溃：cache_retention.py 读取 draft.draft_swa_window，官方 v0.5.22 的 EAGLEWorkerV2 无该属性（官方树全仓 0 命中，系内部 fork 表面属性）。非投机路径 draft 为 None 不触发，故此前的非 spec 服务全绿。

修法沿用 PR #7 的 registry setdefault 范式：register_hooks 注册期对 BaseSpecWorker（EAGLEWorkerV2 / DFlashWorkerV2 公共基类）做属性级 setdefault，官方树缺属性时垫 0（builder 侧断言要求 SWA 模型才有非零语义，Qwen3.6 混合形态恒为 0），fork 或上游自行定义时一个字节不碰，自退役。运行时改动 2 行无注释。

验证：L0 门禁 556 Python 文件 + 8 Skill 零错误；117 项测试 104 过 13 跳（与基线一致）；build_guide 重建与 --check 通过。真机 TP2 NEXTN（日志 dev77-spec-fix1.log）13:55:37 ready，冒烟两题 巴黎/323 全对，accept len 2.48 / accept rate 0.49（Stage C 健康区间 2.4~2.9 / 0.48~0.62）。未验证：DFlash 等非 EAGLE draft worker、SWA 目标模型上的非零窗口语义（官方树无该属性来源，恒为 0）、PD 模式。
