# V3 2026-10-26 实施与验证记录

> **用途**：记录 `plugin-v3-1026.txt` 指定输入下的插件开发、内嵌源码生成、HCU 逻辑对照及本轮验证。执行跨 2026-10-26 至 2026-10-27（Asia/Shanghai）。方法库保留 Skill、工具和记录，运行时代码位于独立 `output-code` 仓库。

> **当前交付**：以下 §1–7 保留各阶段快照；正式输出、最终验证和限制统一见 [§8](#8-最终交付与需求验收)。

## 1. 固定输入与产物

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

## 2. 已实现模块与证据边界

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

## 3. 故障复现与工具修正

### 3.1 常量重写与类闭包

旧正则先把 `_is_hcu` 改成 `C._is_dcu`，又在下一轮替换 `_is_dcu`，产生 `C.C._is_dcu`。类方法抽成普通函数后，原来的 `super()` 也失去隐式 `__class__` 闭包。仅检查 AST、注册和 apply 均无法发现这些执行错误。

当前 [方法同步工具](../tools/sync_deepseek_methods.py) 根据 AST Name 的 UTF-8 字节位置修改引用：不改变属性名、字符串和注释；重复执行不会再次限定；遇到本地同名变量拒绝猜测。构造方法使用明确拥有类和 receiver；保留普通装饰器，遇到 classmethod/staticmethod/property 描述符时要求先声明绑定契约。该改进不代表历史抽取 profile 已适配当前 HCU 的类拆分，`--check` 仍需完整预检。

回归包括真实函数执行、父类调用、重复变换、中文文本偏移、局部变量冲突、装饰器行为以及来源符号漂移时不写输出。方法库完整门禁结果单独记录，不计入插件测试数。

### 3.2 预导入别名

旧 patcher 先修改 owner 常量，再调用传播函数；传播函数读到的已是新对象，因而跳过旧消费者。修正后先捕获旧值并限制作用域，只更新明确存在、仍引用旧对象的 DeepSeek 消费者。测试同时检查无关模块和缺失属性，避免修复变成全局副作用。

### 3.3 执行身份

容器 root 执行只读 Git 审计时触发仓库所有权检查。审计、源码构建及回溯改由源码 UID 执行；运行服务沿用容器已有合法设备用户。未修改全局 Git safe.directory、HOME、系统 Rust 或设备权限。

## 4. 源码、内嵌与 HCU 逻辑对照

本轮执行了三个不同层次的比较，原始结果为 `fork-logic-comparison.json`：

1. **产物保真**：442 个内嵌插件文件逐字节等于本轮插件；448 项生成清单核验通过。启动集成使用生成器定义的 bootstrap 映射，不能把不生成全局 `usercustomize.py` 误报为丢文件，也不能忽略入口顺序。
2. **来源保真**：284 项 JIT 资源逐项比较当前 HCU、插件、embedded 三方；Channel FP8 转换 helper 的 AST 与当前 HCU 相同。
3. **行为与版本差异**：检查整个 `python/sglang` 当前树，社区与 HCU 有 1206 项内容不同、387 项 HCU 独有、20 项社区独有。抽查的 24 个方法中，21 项需要适配/陈旧行为复核，3 项源拥有类或方法已变更。不能把文件名、签名、AST 相似或 210 个目标全部 apply 当成整个 fork 逻辑等价。

**对照结论：本轮交付形态和明确验证的契约已有证据；整个 HCU fork 的全逻辑等价尚未证明。** 当前 HCU 的 CP LayerSplit 全链路、候选/分页 FP4 索引、PD/DSpark 辅助状态、V4.1 中间 chunk decoder、EnGram、W4A8/DeepEP，以及 AOT `rope_fp4` 构建导出没有在本轮形成完整运行验收。这些能力不得标记为已迁移或已验证。

对旧源码差异的处理遵循当前 caller/consumer 契约。例如 MiniMax 的框架 `FfnExit` 已负责归约，保留 fork 旧生产者归约会重复求和；此处应保留有回归与服务证据的适配差异，不能为了文本相等恢复旧操作。

## 5. 验证矩阵

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

## 6. 复现与 Skill 入口

- 三方输入/差异审计：按 [迁移指南](migration.md) 调用 [迁移 Skill](../skills/sglang-dcu-plugin-migration/SKILL.md)，固定本轮 SHA 后生成新的审计和台账。
- 修改能力或抽取方法：按 [插件开发指南](plugins.md) 调用 [开发 Skill](../skills/hcu-plugin-dev/SKILL.md)，先保留真实构造、装饰器、状态与出口边界，再执行对应回归。
- 生成新 embedded：按 [回溯指南](backtracking.md) 调用 [回溯 Skill](../skills/hcu-plugin-backtrack/SKILL.md)。两个生成输入为 official/plugin；提供的 HCU 输入用于独立逻辑对照，不替换生成来源。
- 安装交付：按 [安装指南](installation.md) 使用已有源代码或 wheel，先验证实际解释器和模块来源，再运行目标服务。

证据脚本在本轮 ignored artifacts 中固定路径与 profile，只作为可复现实验记录；未来版本按对应操作指南重新固定输入和执行范围，不复用本页通过状态。

## 7. 逻辑对齐后续工作

以下保留 v0.3.1 起的检查点与后续开发过程；各阶段的 pending 状态按当时范围解释，当前需求验收见 §8。完整 HCU fork 的所有模型与分支等价不属于这些局部通过结论。

后续开发位于独立的 `output-code/.sglang-plugin-v3-parity`（分支 `v3-hcu-parity`），以本页插件提交为起点。初始提交 `f2394ab` 迁移了 CP LayerSplit 的十个模块及分页/状态/描述符回归；后续 `0.3.2.dev1` 已接入父池、容量预算、工厂和握手规范化，进展见 §7.2。完整模型前向与 PD 网络字节传输尚未闭合，不能标记为可启用的完整 LayerSplit 服务。


### 7.1 CP 索引重排的设备准入修复

候选第二次局部回归为 23 项通过、1 项失败：HCU 主机上的 CPU 张量进入 LightOP 的 device-only remap，报 `indices must be device int32 or int64`。设备能力与输入 tensor 的实际设备不是同一个条件；当前候选同时检查 HCU 与 `indices.is_cuda`，CPU 张量使用原 PyTorch 路径。

修复后，候选局部测试为 **32 passed / 136 subtests passed**，证据 `cp-layer-split-candidate-attempt3.log`。新增对照包含真实 LightOP 的 int32/int64、乱序页、静态负值 padding、缺页、负 token ID，以及三轮变输入 Graph 重放；CPU 路径另检查不导入设备专用依赖。状态机测试中的部分边界沿用 HCU 的 AST 节点提取方式，不能代替完整模块导入、pool 构造或模型运行。方法来源及适配哈希在候选 `source-provenance.json`，能力接线状态在 `PARITY-CANDIDATE.md`。


### 7.2 CP 父池、容量与握手接线

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


### 7.3 PD 描述符生产、协商与复制

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


### 7.4 MQA 类图、缓存读取与原生边界

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


### 7.5 低压缩比缓存的持久写入边界

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


### 7.6 完整后端依赖图与 CP 入口集成

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

## 8. 最终交付与需求验收

2026-10-27 在历史验证容器完成本轮验证，并将已验证提交同步到正式输出。运行时代码继续位于方法库之外；方法库的 HEAD 和暂存区保留原状，文档及 Skill 更新保存在工作区。

| 产物 | 正式位置（工作根目录下） | 来源与状态 |
|---|---|---|
| 源码插件 | `output-code/sglang-plugin` | v0.3.2.dev5，干净提交 `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c` |
| 内嵌源码 | `output-code/sglang-das` | official `baa9befa381511cd4d56f83980951e5a45fffc6d` + 上述插件；538 项清单、532 个插件文件逐项核验 |
| wheel | `output-code/dist/sglang_dcu_plugin-0.3.2.dev5-py3-none-any.whl` | SHA256 `c2260d30015e932faf33c09082758dee675113785353717a2750f75775544fac`；独立 CPU 安装/加载已验证 |
| v0.3.1 回退 | 插件分支 `validated/v0.3.1-before-v3-1026`；`output-code/.sglang-das-v0.3.1-validated-backup` | 旧插件 `39038d5` 和原 448 项内嵌文件保留 |

### 8.1 最终验证矩阵

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

### 8.2 HCU 逻辑对照结论

`delivery-fork-logic-comparison.json` 比较正式内嵌核心与固定 HCU 当前树，列出 1208 个内容不同文件、387 个 HCU 独有文件及 20 个内嵌核心独有文件；这些计数是审阅清单，不直接代表行为相同或不同。生成器以官方源码为核心，HCU 行为经插件注册接入。

532 个插件文件与生成树相同，284 项当前 JIT 来源资源核验通过；完整后端的 42 个模块、482 个顶层节点、9 个桥接节点及十个前向字段有声明映射对照，BF16 SWA 维度修正单独记录。GLM 缺失模块导入、Python 序列切片、候选评分行对齐、FP8 原生载体和 BF16 元素维度均是明确的适配/修正。

**结论：明确迁移的节点及已执行契约有来源和行为证据；没有宣称整个 HCU fork 全逻辑等价。** 完整 DeepSeek 模型引擎、全部 decoder/DSpark streaming source-event/ACK、W4A8/DeepEP shared-expert、AOT FP4 构建导出及所有原生分支仍未形成全范围验收。CP4 的原生完整批次数值差异与显式匹配批次对照分别保留，详见 §7.6，不将对照选项解释为生产精度修复。

### 8.3 需求与证据对应

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
