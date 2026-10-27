# 开发经验与故障案例索引

> **本文用途**：把上次开发中实际遇到的问题整理为可查询的“症状→检查方法→修复经验”，供后续相似问题排查。保留原路径以兼容旧链接；本页不是新人开发总流程，也不是要求每次执行全部检查。
>
> 第一次开发看 [最小案例](first-plugin.md)，安装看 [安装手册](installation.md)，已有故障从 [排障入口](troubleshooting.md) 开始。固定版本的完整经过、提交和日志位置在 [V3 实践记录](v3-runtime-validation.md)。以下经验来自该记录，不代表本次又执行了模型验证。

## 1. 症状与案例映射

| 现象 | 本页案例 | 原记录 |
|---|---|---|
| 子进程/导入卡住，JIT 不结束 | 构建与环境 | V3 §2、§14 |
| apply 成功却 TypeError | 参数、公共分发与状态交接 | V3 §8、§9、§13 |
| KV 构造报错或下游 view/layout 不匹配 | KV 与 RoPE | V3 §7、§11、§12 |
| health 通过但生成乱码 | 验收器反例、TP 归约 | V3 §17、§18 |
| SSH 中断、资源被抢、停止失败 | 资源与服务观察 | V3 §15、§16 |
| 源码通过，换 wheel/embedded 后失效 | 产物与来源 | V3 §4、§5、§18 |
| HCU 主机上的 CPU 测试误入设备内核 | [设备能力与张量准入](#11-设备能力与张量准入) | 本轮实施 §7.1 |
| 方法抽取后构造失败、预导入引用未更新 | [方法与别名](#10-抽取方法与预导入别名) | 本轮实施 §3 |
| collective 未执行、输出未写入或通信器失效 | [通信器生命周期](#12-通信器生命周期与验收边界) | 本轮实施 §7.2、§7.6 |
| PD 字段已传递但复制前被旧检查拒绝 | [描述符身份](#13-描述符身份与位置式校验冲突) | 本轮实施 §7.3 |
| 类替换后父类关系错误、空查询时通信卡住 | [Attention 类图与空查询](#14-attention-类图与空查询通信顺序) | 本轮实施 §7.4 |
| JIT 仍编译旧头文件 | [头文件优先级](#15-jit-实际头文件优先级) | 本轮实施 §7.4 |
| 原生算子缺失或 FP8 载体不匹配 | [Provider 与 Carrier](#16-原生算子-provider-与-fp8-carrier) | 本轮实施 §7.4 |
| 只读 Git 检查后索引哈希变化 | [索引缓存](#17-只读-git-检查与索引缓存) | 本轮实施 §7.4 |
| CP 非 owner 写入 staging 或 None 引发异常 | [读写目标分离](#18-缓存读取-staging-与持久写入目标) | 本轮实施 §7.5 |
| ForwardBatch 构造、复制或 worker 状态缺字段 | [前向记录 Schema](#19-前向批次-schema-与复制生命周期) | 本轮实施 §7.6 |
| 候选评分因行跨度未对齐被原生算子拒绝 | [评分行对齐](#20-候选评分的行对齐与消费边界) | 本轮实施 §7.6 |
| Python list/tuple 分片报 range 索引错误 | [序列分片](#21-python-序列分片与-tensor-高级索引) | 本轮实施 §7.6 |
| 环境变量属性缺失而已有单测仍通过 | [配置对象与测试替身](#22-环境变量对象与测试替身的接口一致性) | 本轮实施 §7.6 |
| CP 与单卡结果偏差，但 Q/KV 数学输入一致 | [原生批次形状与数值参考](#23-原生计算批次形状与数值参考) | 本轮实施 §7.6 |

## 2. 参数、公共分发与状态交接

**故障现象：** loader 的注册/apply 全绿，但 Graph 调用带新的 keyword-only 参数时旧 Hook 抛 TypeError；另一次 `hcu_skip_layernorm` 在到达内部 Hook 前，先被公共 `prepare` 的严格 reader 拒绝。RMSNorm 的第三个可选 `None` 也属于真实调用契约。

**诊断方法：** 沿当前 caller→公共分发→reader→实际 Hook 查参数首次丢失/被拒绝的位置。核对 receiver、默认值、关键字参数和返回结构，不只看 `inspect.signature.bind` 或 `**kwargs` 能否接住。状态回归通过真实 `record/prepare/finish` 生成合法 residual 状态，避免手工拼出模型不会产生的 hidden/residual。

**修复措施与适用边界：** 在恰当的公共边界消费插件自己的参数，社区参数继续转发；本次使用单次调用的浅副本，不修改共享 plan。没有实现的非 None 语义明确拒绝，不为了兼容任意签名静默丢参数。测试方法见 [test-contract](../skills/hcu-plugin-dev/references/test-contract.md)。

## 3. KV 物理布局与 RoPE 消费者

**故障现象：** page-major KV 继承 token-row 描述，真实 pool 构造失败；修好后，下游 FA getter 又按另一种轴顺序解释 V。RoPE 的消费者需要 BF16，直接复用 canonical FP32 cache 则类型不匹配。

**诊断方法：** 不只看 tensor shape。穿过真实构造器、descriptor、byte span、getter 和 native consumer，检查 pointer/stride/轴含义及页边界。区分 canonical buffer 与特定消费者的 cast/view，检查缓存扩容、换源或 dtype/device 改变时是否失效。

**修复措施与适用边界：** 本次保留物理 HCU carrier，并为需要的消费者建立独立 cast；不覆盖其他路径仍需的 FP32 主缓存。用 `__new__` 填字段的旧测试绕过了构造失败点，不能证明生命周期正确。具体 dtype、布局和容差只适用于当时实现，换版本重新核对。[精度/Graph 参考](../skills/hcu-plugin-triage/references/precision-and-graph.md) 给出定位路线。

V3 后端集成另复现 BF16 SWA 视图错误：实际分配为每 token 512 个 BF16 元素，getter 却返回 packed FP8 的 584 字节宽度，公共 attention reshape 失败。修复使用实际池的元素维度；父池及四个 CP rank 的 BF16/FP8 consumer view 覆盖了修复前 5 个失败、修复后相关 37 项通过。名称包含 `bytes` 的 API 仍须按消费者核对单位，不能由名称推断最后一维。来源差异与设备复验见 [本轮后端集成](v3-20261026-development.md#76-完整后端依赖图与-cp-入口集成)。

## 4. MoE 归约所有权与重复通信

**故障现象：** TP4 服务 ready，真实生成乱码。实际模型形状/provider 的线性、MoE、attention、KV 探针未发现灾难性误差；最终发现旧 MoE 方法额外 all-reduce，而当前官方与 pinned HCU 的 FFN 出口已经负责归约。旧 skip 标志不再表示所有权转移。

**诊断方法：** 画清生产者输出是局部贡献还是全局结果、由谁以哪个 group 求和。对 TP1 与实际 TP>1、空/非空输入，检查数值和归约次数。把回归接到真实 `FfnExit/ResidualStream`，只 mock 通信，不能把出口也 mock 掉。本次最小反例是预期 8、实际 32。

**修复措施与适用边界：** 删除生产者多余的归约，出口继续拥有职责；回归由失败变通过，再以原接受配置验证模型。不要把“遇到乱码就删 all-reduce”当成规则；先证明自己的调用链和 owner。实际操作案例见 [故障诊断案例](troubleshooting.md#3-诊断案例tp4-生成异常与重复归约)。

## 5. 响应验收器与负例验证

**故障现象：** 旧检查搜索数字 `2`，乱码中包含 `2.0.2` 就误判算术通过。health 200、正常加载和算子对拍都未能发现这一验收漏洞。

**诊断方法：** 保留原始失败响应作为反例；同时验证正常样本可接受、坏样本必须失败。检查协议、模型、finish_reason、截断/控制字符与完整最终答案。若 checkpoint 的 content 带前导 think 段，只有本次协议允许且闭合时才提取最终文本。

**修复措施与适用边界：** [离线生成校验工具](../tools/validate_generation_response.py) 对确定性短答案做完整匹配；使用方法见 [响应验收](validation.md#7-确定性生成响应的离线验收)。它不适用于开放题/数据集评分，不证明 HTTP/监听者归属、cache hit、性能或清理；文件名带 cache-hit 也不能证明命中。

## 6. 构建进程环境与工具链一致性

**故障现象：** Python 启动兼容层提前导入 vendor，使 `ninja --version` 等子进程递归进入 JIT 锁；另一次 shell 的 Cargo 已正确，但 Torch builder 把解释器目录前置，仍选到旧 `/usr/bin/cargo`。

**诊断方法：** 分开导入、编译、模型启动三阶段，在实际 builder 子进程环境中记录 `sys.executable`、realpath、PATH、工具版本与 cache fingerprint。检查是否真正执行模型选中的 provider，而不是同名的其他实现。

**修复措施与适用边界：** Python 启动层延迟导入；任务级 wrapper/cache 修正实际子进程环境，不改 HOME 或全局工具。Rust 1.92 是当时 pinned 扩展需求，不是所有任务默认要求；缓存复用受 Python/Torch/ABI 限制，也不自动进入 wheel。详见 [runtime-and-env](../skills/hcu-plugin-dev/references/runtime-and-env.md)。

## 7. 资源冲突、观察中断与服务阶段

**故障现象：** 服务加载期间其他任务占用后四卡；用户授权停止冲突容器，但节点权限拒绝。后续该容器自行释放设备后才继续；没有绕过权限，也没有把拒绝写成成功。外层队列曾 pending，后续独立 attempt 实际通过，两份状态都保留。

**诊断方法：** 设备显存、利用率、可见设备/NUMA 映射、端口和进程身份一起看；一次空闲快照不等于独占。SSH 中断先核对原服务，不重复加载；退出 137 需回查各 rank 最早异常，不能单独认定 OOM。

**修复措施与适用边界：** 准入、初始化、owned ready、真实生成、清理分别记录。诊断时关闭 Graph 或换拓扑，不覆盖原接受配方的失败；用户允许改 TP 后保留原文及精确差异。默认只清理自有资源，用户授权也受节点权限约束。恢复和身份核实见 [分布式/资源参考](../skills/hcu-plugin-triage/references/distributed-and-performance.md)。

同一端口复用还遇到准入误判：上一轮服务已经清理、没有 LISTEN 进程，但不设置 `SO_REUSEADDR` 的临时 bind 仍可能被 TCP 连接状态拒绝。验证器按实际服务的端口复用语义进行 bind，并同时确认不存在活跃监听者；没有把 bind 失败直接等同于仍有服务或 GPU 被占用。原始 pending 轮次保留，后续新轮单独记录。

## 8. 交付产物与模块来源验证

**故障现象：** 源码、wheel、embedded 都需要独立验证；pytest 路径或同名外部 entry point 可能把本想验证的产物替换成源码。早期 package-data 也漏过 native 文件扩展名。

**诊断方法：** 先看真实模块 `__file__`、entry point 和目标集合，再核对产物哈希、资源清单及行为。验证 embedded 时屏蔽外部同名入口，验证 wheel 时移除源码路径；从外部测试目录运行可用 `--import-mode=importlib`，并复查来源。AST/源码提取测试还需检查实际读取文件路径：模块来自 embedded 并不保证提取节点也来自 embedded。

**修复措施与适用边界：** 按改动影响面补相关验证，不继承旧“全绿”，也不要求每次文档变更重跑模型。重新生成已有输出前核验旧 manifest 与全部 Git 变更，保护用户编辑，具体见 [回溯手册](backtracking.md)。

本轮服务验证还收紧了启动目录：候选预检通过后，原控制器仍沿用旧工程的工作目录，不能覆盖 `python -c`、`python -m` 或派生进程按当前目录找包的情况。该轮被主动中止并清理；新轮固定配方的 `cwd` 到待测源码或内嵌 `python` 目录，再以相同目录的新进程核验模块来源，并记录服务子进程目录。这是验证配置缺口，不将被中止轮次描述为已复现错误导入，也不继承其模型结果。具体状态见 [最终验证记录](v3-20261026-development.md)。

## 9. 案例维护规范

只加入有实际复现依据、能改变下一次排查决策的条目：症状、检查路径、根因、修复和未覆盖边界。版本、提交、PID、完整日志继续写运行/交付记录。本页提供经验索引；具体任务先由 [排障入口](troubleshooting.md) 或 [开发指南](plugins.md) 选择路线，不把全部历史检查变成永久必经流程。

## 10. 抽取方法与预导入别名

**故障现象：** V3 后续比较发现 `C.C._is_dcu`，并在构造时触发缺少 `__class__` 的 `super()`；旧常量 patcher 还遗漏预先导入的消费者。这些故障在注册/apply 和原有测试通过后仍存在。

**诊断方法：** 从公开构造或实际分支执行抽取方法；核对原拥有类、装饰器及全局命名空间。对引用重写做重复执行检查，分别观察标识符、属性、字符串和注释。常量更新前导入消费者，再检查 owner、旧别名、缺失属性和无关模块。

**修复措施与适用边界：** 使用 AST 标识符范围改写并明确类绑定；保留普通装饰器，对 descriptor/类拆分先声明映射。传播前保存旧引用，只更新声明作用域中已存在的消费者。本轮 8 项方法执行及 2 项别名反例、源码与回溯对照见 [实施记录](v3-20261026-development.md#3-故障复现与工具修正)。这些边界回归不证明整个 DeepSeek 模型或完整 HCU fork 已等价。


## 11. 设备能力与张量准入

**故障现象：** 迁移 CP LayerSplit 的索引 remap 时，HCU 节点的 CPU 测试进入 LightOP，报输入必须为设备 int32/int64。主机能运行 HCU 并不代表每次调用的张量都在 HCU 上。

**诊断方法：** 同时检查物理平台判断、实际输入 device 和 kernel 支持范围。保留 CPU 参考路径，另以真实设备输入证明优化分支实际命中，并覆盖负索引、页 padding 与 Graph 变输入重放。仅把平台探测 mock 为 False 会隐藏本例中的失败。

**修复措施与适用边界：** 本候选以 HCU 与设备 tensor 的组合条件选择 LightOP，CPU 路径不导入该依赖。局部数值和 Graph 通过只证明这一分派边界，完整 LayerSplit 的 pool 与 PD 接线仍未验证；见 [当前记录](v3-20261026-development.md#71-cp-索引重排的设备准入修复)。


## 12. 通信器生命周期与验收边界

**故障现象：** CP4 验证器在 PyNCCL 构造后检查 `available and not disabled` 并失败，但实际库与符号均正常。此前 native all-reduce 通过也不足以判断这条 PyNCCL 路径。

**诊断方法：** 阅读当前 communicator 的初始化、可用性字段和操作作用域。直接记录所选库、版本和缺失符号；核对 `disabled` 是初始化失败还是默认调用状态。用真实输入和预期归约值检查实际通信，不能只看属性或“未抛异常”。

**修复措施与适用边界：** 当前 PyNCCL 构造完成后默认 disabled，操作在 `change_state(enable=True)` 中执行。验证器按此生命周期检查 available 与作用域内状态，并在池广播前断言四 rank 归约结果。修复后真实 CP4 池与握手通过；这不证明模型网络传输已完成，详见 [CP 接线记录](v3-20261026-development.md#72-cp-父池容量与握手接线)。


## 13. 描述符身份与位置式校验冲突

**故障现象：** 新增 PD 描述符后，类型和全局层号可以正确配对，但旧 stride 预检仍按位置比较，合法重排在进入复制前返回 `-1`。另一个反例中，状态组件位置已映射，目标描述符仍沿用原组件索引，导致缺项错误。加载和 apply 均通过，无法发现这两类数据链路问题。

**诊断方法：** 将生产者、注册编解码、peer 发布、复制计划和最终字节串起来核对。源组件索引与解析后的目标索引分别跟踪，检查对应指针、长度、格式、位置索引和描述符是否一起使用映射结果。加入同一层号的不同 family、目标包含未拥有项、分批/线程路径及组件重排；预期结果从身份关系与原始缓冲计算，不能复用被测计划。

**修复措施与适用边界：** 描述符路径先按身份验证每项字节 ABI，原位置式路径保留原检查；重新注册时撤销旧 verdict，并在发布新快照前完成检查。worker 传递其已经取得的注册快照，避免元数据与指针来自不同时刻。当前函数级失败反例、真实 HTTP/ZMQ 与原生 Mooncake TCP 的整段缓冲对照见 [PD 实施记录](v3-20261026-development.md#73-pd-描述符生产协商与复制)。这不证明模型前向或流式 buffer 生命周期已完成。


## 14. Attention 类图与空查询通信顺序

**故障现象：** 仅保留旧 MQA 方法替换时，当前 HCU 新增的基类状态与 CP 生命周期未完整接入；空查询在预取/等待前返回。替换基类的模块引用也不会自动改变已有 DSpark 类的 `__bases__`。

**诊断方法：** 先比较当前类图、构造器、class closure 和派生类，再沿真实 `ForwardContext` 检查 prepare、SWA/extra prefetch、wait 与空输入出口的顺序。用真实池和公开 forward，mock 数学及通信载荷，保留决定是否参与通信的控制流。后端稀疏 prefill 的 flat indices 是独立消费者，不能因普通 attention 已 remap 就省略检查。

**修复措施与适用边界：** 本候选以完整 MQA 基类、MQALayer 和 DSpark 子类接入，保留 CP 空查询的通信顺序，局部读取索引不改写共享 metadata。旧 forward 的失败反例与当前构造/读取回归见 [attention 实施记录](v3-20261026-development.md#74-mqa-类图缓存读取与原生边界)。这不证明完整 CP 模型或 compressor/indexer 已接入。新增能力按 [开发指南](plugins.md#使用-skill-开发插件能力) 调用开发 Skill，按本案例选择相关边界，不要求执行全部历史案例。

## 15. JIT 实际头文件优先级

**故障现象：** overlay 文件哈希正确，`extra_include_paths` 也以前置 overlay 开头，真实 MQA JIT 仍报缺少 `__syncwarp`、`__shfl_xor_sync`。当前 loader 在 extra 之前拼接官方 `DEFAULT_INCLUDE`，编译器优先解析了官方同名头文件。

**诊断方法：** 从实际 provider 的构建记录检查最终 include 顺序、共享默认路径与缓存 fingerprint，而非只检查 Hook 入参。确认是哪份同名头文件提供 HIP 宏；用触发原故障的真实调用复验，资源哈希不能代替编译与执行。

**修复措施与适用边界：** 在本次固定 loader 的 BuildSpec 冻结前，将插件 header 根置于共享默认 include 列表首位，保持列表引用与幂等性，使有效顺序进入缓存键。只修改运行进程配置，不修改输入源码或全局工具链。此方法针对当前 loader 的拼接契约；其他 builder 先核对其行为，不把修改共享列表作为通用方案。见 [attention 实施记录](v3-20261026-development.md#74-mqa-类图缓存读取与原生边界)。

## 16. 原生算子 Provider 与 FP8 Carrier

**故障现象：** 小型真实 MQA 先遇到当前环境缺少 vLLM `_C.rms_norm`，修复后又遇到 FlashMLA 不接受 FNUZ 类型的 packed cache carrier。单独导入 native 包和 Hook apply 都成功。

**诊断方法：** 记录实际选择的 op、输入 device/dtype、物理字节及 stride。区分 FP8 数值转换与 raw packed bytes 的载体解释；检查直接原生调用及其 dtype matcher。参考结果按边界拆分：Q 用独立 PyTorch，attention 运输对照直接 native，输出投影另行计算。不能把与 native 一致表述为独立 FP32 算法精度证明。

**修复措施与适用边界：** 本次独立无 residual 的 HCU RMS 使用实际可用 LightOp；已有 residual 与 batch-invariant 路径继续按各自契约执行。HCU FP8 FN/FNUZ packed cache 在 FlashMLA 入参边界做零拷贝 uint8 view，验证 pointer、shape、stride 和字节不变，BF16 不转换。FP8 与可选 BF16 gather 的真实小型 MQA 均通过，仍不等于完整模型服务验收。见 [attention 实施记录](v3-20261026-development.md#74-mqa-类图缓存读取与原生边界)。


## 17. 只读 Git 检查与索引缓存

**故障现象：** 回溯后 official 的 HEAD、工作区与暂存内容均未变化，但索引文件 SHA256 变化。`git status` 在 stat cache 过期时会刷新索引，普通只读检查也可能产生这一写入；本次时间与回溯的来源校验一致。

**诊断方法：** 区分源码内容、暂存语义和索引文件字节。固定前后 SHA、mtime，检查 staged diff 与 HEAD/tree；不要为了哈希一致重置或替换用户索引。原始索引字节未留存时，不能声称逐字段确认仅缓存变化。本次最小反例在两个临时来源中只改变文件 mtime，不改变内容，旧生成器使两份索引字节变化。

**修复措施与适用边界：** 生成器子进程设置 `GIT_OPTIONAL_LOCKS=0`，避免来源检查的可选缓存刷新；生成 checkout 所需的正常写锁不受此选项禁止。回归检查真实回溯前后两份索引字节完全一致，并继续验证 dirty 拒绝与生成行为。实际变化及后续保留性验证见 [attention 实施记录](v3-20261026-development.md#74-mqa-类图缓存读取与原生边界)。调用回溯 Skill 时按 [来源保护说明](backtracking.md#9-来源索引保护) 执行；该选项不是禁止 Git 所有写操作的沙箱。


## 18. 缓存读取 Staging 与持久写入目标

**故障现象：** CP 池已经提供 owner-aware 写 getter，但旧 fused compressor 仍使用读 getter。非 owner 的紧凑读 staging 可返回 tensor，若拿全局 slot 写它，会混淆读视图与持久存储。仅把调用点改为返回 None 还不够：旧 C1/C2 Python wrapper 会对 None 调用 `.view`，在 native no-store 分支前失败。

**诊断方法：** 从 backend 写入调用点穿过 pool getter、Python carrier 和 native pointer 分支，检查各 rank 仍须执行的 latent、pair state 与复制索引 K。分别验证有主 KV 和无主 KV；非 owner 的无写入不意味着整段计算可提前返回。按 page-major ABI 检查 payload、scale 与保留区，不能把每 token 的逻辑 584 字节误作物理交错布局。

**修复措施与适用边界：** 迁移当前 HCU 的写 getter 调用及 C1/C2 包装：None 转为零页 uint8 载体，由现有 native kernel 禁止主 KV 写入，其他产物继续生成。真实池边界保留读 staging 并检查未修改，独立参考验证 latent/ring，变输入 Graph 检查动态状态；旧实现的三个非 owner 对照均失败。测试不假设读 getter 在所有生命周期阶段都会抛错，也不把函数包装器保留的原 `__module__` 当实际 provider。完整 CP indexer 与模型尚待集成，见 [低压缩比缓存实施记录](v3-20261026-development.md#75-低压缩比缓存的持久写入边界)。


## 19. 前向批次 Schema 与复制生命周期

**故障现象：** 当前 HCU 后端依赖新的 `encoder_swa_row_floor` 等运行字段，固定官方 ForwardBatch 尚无对应声明；沿真实后端初始化旧批次会在字段读取处失败。只添加类默认属性或临时实例属性，还可能在 `dataclasses.replace` 等复制路径丢失状态。

**诊断方法：** 同时比较后端 caller 和前向记录的字段声明、默认值、构造、复制及序列化。区分冻结 ServerArgs 输入与可变 ForwardBatch 运行状态；不能用给前者补属性的方式迁移后者。以完整 pool、allocator 与公共 metadata 初始化复验，避免手填 core metadata 绕过映射绑定。

**修复措施与适用边界：** 本候选继承固定官方 ForwardBatch，以 dataclass 声明来源新增的十个唯一字段，并通过 class REPLACE 供现有工厂使用；保留官方既有方法，未将整个前向执行器声明为已迁移。复制、pickle 和实际 GPU metadata/层链路分别验证，原记录反例与当前范围见 [后端集成记录](v3-20261026-development.md#76-完整后端依赖图与-cp-入口集成)。字段声明通过不证明后续流式事件、ACK 或跨 stream 状态生命周期已完成。

## 20. 候选评分的行对齐与消费边界

**故障现象：** 原生 dense logits 已按行对齐，Torch 候选源在请求长度 259 时却生成 stride 为 259 的 FP32 矩阵，进入 LightOp 后报 `logits must be FP32 [B,W] with 16-byte aligned rows`。仅验证逻辑 shape 或已对齐的底层算子测试未发现该调用端缺口。

**诊断方法：** 记录实际 producer 的 dtype、shape、stride 和起始指针；沿公开 caller 检查进入 native consumer 前的 carrier。测试不同请求长度、非连续输入、原数据不被改写和变输入 Graph。候选模式的参考评分只比较被选择的块，排除块的 `-inf` 是协议行为，不能当作数值失败；block 选择算法另有独立参考。

**修复措施与适用边界：** 在 Torch candidate-source 调用点准备保持逻辑维度和数值的对齐视图，已对齐输入继续复用。候选算子的非法 dtype/offset/stride 拒绝保持原合同，不捕获 native 错误后静默切换其他算法。连续奇数宽度、偏移、列 stride 的实际 kernel/Graph 对照，以及真实候选源→消费者链路见 [后端集成记录](v3-20261026-development.md#76-完整后端依赖图与-cp-入口集成)。这项适配针对当前 16-byte 行 ABI，不作为所有 kernel 的默认 padding 规则。

## 21. Python 序列分片与 Tensor 高级索引

**故障现象：** 来源分片 helper 将 list/tuple 与 Tensor 放在同一分支，用 `input_[range(...)]` 选择行，Python 序列抛 TypeError。只以 Tensor 验证会遗漏这一声明支持的输入类别。

**诊断方法：** 按接口接受的真实容器检查索引语义和返回类型，分别覆盖各 rank、不能整除的长度及空结果；不要从 Tensor 的高级索引能力推断 Python 序列也支持同一写法。

**修复措施与适用边界：** legacy 与 interleave 的序列路径使用步长 slice，保留 list/tuple 类型；Tensor 路径保持来源行为。当前 interleave 四个 rank 的同一用例由 4 failed 变为 4 passed，见 [后端集成记录](v3-20261026-development.md#76-完整后端依赖图与-cp-入口集成)。

## 22. 环境变量对象与测试替身的接口一致性

**症状**：BF16 KV 的真实前向进入 norm/RoPE Hook 后，读取 `SGLANG_OPT_USE_FUSED_COMPRESS` 报模块属性缺失；已有单测仍通过。

**定位**：当前官方配置项定义在 `sglang.srt.environ.envs` 对象。旧 Hook 导入了 `environ` 模块；单测又向该模块添加虚构属性，因而掩盖实际接口错误。

**修复与验证**：Hook 改用官方 `envs` 对象；测试通过插件注册实际配置描述符，再设置环境变量切换融合与回退路径，不给错误对象补属性。真实 BF16 KV 前向在后续缓存维度修复后完成四卡复验，CPU 路由通过不能替代设备执行。原始失败见 [V3 后端集成记录](v3-20261026-development.md#76-完整后端依赖图与-cp-入口集成) 中的 BF16 CP4 探针。

**适用场景**：配置 API 升级、环境变量注册和分支测试。按 [开发 Skill](../skills/hcu-plugin-dev/SKILL.md) 执行时，保留实际配置对象与 descriptor，只替换昂贵计算；历史模块级 API 需先核对当前定义。

## 23. 原生计算批次形状与数值参考

**症状**：CP4 四层链路与单卡完整批次的输出对照失败，偏差逐层放大；首层进入注意力的 Q 完全相同。

**检查方法与证据**：捕获原生调用实际使用的 Q、KV、索引和有效长度。在首层 15 个查询中，逐查询解码后的 KV 集合和独立 FP32 注意力结果均相同。将这些输入放在同一张卡上重放，完整批次与 CP 批次形状仍相差 `0.0078125`；按相同物理批次形状重放后，与 CP 输出最大差为 0。该证据把首个差异限定在原生执行的批次形状依赖，不能据此宣称整个模型没有其他误差。

**处理方式与适用边界**：保留完整批次的失败结果和原误差阈值；新增显式的批次形状对照，分别验证 CP 数据流和原生数值差异，不把参考路径的改变隐藏为“精度问题已修复”。后续版本或 provider 变化需重做对照。原始输入与单卡重放记录见 [V3 后端集成](v3-20261026-development.md#76-完整后端依赖图与-cp-入口集成)。

**Skill 使用**：使用 [开发 Skill](../skills/hcu-plugin-dev/SKILL.md) 建立验收时说明参考精度、实际 provider、分块和批次形状；已有数值故障由 [诊断 Skill](../skills/hcu-plugin-triage/SKILL.md) 沿首次分叉定位。只替换需要隔离的计算因素，并将完整服务与局部数值结论分别记录。
