# V3 实践：固定输入、插件输出与源码回溯

相似故障的经验索引见 [开发经验与排障索引](development-methodology.md)；本页保留固定版本的失败、修复和原始证据，不把历史设备、拓扑与数字作为默认配置。

> **本文用途**：记录 2026-10-22 指定源码与容器上的实际开发流程、已发现的兼容问题、可重复验证命令和资源限制。方法库仍不内置业务包；本次运行时代码交付在独立 `output-code` 仓库。结论按接线、CPU、GPU 和服务分别记录。

## 1. 输入与交付目录

工作根目录以 `${BASE}` 表示；下表保留历史相对布局，不要求当前部署沿用。`${BASE}` 仅是历史位置标识；执行命令使用 [路径配置](configuration.md#1-当前工具的路径配置) 中的 `OFFICIAL`、`PLUGIN`、`OUTPUT` 与 `ARTIFACTS`。

| 对象 | 相对工作根路径 / 固定基线 | 用途 |
|---|---|---|
| 社区输入 | `input-code/sglang`；`baa9befa381511cd4d56f83980951e5a45fffc6d` | 固定官方框架与插件接口 |
| HCU 输入 | `input-code/sglang-das`；`5f67ce01c3ec1894916d16627eaca4959fbe2d90` | HCU 方法、模型和 native source 来源 |
| 插件起点 | 旧 `sglang-plugin-das`；`ce870e225df6b6d3db9e019cdbc620517f53648a` | 继承已有能力并修复版本漂移 |
| 运行时输出 | `output-code/sglang-plugin`；包名保留 `sglang_dcu_plugin`，版本 0.3.0 | 独立插件与测试；entry point 为 `dcu` |
| 回溯输出 | `output-code/sglang-das` | 由干净社区树与运行时输出生成 |
| 方法库 | `hcu-sglang-plugin` | 本文、skill 和审计/回溯/验证工具 |
| 原始配方 | 工作根上级的 `minimax-debug/d.sh` | MiniMax M2.5 FP8、TP8、FA3、FP8 KV、30001 |
| 执行环境 | 历史验证容器；Python 3.10.12，torch 2.11 DTK | 固定目标容器 |

输入仓保持不改。审计需要导入官方对象时，在 ignored artifacts 下创建独立 fork clone，避免修改输入 Git refs。输出插件需要提交成干净输入后才能回溯；这次提交是构建可重放源码的前提，不表示推送外部仓库。

## 2. 本次发现与修正

| 证据/问题 | 修正 | 防回归方法 |
|---|---|---|
| Python 子进程 `ninja --version` 等待父 AITER JIT 锁 | usercustomize 只安装延迟 import finder，不提前导入 AITER/torch | 新 Python 与嵌套子进程必须及时退出；真实 vendor 模块优先 |
| 源码所属 UID 在容器 passwd 中不存在，默认缓存失败 | 显式使用可写任务缓存；构建按源码所有者执行 | 不改 HOME；记录解释器、UID 和缓存路径 |
| 容器 root 的 editable build 遇 Git dubious ownership | 由源码 UID 安装到隔离 target，再将 metadata 目录加入 PYTHONPATH | 不配置全局 safe.directory 通配符 |
| 旧 loader 测试没有 entry point 也能通过 | 严格校验非空集合、预期 targets、全量 apply、来源路径与框架软错误 | [严格运行时验证](../tools/validate_plugin_runtime.py)；空加载/部分失败单测 |
| 旧 ServerArgs 私有方法已删除，框架只记日志 | 改为正式具名 resolution hooks、model overrides、post-process seam | 单列具名 seam，不能只统计 HookRegistry |
| ServerArgs 已从 dataclass 演进为 msgspec 记录 | 用官方 `is_record`；兼容选项由插件管理，不补写冻结输入 | 真实 Struct 测试及新进程选项继承测试 |
| 原配方 `--cuda-graph-max-bs` 产生歧义 | 精确兼容别名指向 decode 参数，保留独立 prefill 参数 | 解析别名与未设置默认值测试 |
| MiniMax decoder residual 参数与边界接口变化 | 使用当前 residual stream 方法，停用时保留当前签名 | residual 更新与共享 plan 不变测试 |
| 来源 INT8 fused projection 与 FP8 quant_method 不兼容 | RMS 交接保留配置的 FP8 quantizer，不把 INT8 buffer 塞入 FP8 方法 | 数值/状态对拍；不宣称此兼容分支有 INT8 融合性能 |
| 旧 native overlay 的头文件/实现已漂移 | 同步固定 HCU 来源的 283 个 native 文件，记录逐文件 SHA256 | 输出 `sglang_dcu_plugin/jit_overlay/source-provenance.json`、wheel 与回溯清单检查 |
| 原回溯 package-data 只包含 `.h/.cuh` | 生成器 v2 使用 `jit_overlay/**/*`，覆盖 `.cu/.cpp/.hpp` | native 资源回溯/哈希测试 |
| MiniMax 权重加载后 KV pool 构造失败：page 缓存继承 NHD token-row 描述 | 覆盖 `_kv_tokens_per_row` 返回实际 page_size，沿用官方描述符与 byte-span 计算 | 真实 CPU 构造及 K/V 不同 head_dim 的 descriptor/transfer span 回归 |
| DeepSeek 类拆分导致同步脚本 StopIteration | 源符号和生成语法全部预检，再逐文件替换；明确报告缺失类/方法 | 后续 job 漂移时前面的输出也不能被改写 |

硬件探测使用真实 torch HIP/架构信息；adapter 暴露给框架的平台语义不等同于真实设备。未移植或未验证的模型/优化仍按支持范围和台账保留，不能从固定目标接线推出全 fork 等价。

## 3. 差异清单的读法

原始 merge-base 清单有 9070 个条目，包含大量双方共同继承的历史；生成台账时可排除字节相同源码和测试/文档项。初始台账为 7268 excluded、1802 pending，**这是审计起点，不是 1802 项已实现或全量等价**。

结合当前树差异、具体符号、调用链与验收能力逐项处理；既不能按硬件关键词丢弃通用改动，也不能把 merge-base 以来所有上游功能都误认为新增 HCU plugin 工作。运行时输出的 baseline/native 清单和本次验收记录说明实际交付范围。

## 4. 插件接线和 CPU 验证

固定官方源码、输出插件以及隔离 metadata 目录的优先级，先确认实际导入路径。容器中普通 editable 安装可用时直接安装；本次使用源码所有者 UID 安装到独立任务目录（以下为 `$SITE`），避免 root Git ownership 问题。

先加载 [路径配置](configuration.md#1-当前工具的路径配置)，以下命令沿用该次历史版本的合同文件；当前版本按对应交付记录选择合同。

```bash
: "${OFFICIAL:?请设置官方源码路径}"
: "${PLUGIN:?请设置外部插件路径}"
: "${ARTIFACTS:?请设置本次证据目录}"
SITE=$(mktemp -d)
python3 -m pip install -e "$PLUGIN" \
  --target "$SITE" --no-deps --no-build-isolation
export PYTHONPATH="$OFFICIAL/python:$PLUGIN:$SITE"
export SGLANG_PLUGINS=dcu
python3 tools/validate_plugin_runtime.py \
  --official-python "$OFFICIAL/python" \
  --plugin-root "$PLUGIN" \
  --expected-targets "$PLUGIN/tests/contracts/targets_v0_5_22.json" \
  --output "$ARTIFACTS/runtime.json"
python3 -m pytest --import-mode=importlib -q "$PLUGIN/tests"
```

运行时当前目标合同为 208 个 Hook 目标及三个具名参数解析 seam。以最终 JSON 和预期集合为准；目标数量不等于能力数量。生成源码、apply、CPU、GPU 和模型输出各自有证据，不能互相替代。

## 5. 回溯与外部 entry point 隔离

```bash
python3 tools/reconstruct_sglang_das.py \
  --upstream "$OFFICIAL" \
  --plugin "$PLUGIN" \
  --output "${OUTPUT:?请设置与输入不重叠的新输出目录}" \
  --revision baa9befa381511cd4d56f83980951e5a45fffc6d
python3 tools/verify_reconstruction.py --output "${OUTPUT:?请设置与输入不重叠的新输出目录}"
```

核验清单后，移除 PYTHONPATH 中输入树、外部插件根及其 metadata 目录，仅保留生成树的 `python`；从方法库执行验证：

```bash
PYTHONPATH="$OUTPUT/python" \
python3 tools/validate_plugin_runtime.py \
  --official-python "$OUTPUT/python" \
  --plugin-root "$OUTPUT/python" \
  --disable-entry-points \
  --expected-targets "$PLUGIN/tests/contracts/targets_v0_5_22.json" \
  --output "$ARTIFACTS/embedded-runtime.json"
```

`--disable-entry-points` 屏蔽外部发现后仍须非空且全部应用，才证明内嵌入口有效。跑外部插件测试文件检查生成包时使用 `--import-mode=importlib`，防止 pytest 把外部插件根插到 PYTHONPATH 前面；源文件身份仍须由严格验证确认。

## 6. GPU 与原配方服务验收

| 阶段 | 检查 | 不能宣称 |
|---|---|---|
| 配方解析 | 原 d.sh 环境和 CLI，经 prepare_server_args **及 resolve_once** | 不能宣称服务 ready 或推理成功 |
| GPU 准入 | 按测试实际拓扑检查所用卡的显存/计算利用率；服务另查 30001 端口，TP8 需完整 8 卡 | 利用率为 0 但模型占用约 118000 MiB 不算可用 |
| 两份源码 GPU 用例 | FP32 gate、LightOp FP8 quant、变输入/状态 graph replay | 不等于完整模型精度/吞吐 |
| 原 d.sh 服务 | 原参数启动、health、实际生成和目标分支日志 | 不能只凭 shell 进程存在判成功 |
| 清理与交接 | 仅清理本任务创建的进程，保留日志/响应/缺失项 | 不停止其他服务来造空闲 |

GPU 用例默认不执行；在资源就绪后设置 `HCU_PLUGIN_RUN_GPU_TESTS=1`。单卡算子用例检查并固定实际空闲卡，例如同时记录 ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES 为 0；不必等待其他未使用卡空闲。插件与回溯串行运行，记录各自来源路径；最后以原 d.sh 的 TP8 配置完成服务验证。变更已有等待流程前须核对实际进程并确认尚未启动服务，避免测试和 TP8 抢占同一张卡；后续资源等待沿用原截止时间，不因调整顺序延长。若显存/端口被占用，状态保留 pending，不能改成 TP4、小模型或删改原参数后声称原配方通过。

本次任务证据保存在方法库 `artifacts/v3-20261022/`。最新状态和汇总见 [交付记录](delivery-validation.md)，源文件保护、回溯覆盖和模型正确性的边界见 [回溯手册](backtracking.md)。

## 7. 原始服务失败的实证与修复（恢复连接后补充）

首次服务原始记录 `artifacts/v3-20261022/plugin-service-result.json` 为 failed：配方在 ready 前退出 137。完整 rank 日志说明权重已加载，随后 `_build_kv_buffer_descs` 把 HCU page-major `(97633, 1, 64, 128)` 误当作 `6248512` 个 token-row 而抛 ValueError，父进程收到子进程失败信号后清理。仅凭 137 或 SSH 断线不能归因为 OOM；容器后续的退出时间与服务失败时间也要分开。

运行时提交 `6984176` 修正 `_kv_tokens_per_row`，保留社区 KV 描述符与物理页布局，不修改原 d.sh 参数。旧单测用 `__new__` 手工填缓存，绕过真实 `_create_buffers→descriptor` 构造链，因此没有发现该错误。新增真实构造和双缓冲 byte-span 测试；修复后容器 CPU 套件 231 passed、3 skipped（GPU 门控），最终严格 loader 206 registered/applied 与 3 个具名 seam 通过。回溯与服务状态分别记录，不从这些结果推断完整模型已成功。

后续服务重试使用独立 run-id 目录，保留首次失败日志；控制器记录历次看到的子进程句柄、仅清理自有进程，HTTP 请求绕过本地 proxy。恢复连接或控制器观察中断后先核验原服务，不能直接重复加载。当前 GPU 资源准入与最后模型状态见交付记录。

## 8. Graph 调用契约与加载生效的区别

`plugin-kvfix-attempt3` 已通过 KV 初始化，随后在 Graph batch size 配置失败：官方 `get_cuda_graph_batch_size_alignment(*, gathered_buffer_required=None)` 不再接收 ServerArgs。旧 AROUND Hook 虽可注册/apply，调用时仍因新关键字 TypeError 失败。`8681fb9` 使用目标关键字签名；HCU MiniMax 需要额外 TP 对齐时把 gathered 条件置 True，其余 overlap/CP 计算直接委托官方原函数，不复制整段逻辑或关闭 Graph。

调用形态审查还发现 `RMSNorm.forward_hip` 的 `quant_linear` 参数未转发，`57f4f84` 在原路径/原生回退都完整传递。新增测试覆盖默认/显式 bool、HCU/non-HCU、不同调用方式，外部完整 CPU 246 passed/3 GPU skipped。后续测试与原服务状态见交付记录；之前 6984176 的 231 CPU 和 3 GPU 记录按其原提交解释。

因此严格 loader 的非空/全量 apply 是接线证据，不是 ABI/调用和数值全兼容证明。对 AROUND 还需检查目标真正调用的 keyword-only、可选参数及绑定 receiver，再以最小调用和真实生命周期复验。`inspect.signature(...).bind` 可生成静态候选；类方法 self/cls 通常由 descriptor 以位置传入，不能把 receiver 名称差异自动判为破坏，也不能把 **kwargs 可绑定当作参数已正确转发。此次审查其余 DSA/DSV4 等候选另留待核验，不记为全 fork 已兼容。


## 9. 公共 StageBoundary 分发与 fused norm 交接

原配方第 4 次运行在 `57f4f84` 上通过了权重、KV 初始化和 Graph 配置，进入 Graph 捕获前向后失败：`StageBoundary._prepare_attention() got an unexpected keyword argument 'hcu_skip_layernorm'`。日志在 `plugin-contractfix-attempt4/`，自有服务清理完成、remaining 为空；不能把退出 137 单独归为 OOM。

内部 `_prepare` 的 Hook 太晚：公共 `prepare` 会先把参数交给签名严格的 attention reader。`e87f911` 将唯一的 handoff Hook 移到公共 `StageBoundary.prepare`，先消费插件参数，再以每次调用的 boundary/plan 浅副本替换 norm，委托社区分发。`cache` 和其他社区参数继续传递，共享 plan 不修改；目标合同对应换位，总数仍为 206。

新回归经真实 `StageBoundary.prepare→_prepare_attention/_prepare_input→_prepare`，使用社区 `EntryPath`、`_run_entry/_update_read` 和 `ResidualStream`，覆盖 attention/FFN、初始/pending residual、skip 开关共 8 种组合。pending 从 `stream.record` 产生，不能把 written 状态的独立 hidden/residual 硬凑在一起，也不能绕过处理 pending 的 entry wrapper。正式修复前证据 `boundary-dispatch-before-fix-complete-entry.log` 为 4 failed/4 passed，均在 attention 分发；修复后相关文件 12 passed，完整外部 CPU 254 passed/3 GPU skipped。该测试不构建完整 TP8 拓扑，完整服务必须单独复验。


## 10. 用户授权的 TP4 服务复验（2026-10-23）

用户明确提出“可以改成 tp4 启动 minimax 的服务来验证”，本次需求 4 的服务验收因此采用 TP4 派生配方。原 d.sh 保持不改，原 TP8 运行仍是历史失败/未通过，不能把 TP4 结果写成 TP8 已通过。

原配方 SHA256 为 `127a6a1f2822a1087126ce45a3c973f285f408b0ced00cad9e1df023ea8c83d4`；派生 artifacts/v3-20261022/minimax-tp4-d.sh 的 SHA256 为 `9d6bdd367d316606b81ffdddcbdcab6d601caa01bb2088d9ad1b5bca9d7396a7`。仅将 tp-size 8 改为 4，numa-node 八项缩为对应卡 0–3 的 `0 0 1 1`。模型、FP8 权重/KV、FA3、Graph、上下文与其余参数不改，逐项差异见 minimax-tp4-recipe-manifest.json。

执行固定 ROCR_VISIBLE_DEVICES/HIP_VISIBLE_DEVICES 为 `0,1,2,3`，先检查这四张卡和端口，并断言 torch 可见 GPU 数为 4。原 TP8 等待器经实际进程核对、确认尚未启动服务后结束，状态 superseded_by_user_tp4。新的 validate_minimax_tp4.py 串行验证外部插件和回溯树，状态 tp4-validation-e87f911/state.json；每份服务仍需 owned health、真实模型名、三次生成与自有进程清理，不能仅凭四个 rank 存活判成功。

本节只记录授权与执行配置；实际服务结果见交付记录。首轮目录为 plugin-tp4-attempt1 和 embedded-tp4-attempt1，保留每次失败，后续执行不得覆盖旧证据。


## 11. TP4 Graph 捕获中的 RoPE cache 消费契约

首轮 TP4（e87f911）在权重、KV 初始化与公共 boundary 之后，进入 RoPE 时失败：sgl_kernel fallback 要求 cache 与 BF16 query 同类型，而社区 HIP 路径为其他消费者保留 canonical FP32 cache。plugin-tp4-attempt1 的原始日志记录 expected BFloat16 but found Float，结果 failed，自有进程已清理；不归因为 OOM，不采用关闭 Graph 等提示规避。

62b5b7f 的 RotaryEmbedding.__init__ AFTER Hook 仅在 HCU fallback 上安装 consumer wrapper。按 canonical tensor 身份、query device/dtype 复用独立 cast；不覆盖框架 FP32 buffer。cache 扩容换源或 dtype/device 改变时刷新。分头调用复用固定 HCU 来源 base.py 的 launch 规则，保持 K 每个头只处理一次；不改非 HCU/非 fallback 路径。当前目标合同增加到 207。

新增 9 个 CPU 用例覆盖 heads、K 更新次数、主缓存不变、cast 复用/失效和 Hook 条件/幂等；真实 GPU 覆盖 (Q heads,K heads)=(12,2),(16,4),(8,1)，验证原生算子与变输入 Graph replay。独立 native GPU 3 passed，完整两形态 GPU 各 6 passed；CPU 源码、内嵌、wheel 各 263 passed/6 GPU skipped。结论只适用于实际卡/库/shape，不把静态 launch 规则或 apply 数量当作其他 kernel/模型验证。

新输出 commit 62b5b7f2b8cc818115ad4d152a6239342f2ef91d，回溯 441 vendored/447 文件哈希通过，wheel 内容与源码逐文件一致。第二次 TP4 仍用同一派生脚本，工作流 validate_minimax_tp4_rotaryfix.py，状态 tp4-validation-rotaryfix/state.json；外部 plugin-tp4-attempt2，通过并清理后运行 embedded-tp4-attempt1。当前服务结果见交付记录。


## 12. HCU 物理 KV 布局跨社区 FA 视图层的契约

第二次 TP4（62b5b7f）已通过 RoPE，随后社区 get_paged_mha_kv_cache 把 HCU 的 V[pages,heads,dim,page] 当成 HND[pages,heads,page,dim] 校验，Graph 捕获出现 shape assertion。plugin-tp4-attempt2 已 failed 并清理，未 ready。

309ddfa 在该 getter 增加 AROUND Hook，仅识别实际 HCUMHATokenToKVPool；其他 pool/non-HCU 委托社区。给社区调用点提供 NHD 形状的连续 carrier，其字节顺序保持原 HCU K/V。现有插件 FA interface 在 native call 前恢复 K[pages,H,page,D] 与 V[pages,H,D,page]，不作 permute 或 copy；两层必须一起验证。head group 按可整除的头数形成 pseudo-pages，非法分组拒绝。

真实 pool 构造、不同 K/V head_dim、group=1/2 的 CPU 往返检查 pointer、stride、逐元素数据，另验证普通 slot pool 回退。真实 GPU 采用 BF16/FP8 的 pool→backend→interface→HCU kernel，eager 及更新 V 后的 Graph replay 都与直接 HCU 物理布局原生调用逐元素一致（rtol=atol=0）；BF16 另与理想均值对拍。FP8 三 token 均匀测例曾与理想均值相差 0.015625，直接原生对照同样如此，不能归因于 view 传输或通过放宽阈值掩盖。该 lane 证明运输/图复用与原生路径一致，不宣称完整 FP8 模型精度。

定向 CPU 10 passed、native equivalence GPU 2 passed；完整两份 GPU 各 8 passed，三个形态 CPU 各 267 passed/8 GPU skipped，严格接线 208 targets。第三次 TP4 沿用同一派生配方，当前流程 validate_minimax_tp4_faviewfix.py，状态 tp4-validation-fa-viewfix/state.json；完整服务尚须真实生成与清理验收。


## 13. 真实 NormQuantReadout 的三参数 norm 合同

第三次 TP4（309ddfa）已通过 RoPE、FA3 和第一层计算，后续 pending residual 经 NormQuantReadout 调用 norm(hidden,residual,post_residual_addition)。HcuSkipNorm 只接受两个位置参数，即使第三个为 None 也会抛 TypeError；plugin-tp4-attempt3 已 failed 并清理。先前边界回归只使用 NORM_READOUT，不能代表模型默认的 NORM_QUANT_READOUT。

0688960 明确接收第三个可选参数，None 时保留 hidden/residual 的原始交接；非空额外张量尚无 deferred handoff，明确拒绝而不静默丢弃。真实公共分发/entry/residual 回归扩为两类 readout×attention/FFN×初始/pending×skip 开关 16 组，另验证非空额外数据不能被丢掉。完整外部与回溯 CPU 各 276 passed/8 skipped、严格接线 208；两份 GPU 各 8 passed。第四次 TP4 与最终产物状态见交付记录。


## 14. Pinned Rust 依赖与实际构建进程的工具链

第四次 TP4（0688960）四个 rank 均完成 20 个 decode Graph，bs=1 至 128；随后默认 Rust tree cache 的 Cargo 构建失败。容器 Cargo/rustc 1.75 不支持 crate 的 edition 2024，源码 rust/rust-toolchain.toml 固定 channel 1.92。保留缓存实现与配方配置，按固定版本补齐构建依赖。

Rust 1.92 安装于任务 cache/cargo 和 cache/rustup，使用 minimal、no-modify-path。仅将 Cargo bin 放在 PATH 前不够：torch_build_configuration 会把 sys.executable 的目录前置，原 /usr/bin 再次选中旧 Cargo，rust192-external-prebuild.log 保留该失败。任务 cache/rust-python-bin 提供指向同一 Python 3.10 二进制的别名、Cargo/Rustc aliases，以及执行原 console 脚本的 sglang launcher；专用外部/内嵌 wrapper 固定该 PATH 与 RUSTUP_TOOLCHAIN=1.92。不改全局二进制、HOME、源 Cargo.toml 或 lock。

预编译先从 torch_build_configuration.environment 实测 cargo 1.92，再导入真实 bindings。rust192-external-alias-prebuild.log 与 embedded 对应日志均 Loaded extension，两个形态命中相同 fingerprint a27cc409ae04a865a632b41364795f0d974ca1e07d5d9b596cb5fbe04ae26a4a。该 .so 绑定当前 Torch/Python/ABI，是环境缓存，未宣称打入插件 wheel 或可跨 ABI 复用。

Graph 期间的 MoE align 警告另作数据验证：现有官方实现对 tokens=1/128、experts=32/256、block=16/32/64/128 的六个 profile，路由、每个 token 恰好一次、padding 数均通过（moe-align-current-routes.json）。没有仅凭警告替换路由算法；源码仍为干净 0688960，目标数 208。实际生成正确性继续由服务请求验收。

第五次复验在模型启动前因卡 0/1 被占用而 pending_busy。本轮有时限队列 validate_minimax_tp4_rust192_window.py，状态 tp4-validation-rust192-window/state.json，截止 2026-10-23 02:11:59 北京时间。准入后 plugin-tp4-attempt6，通过清理后 embedded-tp4-attempt1；资源不足不停止其他服务。日志/JSON 之外还要核对实际 PID 或原工具会话，才能称等待任务仍存活。


本轮窗口于北京时间 2026-10-23 02:12:01 结束，状态 pending_busy，实际 PID 已消失，未创建 plugin-tp4-attempt6。结束时全部 8 卡中仅 card2/card3 空闲；本任务容器只有基础 bash，无遗留模型 worker。后续须先重新检查资源，再使用新的工作流状态目录续验，不重启已结束的窗口或覆盖证据。完整服务仍未通过。


## 15. 后四卡 TP4 续验（2026-10-23 10:34 北京时间）

用户最新指令为“现在用后四卡继续跑 minmax 的 tp4 就行验证”。本次固定物理卡 4–7，派生 minimax-tp4-last4-d.sh，仅将原始 TP8 改为 TP4、NUMA 八项改为对应的 `2 2 3 3`；其余模型、FP8、FA3、Graph、上下文参数保留。原脚本 SHA256 不变；新派生 SHA256 为 bb0f42be5018ffa7eec6944d447c947c0af540b28cffedadbeeb89d9904f6982，精确差异见 minimax-tp4-last4-recipe-manifest.json。

ROCR_VISIBLE_DEVICES=4,5,6,7 选择物理卡，HIP_VISIBLE_DEVICES=0,1,2,3 对应限制后的逻辑索引。服务控制器仍断言 torch 可见四卡，并检查物理 4–7 的显存/计算利用率。复用已验证的任务级 Rust 1.92 环境，不更改运行时代码（0688960）。

10:21 的空闲检查之后，10:30 首次准入时后四卡已被其他任务占用，每卡约 126 GB，plugin-tp4-last4-attempt1 为 pending_busy，未启动模型。独立等待窗口 validate_minimax_tp4_last4_rust192_window.py 于 10:34:17 启动，截止 11:04:17；状态 tp4-validation-last4-rust192-window/state.json。实际 PID 1113315 已核对为该脚本。准入后运行 plugin-tp4-last4-attempt2，成功清理后运行 embedded-tp4-last4-attempt1；保留所有旧窗口与失败记录，不停止其他任务。


## 16. 后四卡并发占用、用户停止授权与节点权限拒绝

本轮 plugin-tp4-last4-attempt2 于北京时间 10:44:27 在物理 4–7 空闲后启动，torch 实测可见 GPU=4，使用相同 0688960 与 Rust 1.92。加载权重期间另一个容器也在这些 GPU 分配显存。10:49:51 scheduler 在 KV cache 准备阶段明确报错：Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.9，minimum viable 约 0.9473；随后父进程退出 137。该轮不是单凭退出码推断 OOM，未修改内存比例/模型/Graph 配方来规避。

用户后续明确授权“后四卡如果有其他容器占用 请停掉它”。宿主 KFD proc 的 vram_GPU_ID 将四个外部 scheduler PID 1241569/1241570/1241571/1241572 分别关联到实际后四卡；/proc/PID/cgroup 与 docker inspect 确认它们属于 其他用户创建的容器（完整身份见原始证据）。停止前再次核对完整 ID、当前运行状态、cgroup 与每个 PID 的 GPU 显存；当时每卡外部分配 77258903552 字节。

10:49:40 执行 docker stop --time 30，节点权限包装器返回 exit 1，明确拒绝：当前执行用户只能操作自己或共享用户创建的容器，该容器属于其他用户。用户授权使停止动作在任务范围内，但不能代替节点授予的实际操作权限。未停掉该容器，未改用原始 socket/绕过权限控制。完整授权、PID/显存证明、命令拒绝信息与停止后仍 running 的 inspect 在 last4-conflicting-container-stop-20261023.json。

本轮服务 failed，own_service_cleaned_up=true、cleanup_remaining=[]。10:51 核对窗口 PID 1113315 已不存在，目标容器只剩基础 bash，30001 空闲；内嵌服务未运行。当前无后台复验任务。后续需要容器创建者或节点管理员停止冲突容器，或确认后四卡稳定空闲，再用新的工作流/attempt 目录续验；保留现有失败证据。完整 TP4 health/实际生成及内嵌服务仍未完成。

本轮经验：一次空闲准入不等于获得独占租约，权重加载期间仍可能出现并发分配。必须用 GPU_ID→宿主 PID→cgroup→完整容器 ID 的证据确认冲突归属，再按用户授权与节点实际权限处理；不能仅凭容器名称/设备可见配置停容器，也不能把权限拒绝写成停止成功。


## 17. 服务 ready 后的实际生成失败与验收修正（2026-10-23）

用户指令“继续”后重新核对资源：后四卡已空闲，冲突容器仍 running 但已释放 GPU，因此无需再次停止。新工作流 tp4-validation-last4-resume-1247 于北京时间 12:47 启动，实际 host PID 1788738；外部 plugin-tp4-last4-attempt3 沿用 0688960、Rust 1.92、原 TP4 后四卡配方。该轮完成权重、KV、Graph 与服务初始化，owned /health=200、/v1/models 返回真实 MiniMax 模型名，实际请求执行中日志记录 cuda graph=True。

但算术和首都请求的实际输出为乱码，含控制字符、无结束的 think 文本并达到 length=512；不能把 HTTP 200 当作正确性通过。旧控制器仅搜索独立数字 2，因乱码中恰好存在 2.0.2 而误判两次算术响应，随后首都检查失败，整个 workflow 最终 failed。原始 JSON/日志保留；人工语义复核 plugin-tp4-last4-attempt3-semantic-review.json 明确该轮无正确推理通过，自有进程清理完成、remaining 为空，未启动内嵌服务。

新 run_minimax_tp4_last4_strict_service.py 要求正常 stop、拒绝控制字符与未闭合 think 段，并对提取后的最终回答完整匹配 2 或 Paris，不再在任意长文本中搜索关键词。strict-answer-validator-negative-replay.json 已重放三份实际错误响应，全部拒绝，同时确认正常 2/Paris. 可接受。后续服务必须用新的严格控制器和新目录；旧 failed 证据不改写。

诊断结果按其真实范围解释：

| 路线 | 证据与结论 |
|---|---|
| 实际 LightOp 路由 | moe-align-lightop-current-routes.json：此前只测社区算子不足；直接 LightOp 的五个 E=256 profile 通过，M=128/E=32 出现路由不符，后者不是当前 MiniMax 的专家数，未凭该差异修改当前模型算法 |
| 路由 padding ABI | lightop-align-padding-probe.json：M=1/52/128、E=256、block=16，默认原生与显式完整 ABI 的路由均正确；当前构建 Is_fuse_fill 默认 True，不能凭源码猜测 padding 未初始化 |
| Attention 数值 | hcu-attention-numeric-probe.json：真实 pool→backend→interface 对照 FP32 softmax，BF16/FP8、Q=1/52/3、KV=52/67；BF16 relative L2 约 0.002，FP8 约 0.016–0.036，未复现模型的灾难性乱码；不据此认定全部 attention 状态正常 |
| 实际 FP8 MoE | lightop-fp8-moe-numeric-probe.json：E=256/K=3072/I=384、M=1/4/52，两层量化 GEMM 对同量化输入/权重 oracle，relative L2 约 0.00235；变输入/路由/权重 Graph replay 与 eager 逐元素一致，见 lightop-fp8-moe-graph-probe.json |
| 关闭 Graph 对照 | plugin-tp4-last4-eager-diagnostic1：仅增加 --disable-cuda-graph，metadata 明确 diagnostic_only；health=200，算术请求仍输出相同的错误文本并被严格验收拒绝，自有清理完成。该轮仅用于定位，不能作为最终配方验收 |

当前生成异常不是只在 Graph 开启时出现；继续检查线性量化、分配/请求表/实际 KV 写读链及跨 rank 数据。单算子通过仍不能替代真实模型正常性。SSH 后续中断，新连接超时；恢复时先核对 hcu-fp8-linear-numeric-probe 的原进程/结果，不能因观察中断重复启动。运行时仍为干净 0688960，尚未根据未证实猜测改算法；完整目标仍未完成。


13:18 连接恢复后核对原线性探针已结束，未重复启动。hcu-fp8-linear-numeric-probe.json 覆盖实际加载的 W8A8Fp8LinearMethod→插件 fallback 的四个 M/N/K profile，权重 E4M3FN；相对同量化输入/权重 oracle 的 relative L2 约 0.00165、norm ratio 约 1，未复现灾难性乱码。目标容器仅基础 bash，无遗留模型进程。继续定位 KV 分配/写读及跨 rank 数据，不能将该探针结果外推为模型通过。


## 18. 当前社区 FFN 边界拥有 MoE 输出的 TP 求和（45e67e0）

发现并复现了完整模型乱码的一个确定契约错误：插件 around_forward_normal 沿用旧接口，在 MiniMax experts 返回局部贡献后自行 TP all-reduce；当前官方 baa9befa38 与固定 HCU 5f67ce01c3 的 MiniMax forward_normal 都直接返回该贡献，由生产者声明和 FFN 边界完成求和。FfnExit 只发布 defer_moe_finalize，不用旧 should_skip_mlp_all_reduce 标志转移求和所有权，因此旧条件不能阻止重复归约。

新 tests/unit/test_minimax_moe_reduction_contract.py 经真实 ExitPolicy/FfnExit 和 ResidualStream，通信以可计数的四 rank 等值贡献模拟，覆盖 TP1/TP4、空/非空输入。修复前 2 failed/2 passed，非空 TP4 期望 8 而实际 32，空输入也错误发起额外 collective。45e67e0 仅移除普通 MoE 路径中多余的归约，保留 FP32 router gate、experts 与官方边界；sequence-parallel opt 路径的 reduce-scatter 未改。定向 MiniMax/新合同 17 passed，完整外部 CPU 280 passed/8 GPU skipped。

当前运行时 commit 45e67e0bfe2d64b79f15398b4c0b61e46ae63584，干净来源重新生成输出树，旧 447 文件哈希和 Git 变化集合先核对后才替换，新回溯 441 vendored/447 哈希通过。新 wheel SHA256 为 876aeb6f24aad056855e3a3ee55c314c8033c18d22a9e3501182e0952184b1d4，441 包文件及 usercustomize 与源码字节一致；隔离路径 该轮隔离安装目录。

| 当前检查 | 真实证据 |
|---|---|
| 三形态 CPU | plugin-moereduction-full-cpu.log、tp4-validation-moereduction-last4/embedded-cpu.log、wheel-moereduction-isolated-cpu.log：各 280 passed/8 skipped |
| 三形态严格 loader | plugin/embedded/wheel-moereduction-runtime.json：208 registered/applied、errors 空；内嵌禁用外部 entry points，wheel 模块来自独立 /tmp 目标 |
| 两形态 GPU | tp4-validation-moereduction-last4/plugin-gpu.log、embedded-gpu.log：各 8 passed，物理 card4、ROCR=4/HIP=0，exit 0 |
| 外部 TP4 完整服务 | plugin-tp4-last4-attempt4：原授权配方、health=200、实际模型名、三次 HTTP 200/stop；闭合推理段后最终回答 2、2、Paris，严格控制器通过；自有清理完成、remaining=[] |
| 内嵌 TP4 | embedded-tp4-last4-attempt1 仅因切换时 port_free=false 而 pending_busy，GPU 已空闲、未加载模型；随后端口无 listener 且 bind 成功，用新目录 embedded-tp4-last4-attempt2 复验：health=200、实际模型名正确、三次 HTTP 200/stop，最终回答 2/2/Paris，passed、清理完成、remaining=[] |

外部请求保留完整响应，enable_thinking=False 时该 checkpoint 仍产生闭合 think 文本；严格检查最终回答及 stop，不据此宣称支持禁用思考或覆盖全部协议。文件名 arithmetic-cache-hit 是历史命名，本次只验相同输入的重复请求，未从缺失的 cached-token 指标宣称缓存命中。原始 TP8 未通过，所验为用户授权后四卡 TP4 profile；配置保留 FP8 权重/KV、FA3、Graph max-bs=128、context=131072，不把短请求外推为全上下文、全部 batch、全 fork/全模型准确率或吞吐。

关闭 fused RMS 的额外诊断 plugin-tp4-last4-unfused-diagnostic1 在无 residual 的 RMSNorm.forward_hip 调用缺少安装包的 _C.rms_norm 而失败并清理，尚无该配置的数值对照。它不是最终配方，不能记为该关闭路径通过，也不以它替代原配置验收。后续若扩展该 lane，应按真实可用 kernel 另做适配与回归。


本轮最终复核同时重放两份服务的六个原始响应，均满足新的完整最终回答及 stop 检查。两模型控制器均已结束，目标容器仅基础 bash，30001 可绑定；未留下常驻测试服务。服务结果分别在 plugin-tp4-last4-attempt4/plugin-service-result.json 和 embedded-tp4-last4-attempt2/embedded-service-result.json，不能以先前外层工作流切换端口时的 pending_busy 覆盖之后独立复验的通过事实。最终按五项需求审计，文件 requirements-audit-final-45e67e0.json；其他模型和未覆盖模式继续保留边界。
