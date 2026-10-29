# hcu_sglang 开发迁移记录

> **本文用途**：持续记录本仓 `hcu_sglang` 每次代码迁移或开发完成的步骤、提交范围、实际验证与未完成条件。2026-10-28 起，本仓从仅维护方法推进到同时维护运行时代码；早期方法库限制不再禁止当前包落地。

## 1. 当前布局与接口

| 对象 | 当前定义 |
|---|---|
| 代码目录 / Python 包 | `hcu_sglang/` / `hcu_sglang` |
| distribution / 开发版本 | `hcu-sglang-plugin` / `0.3.2.dev14` |
| general plugin entry point | `hcu = hcu_sglang:register` |
| 显式服务入口 | `python3 -m hcu_sglang` 或 `hcu-sglang-serve`，参数传给官方 launcher |
| 来源插件版本 | `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c`，原包 `sglang_dcu_plugin` |
| 官方接口基线 | `baa9befa381511cd4d56f83980951e5a45fffc6d`，SGLang 0.5.22 |
| 来源清单 | [逐文件迁入快照](hcu-sglang-import-manifest.json)，用于本次迁入审计，不是未来文件完整性锁 |

复制来源仅为输出仓的运行时包。没有复制外部 docs、tools、tests 或其他生成脚本。包内 7 份不参与运行的来源说明 JSON 未放入运行时目录；源路径、哈希与排除理由保存在迁入清单。保留原生资源、继承的注释/docstring、版权与 vendor LICENSE。

原来的 `usercustomize.py` 位于来源包之外，未作为全局 Python 启动文件迁入。新增包内显式 launcher 先安装同一惰性 AITER shim，再转发官方服务；普通 Python 子进程及冷导入不自动加载 Torch/AITER。

## 2. 迁移与提交台账

| 日期 / 步骤 | 完成内容 | 提交记录 | 验证与边界 |
|---|---|---|---|
| 2026-10-28 / M01 运行时代码迁入 | 525 个来源文件；180 个 Python 文件仅改命名空间；239 个源 Python 文件 AST 对照；286 个原生资源/许可证按原字节保留 | [`4d82a16`](https://github.com/Sherlolo/hcu-sglang-plugin/commit/4d82a1663a54c5d4075c5215acc2d91dcfc66750) | 本次静态迁入对照通过；不把旧 GPU/模型结果记为新包验证 |
| 2026-10-28 / M02 安装与启动入口 | 本仓打包定义、`dcu` 入口、显式 launcher、命名空间可选严格加载工具、阶段门禁及当前使用文档 | 同一代码提交 `4d82a16` | 65 项仓库测试、源码/wheel 各 363/363 项 Hook 和 3 个具名接口通过；完整服务未执行 |
| 2026-10-28 / M03 HCU 发现入口 | entry point 改为 `hcu = hcu_sglang:register`；版本 `0.3.2.dev7`；同步当前安装与开发说明 | [`443eb83`](https://github.com/Sherlolo/hcu-sglang-plugin/commit/443eb83eb56d91ec5a86c1c53989419fb0077667) | 源码/wheel 各 363/363 项 Hook、3 个具名接口；旧 `dcu` 名称隔离通过 |

代码提交：`4d82a1663a54c5d4075c5215acc2d91dcfc66750`，标题 `feat: import HCU runtime as hcu_sglang with isolated packaging`，共 557 个变更文件。范围包括 526 个运行时包文件、根打包定义、迁入清单、当前使用文档与 Skill、结构门禁和运行验证工具/测试；未迁入输出仓的 docs、scripts 或 tests。其后提交 `c8ba45a` 仅回填该代码 SHA 并重建导览。

当前入口在 M03 改为 `hcu`；M01/M02 中的 `dcu` 与 dev6 验证结果保留历史含义。

当前代码来源是已存在实现的迁入，不是宣称整个 DAS fork 已完成转换。模型能力、Native、CP/EP/PD 的历史范围仍以对应来源记录为线索，本命名空间必须建立自己的产物与运行证据。

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
  --expected-targets docs/hcu-sglang-targets.json --output "$ARTIFACTS/hcu-runtime.json"
```

记录实际解释器、entry point、模块文件来源、目标集合与具名 seam、加载软错误，以及是否误导入旧包。当前 entry point 名为 `hcu`，由固定官方 loader 对 `SGLANG_PLUGINS` 中的名称精确匹配；不是 Python 包名后缀自动扫描，也不保留 `dcu` 别名。升级后须重新安装本仓或新 wheel 以刷新 metadata，再设置 `SGLANG_PLUGINS=hcu`。安装其他插件时同样核对实际白名单和模块来源，不能凭入口存在或目标数量认定通过。源码、隔离 wheel 和完整服务分别验收。

## 4. 实际验证记录

### M01/M02：首次迁入（dev6 / dcu）

验证日期：2026-10-28。实际环境为 Python 3.10.12 与既有 HCU/DTK 依赖容器；官方源码固定在上述 `baa9befa`。安装、构建和仓库测试使用源码所属用户；真实 Hook 加载使用容器原有的可访问设备的执行身份。没有修改设备权限或共享 site-packages。

| 检查 | 实际结果 | ignored 原始证据（相对仓库根） |
|---|---|---|
| 在线来源核验 | 525 个来源/目标 SHA256 一致；official、fork、来源 plugin 的 HEAD 与干净状态保持 | `artifacts/v3-20261028/provenance.json` |
| 结构及仓库测试 | L0：261 个 Python、8 个 Skill；65 项 unittest 通过 | `artifacts/v3-20261028/l0.log`、`tests.log` |
| 源码隔离安装与冷导入 | distribution `hcu-sglang-plugin==0.3.2.dev6`，唯一 `dcu = hcu_sglang:register`；冷导入未载入 Torch/AITER/旧包 | `artifacts/v3-20261028/source-install.log`、`source-cold-import.log` |
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
| 旧名称隔离 | 新进程设置 `SGLANG_PLUGINS=dcu`，0 项 Hook，新包未导入 | `old-name-isolation.log` |
| wheel 完整性 | 526 个包文件路径及字节与源码一致，无旧包或全局启动文件 | `runtime-validation.json` |
| 仓库门禁 | L0 与 65 项 unittest 通过；两个更新的 Skill、链接/锚点和导览快照校验通过 | `l0.log`、`tests.log`、`guide-qa.json` |

本次 wheel：`hcu_sglang_plugin-0.3.2.dev7-py3-none-any.whl`，SHA256：`5d61be21697b03c9c5e21cf26c7ec74a161357f7af15e93b0bf2c2e911740ecd`。新版本更新安装元数据后采用 `export SGLANG_PLUGINS=hcu`；dev6 的 `dcu` 结果不作为 dev7 的实际加载证据。本轮未执行模型服务、数值或 Graph 复验。

构建首次使用设备执行身份时被 Git 所有权校验拒绝，已改用源码所属用户完成构建；保留失败日志，未修改全局 Git 信任配置。

## 5. 后续代码开发与记录规范

每次代码改动追加台账，写明完成模块/行为、源码或前置提交、提交 SHA/标题及文件范围、实际执行的验证和原始证据、未验证模式、已知限制与下一步。提交后的代码 SHA 可在后续记录提交中回填，避免用文档自身未产生的 SHA 伪装已提交。

当前方法抽取、差异清单、处置、签名检查和 embedded 生成已支持当前命名空间。方法 profile 与实际类拆分仍单独预检；不因包名支持而自动执行会覆盖既有适配的方法同步。

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

原始反例：映射 `[0, 1, -1, -1, 2]` 下，输入 `-1` 被 Python 负索引映射为共享专家 `2`；修复后负数与超出全局表的 ID 均保持 `-1`。正常 routed/shared IDs、DP pad 的后置屏蔽、空 batch 和跳过映射分别验证。当前完整目标集合为 [当前合同](hcu-sglang-targets.json)；原迁入 manifest 的 363 项保持历史快照。

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
