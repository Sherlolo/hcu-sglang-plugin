# hcu_sglang 开发迁移记录

> **本文用途**：持续记录本仓 `hcu_sglang` 每次代码迁移或开发完成的步骤、提交范围、实际验证与未完成条件。2026-10-28 起，本仓从仅维护方法推进到同时维护运行时代码；早期方法库限制不再禁止当前包落地。

## 1. 当前布局与接口

| 对象 | 当前定义 |
|---|---|
| 代码目录 / Python 包 | `hcu_sglang/` / `hcu_sglang` |
| distribution / 开发版本 | `hcu-sglang-plugin` / `0.3.2.dev6` |
| general plugin entry point | `dcu = hcu_sglang:register` |
| 显式服务入口 | `python3 -m hcu_sglang` 或 `hcu-sglang-serve`，参数传给官方 launcher |
| 来源插件版本 | `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c`，原包 `sglang_dcu_plugin` |
| 官方接口基线 | `baa9befa381511cd4d56f83980951e5a45fffc6d`，SGLang 0.5.22 |
| 来源清单 | [逐文件迁入快照](hcu-sglang-import-manifest.json)，用于本次迁入审计，不是未来文件完整性锁 |

复制来源仅为输出仓的运行时包。没有复制外部 docs、tools、tests 或其他生成脚本。包内 7 份不参与运行的来源说明 JSON 未放入运行时目录；源路径、哈希与排除理由保存在迁入清单。保留原生资源、继承的注释/docstring、版权与 vendor LICENSE。

原来的 `usercustomize.py` 位于来源包之外，未作为全局 Python 启动文件迁入。新增包内显式 launcher 先安装同一惰性 AITER shim，再转发官方服务；普通 Python 子进程及冷导入不自动加载 Torch/AITER。

## 2. 迁移与提交台账

| 日期 / 步骤 | 完成内容 | 提交记录 | 验证与边界 |
|---|---|---|---|
| 2026-10-28 / M01 运行时代码迁入 | 525 个来源文件；180 个 Python 文件仅改命名空间；239 个源 Python 文件 AST 对照；286 个原生资源/许可证按原字节保留 | 完成提交后记录代码提交 SHA | 本次静态迁入对照通过；不把旧 GPU/模型结果记为新包验证 |
| 2026-10-28 / M02 安装与启动入口 | 本仓打包定义、`dcu` 入口、显式 launcher、命名空间可选严格加载工具、阶段门禁及当前使用文档 | 与当前迁入代码一起提交 | 65 项仓库测试、源码/wheel 各 363/363 项 Hook 和 3 个具名接口通过；完整服务未执行 |

当前代码来源是已存在实现的迁入，不是宣称整个 DAS fork 已完成转换。模型能力、Native、CP/EP/PD 的历史范围仍以对应来源记录为线索，本命名空间必须建立自己的产物与运行证据。

## 3. 当前安装与验证方式

先准备匹配的官方 SGLang、Torch/HCU、DTK 和 vendor 依赖；本次 wheel 不代替该环境。按 [配置管理](configuration.md#1-当前工具的路径配置) 提供真实本机路径，安装时使用隔离环境或 task target，避免同名旧入口重复加载：

```bash
# 从本仓根目录执行；仅安装当前插件，不补齐 vendor/HCU 依赖。
python3 -m pip install -e . --no-deps --no-build-isolation
export SGLANG_PLUGINS=dcu
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
  --expected-targets docs/hcu-sglang-import-manifest.json --output "$ARTIFACTS/hcu-runtime.json"
```

记录实际解释器、entry point、模块文件来源、目标集合与具名 seam、加载软错误，以及是否误导入旧包。发现旧/new distribution 同时提供 `dcu` 时，使用自己的干净环境，不能凭入口存在或目标数量认定通过。源码、隔离 wheel 和完整服务分别验收。

## 4. 本轮验证记录

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

## 5. 后续代码开发与记录规范

每次代码改动追加台账，写明完成模块/行为、源码或前置提交、提交 SHA/标题及文件范围、实际执行的验证和原始证据、未验证模式、已知限制与下一步。提交后的代码 SHA 可在后续记录提交中回填，避免用文档自身未产生的 SHA 伪装已提交。

当前旧版 `sync_deepseek_methods.py`、审计 profile 与 `reconstruct_sglang_das.py` 部分路径仍固定 `sglang_dcu_plugin`。本次不把新包直接宣称为这些旧工具的受支持输入；新命名空间的抽取/embedded 生成适配、完整 GPU/模型复验作为后续独立步骤。普通开发和严格加载已按实际命名空间选择，不能把旧抽取 profile 当成新布局。
