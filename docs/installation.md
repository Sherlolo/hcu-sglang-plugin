# HCU 插件安装与启用指南

> **本文用途**：指导使用者安装当前 HCU 插件，确认真正加载，再用自己的模型配方启动服务。当前可安装本仓 `hcu_sglang`；外部 `sglang_dcu_plugin` 安装流程保留为历史版本参考。

## 使用 Skill 安装与验证插件

对应 Skill：[`hcu-plugin-install`](../skills/hcu-plugin-install/SKILL.md)。该 Skill 按本指南组织安装、metadata、设备和实际加载验证，分别记录各阶段结果。

**输入要求**：官方与插件版本、源码或 wheel 路径/哈希、目标解释器/容器、安装方式、安装及运行用户、可写安装/缓存目录和证据目录。仅要求服务验收时提供完整模型配方。

**调用示例**：将占位符替换为实际绝对路径与任务信息，提交给 Agent。

```text
读取 <方法库绝对路径>/skills/hcu-plugin-install/SKILL.md，按该 Skill 执行。
官方源码及版本：<路径和 SHA>。
插件产物：<源码路径及 SHA，或 wheel 路径及 SHA256>。
目标环境：<Python/容器>；安装方式：<wheel 或 editable>。
安装用户 / 运行用户：<实际身份>；安装及缓存目录：<本次可写目录>。
证据目录：<本次任务目录>；服务验收范围：<不要求，或模型与完整配方>。
核验解释器、distribution、入口和实际模块来源，完成适用的设备与加载检查。
分别记录各阶段状态，不将历史服务结果作为本次结论。
```

**预期交付**：安装与环境记录、入口信息、模块来源、加载报告及未执行项。若要求模型服务，另交付本次请求/响应与自有服务清理记录。

## 当前仓库安装入口（2026-10-28 起）

当前 distribution 为 `hcu-sglang-plugin`，包为 `hcu_sglang`，入口 `dcu = hcu_sglang:register`。在匹配的官方 SGLang/HCU 环境，从本仓根目录执行 `python3 -m pip install -e . --no-deps --no-build-isolation`；服务使用 `SGLANG_PLUGINS=dcu python3 -m hcu_sglang` 并追加真实模型配方参数。该显式启动入口安装惰性 AITER 兼容，不安装全局 `usercustomize.py`。

当前加载验证必须指定 `--plugin-package hcu_sglang`。命令、隔离安装、版本与实际验收见 [开发迁移记录](hcu-sglang-development.md)。不要在同一运行环境混装旧 `sglang-dcu-plugin` 与新包的同名 `dcu` entry point。

以下章节保留 v0.3.0/旧 `sglang_dcu_plugin` 的历史安装和复现方式，不代表当前包名或当前包已完成全部模型验收。

## 1. 运行环境与兼容基线

需要 Python 3.10+、兼容的官方 SGLang、HCU/DTK/PyTorch HIP 和该插件所用的 vendor 依赖。`--no-deps` 只安装插件，不准备 Torch、LightOp、AITER、vLLM 扩展或 Rust 工具链。普通 CPU 环境不等于 HCU 服务环境。

本次可复现组合如下；它是历史验证基线，不是最低版本或通用兼容声明。

2026-10-26 需求当前交付为 v0.3.2.dev5，官方基线 `baa9befa381511cd4d56f83980951e5a45fffc6d`、插件提交 `afc6205dd1a0f1979cfab9bf1f3272d2db14f46c`。完整范围见 [最终交付记录](v3-20261026-development.md#8-最终交付与需求验收)；下表保留 v0.3.0 历史证据。

本轮本地 wheel：`sglang_dcu_plugin-0.3.2.dev5-py3-none-any.whl`（历史相对位置 `output-code/dist/`，部署时通过 `WHEEL` 指定），SHA256：`c2260d30015e932faf33c09082758dee675113785353717a2750f75775544fac`。历史源码位于同级 `output-code/sglang-plugin`；该版本发布时本仓尚未提供运行时安装入口。在已准备配套 SGLang、Torch/HCU 与 vendor 依赖的个人环境中，可按后文 wheel 安装步骤使用该文件；`--no-deps` 不补齐这些依赖。本轮独立 wheel 已验证启动隔离、入口、363 项加载与 CPU 合约；GPU 和完整服务结果属于同提交的源码及内嵌形态，未执行 wheel 模型服务或公共 PyPI 发布。

| 对象 | 本次版本或位置 |
|---|---|
| 容器 | 历史验证容器；Python 3.10.12、torch 2.11 DTK |
| 官方 SGLang | `baa9befa381511cd4d56f83980951e5a45fffc6d`，v0.5.22 |
| 插件源码 | `45e67e0bfe2d64b79f15398b4c0b61e46ae63584`，v0.3.0 |
| wheel SHA256 | `876aeb6f24aad056855e3a3ee55c314c8033c18d22a9e3501182e0952184b1d4` |

各个依赖符号、构建缓存、模型验证范围见 [V3 记录](v3-runtime-validation.md)。换镜像或版本需要重新确认依赖和验收，现有 wheel 不包含全部 vendor 运行环境。本库尚未提供一条在任意新机器自动配置全部 HCU 依赖的安装命令。

在已有兼容环境中打开 shell。容器方式先设置本次 `CONTAINER`，在宿主执行 `docker exec -it "${CONTAINER:?请设置执行容器}" bash`。以下命令从方法库根目录、在**实际运行服务的同一环境**执行；先按 [路径配置](configuration.md#1-当前工具的路径配置) 加载本地 `.env`：

```bash
: "${REPO:?请从方法库根目录加载路径配置}"
: "${OFFICIAL:?请设置官方源码路径}"
: "${PLUGIN:?请设置外部插件源码路径}"
python3 --version
python3 -m pip --version
git -c safe.directory="$OFFICIAL" -C "$OFFICIAL" rev-parse HEAD
git -c safe.directory="$PLUGIN" -C "$PLUGIN" rev-parse HEAD
```

外部源码不要求采用同级目录布局。上述 Git 查询只对已确认的本次输入路径设置单命令信任，不改变全局 Git 配置。构建用户需有源码写权限；服务用户还需能实际访问 HCU，不能只凭安装成功推断。历史验证中，源码所属用户能安装但无法初始化 HCU，服务使用了具备设备访问权限的运行用户；UID 不是可复用的部署参数。按节点策略选择合法执行用户，不使用 sudo 或更改设备权限来绕过限制。

## 2. 安装方式与隔离配置

下面两种方式选择其一，均在新进程中验证。为避免改共享 site-packages，示例安装到自己的 task target；服务必须保留相应 PYTHONPATH。临时目录适合验证，长期部署请将 SITE 改为自己可写的持久路径。

```bash
SITE=$(mktemp -d)
export XDG_CACHE_HOME="$SITE/cache"
export TORCHINDUCTOR_CACHE_DIR="$SITE/cache/inductor"
export TRITON_CACHE_DIR="$SITE/cache/triton"
export AITER_JIT_DIR="$SITE/cache/aiter"
export TORCH_EXTENSIONS_DIR="$SITE/cache/torch_extensions"
export SGLANG_CACHE_DIR="$SITE/cache/sglang"
export SGLANG_JIT_CACHE_DIR="$SITE/cache/sglang-jit"
```

这些是任务缓存路径，不修改 HOME；尤其适用于容器内没有 passwd 记录的 UID。AITER 使用独立的 AITER_JIT_DIR，不能只设置 XDG_CACHE_HOME 就认为所有 vendor 都会遵循。不同时把多个同名源码/安装目录加入路径。

### 2.1 Wheel 安装

运行时 wheel 需从交付方获取并单独保管；Git clone 本方法库不会带上它。将 `WHEEL` 设置为本次产物文件，确认文件存在且哈希与交付方一致后，在所选环境安装：

```bash
: "${WHEEL:?请设置本次安装的 wheel 文件路径}"
test -f "$WHEEL"
sha256sum "$WHEEL"
python3 -m pip install --no-deps --target "$SITE" "$WHEEL"
```

让配套官方源码优先于镜像中的其他 SGLang；插件从当前 Python 的安装位置加载：

```bash
export PYTHONPATH="$OFFICIAL/python:$SITE"
export SGLANG_PLUGINS=dcu
```

若使用已正确安装的 pinned 官方 SGLang，可不设置此 PYTHONPATH，但下一步仍需核对来源。wheel 验证时不要再添加 `$PLUGIN`，否则会验成源码。

### 2.2 Editable 安装

editable 构建请在源码所属用户的 shell 执行。容器方式可用 `docker exec -it --user "${CONTAINER_USER:?请设置合法的 UID:GID}" "${CONTAINER:?请设置执行容器}" bash`；在该 shell 重新加载 §1 的路径并设置 §2 的 SITE/缓存后执行下面的安装命令。root 直接构建这份他人所属 Git 树可能被构建后端拒绝，不要照错误提示修改全局 Git 配置。

```bash
python3 -m pip install --no-deps --no-build-isolation --target "$SITE" -e "$PLUGIN"
export PYTHONPATH="$OFFICIAL/python:$PLUGIN:$SITE"
export SGLANG_PLUGINS=dcu
```

target 中的 editable `.pth` 不会仅因 PYTHONPATH 自动执行，因此这里显式加入 `$PLUGIN` 源码路径，并保留 `$SITE` 中的入口 metadata。`--no-build-isolation` 要求当前环境已有 setuptools 等 build-system 所需工具。editable 只适用于含 `pyproject.toml` 的运行时仓，源码修改会影响后续进程；不要对本方法库执行这条安装命令。

若安装用户和运行用户不同，记下上面实际的 `$SITE`；进入节点允许且已确认有 HCU 访问能力的运行用户 shell，在**同一容器、同一 Python 环境**重新设置 §1 的路径、§2 缓存，并使用安装时的 SITE（不要重新 mktemp）。wheel 使用 `$OFFICIAL/python:$SITE`，editable 使用 `$OFFICIAL/python:$PLUGIN:$SITE`。源码及该安装目录需对运行用户可读。后面的设备检查和加载在运行用户中执行。

## 3. 安装、入口与加载验证

先检查包、源码和 entry point；这一步不会启动模型：

```bash
python3 - <<'PYCODE'
import sys
from importlib.metadata import distribution
import sglang
import sglang_dcu_plugin
pkg = distribution("sglang-dcu-plugin")
eps = [ep for ep in pkg.entry_points if ep.group == "sglang.srt.plugins"]
print("python:", sys.executable)
print("official:", sglang.__file__)
print("plugin:", sglang_dcu_plugin.__file__)
print("version:", pkg.version)
print("entry points:", [(ep.name, ep.value) for ep in eps])
assert any(ep.name == "dcu" and ep.value == "sglang_dcu_plugin:register" for ep in eps)
PYCODE
```

期望入口为 `dcu -> sglang_dcu_plugin:register`。源码方式的 plugin 路径应位于 `$PLUGIN`；wheel 应位于 `$SITE/sglang_dcu_plugin`。

先检查实际运行用户的 HCU 访问能力：

```bash
python3 - <<'PYCODE'
import torch
print("HIP:", torch.version.hip, "available:", torch.cuda.is_available(),
      "visible:", torch.cuda.device_count())
assert torch.version.hip and torch.cuda.is_available(), "运行用户尚不能初始化 HCU"
PYCODE
```

这里只查询设备，不加载模型。若失败，安装/metadata 可以已通过，但严格 DCU 加载与服务验证应保留未通过；检查合法运行用户、设备映射与容器权限，不用 mock 硬件检测掩盖。

在本次依赖齐全的运行环境检查真正的 register/apply。此检查会导入运行时与 native 依赖，但不加载模型；其他版本应使用它们自己的目标合同：

```bash
# editable 用这一行；wheel 改用 PLUGIN_ROOT="$SITE"
PLUGIN_ROOT="$PLUGIN"
python3 "$REPO/tools/validate_plugin_runtime.py" \
  --official-python "$OFFICIAL/python" \
  --plugin-root "$PLUGIN_ROOT" \
  --expected-targets "$PLUGIN/tests/contracts/targets_v0_5_22.json" \
  --output "${ARTIFACTS:?请设置本次证据目录}/runtime.json"
```

wheel 的 `PLUGIN_ROOT` 是包含 `sglang_dcu_plugin/` 的 site-packages 或隔离 target 目录，不是包内子目录。该工具专用于当前 DCU 实现及其三个具名 seam，不能用于任意新插件。

首次加载可能编译 AITER 等扩展，冷缓存耗时应与注册错误分开；不要以固定的短超时把编译判为安装失败。观察中断时先核实原进程与结果，不并发重启同一个构建。

本次合同要求 208 个 targets 全部 applied、三个具名 seam 齐全、`errors=[]`。日志有 soft error、空集合或来源错误都不是通过。**入口可见、加载成功、服务生成正确是三项独立检查。**

## 4. 模型服务启动与验收

保持上述环境，使用与模型匹配的完整配方。以下仅是命令形态，`MODEL` 必须由使用者填写；默认参数不保证适合你的 HCU 模型：

```bash
: "${MODEL:?请设置本次模型目录}"
python3 -m sglang.launch_server --model-path "$MODEL" --port 30001
```

先确认分配的 GPU 和端口可用，保留实际模型、TP、backend、量化、Graph 等全部参数。MiniMax M2.5 本次完整配方与 TP4 派生过程见 [V3 记录](v3-runtime-validation.md) §10、§15–§18，不用上面的最短命令替代。只有需要构建该扩展时，才按记录准备对应工具链；不是所有插件都要求 Rust 1.92。

服务日志应确认 plugin 被加载、所有 rank 初始化并 ready；检查该端口确属自己的服务、`/health`、`/v1/models` 和真实请求。保存完整响应、finish_reason 和答案；确定性回答可按 [响应校验](validation.md#7-确定性生成响应的离线验收) 验收。health 200 不代表生成正确，测试后清理自己创建的服务。

## 5. 停用、卸载与常见问题

`SGLANG_PLUGINS` 是逗号分隔的入口白名单。**在本次 pinned SGLang 中，未设置或空字符串表示发现所有插件，并不表示停用。** 若只测试框架、不加载任何 general plugin，可在新进程使用一个确认不存在的名称，例如 `SGLANG_PLUGINS=__no_general_plugins__`。这不是 HCU 上一定能独立跑服务的承诺；保留当前部署需要的 adapter/platform 基线。

本页隔离 target 不改共享安装；不用它时从服务环境移除该 SITE/源码路径，并停用对应入口。若安装在个人虚拟环境的标准 site-packages，卸载命令是 `python3 -m pip uninstall sglang-dcu-plugin`。不要在共享环境贸然卸载；停用和安装变化对已经运行的进程无效，需新进程验证。若同时用 adapter，按 [治理](governance.md) 验证固定版本、加载顺序和 fallback。

| 失败 | 下一步 |
|---|---|
| 找不到 wheel 或对方法库 pip install 失败 | 检查安装对象；向交付方获取运行时源码/wheel |
| 包可 import 但没有 `dcu` 入口 | 检查同一 Python 的 distribution metadata，避免只有 PYTHONPATH 没安装 |
| 入口可见但 register/apply 失败 | 看首条 soft error、真实来源、依赖符号和目标合同 |
| 模型启动失败、乱码、hang | 进入 [排障入口](troubleshooting.md)，不反复重装掩盖错误 |

下一步：使用者保留安装/验收结果即可；要开发新能力请转 [最小案例](first-plugin.md) 或 [开发指南](plugins.md)。
