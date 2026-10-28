# 插件工程初始化指南

> **本文用途**：通过无需模型加载或 GPU 分配的示例，说明包创建、安装、发现、Hook 应用、启停与行为验证。面向第一次接触插件的新开发者；Agent 主入口是 [hcu-plugin-scaffold](../skills/hcu-plugin-scaffold/SKILL.md)。

## 使用 Skill 创建新插件

对应 Skill：[`hcu-plugin-scaffold`](../skills/hcu-plugin-scaffold/SKILL.md)。该 Skill 根据本指南在外部目录创建插件工程，完成打包定义、入口注册和最小行为验证；已有工程的能力扩展使用 [开发 Skill](../skills/hcu-plugin-dev/SKILL.md)。

**输入要求**：官方源码路径及 SHA、方法库外的工程目录、插件命名、首项功能、Python/容器环境和证据目录。

**调用示例**：将占位符替换为实际绝对路径与任务信息，提交给 Agent。

```text
读取 <方法库绝对路径>/skills/hcu-plugin-scaffold/SKILL.md，按该 Skill 执行。
官方源码：<official 绝对路径及 SHA>。
外部工程目录：<不属于方法库或官方源码树的新目录>。
distribution / Python 包 / entry point：<三个名称，或按功能提出命名>。
首项功能：<功能描述；学习时可采用本文 add_prefix 示例>。
执行环境：<Python 或容器>；证据目录：<本次任务目录>。
创建独立工程，完成隔离安装、入口发现、Hook 应用、实际调用与三个启停对照。
仅在要求真实 HCU 功能时衔接 hcu-plugin-dev，按本次范围验证。
```

**预期交付**：外部工程路径、命名与入口映射、模块来源、可重放命令、三个对照结果及未验证项。工程初始化不等同于模型精度或性能验收；本示例生成到独立目录，已有 HCU 实现维护在本仓 `hcu_sglang/`。

## 1. 适用范围与代码归属

需求：给 SGLang 的 `add_prefix(name, prefix)` 增加可关闭的诊断日志，**保持原返回值不变**。例如 `add_prefix("weight", "layer")` 始终返回 `layer.weight`，启用 demo 时额外记录一次调用。该字符串函数用于说明插件接入机制，不涉及 HCU kernel 优化或性能评估。

本案例基于官方 `baa9befa381511cd4d56f83980951e5a45fffc6d` 的真实 `sglang.srt.plugins` / `HookRegistry` API。目标为 `sglang.srt.utils.common.add_prefix`。其他版本先检查目标和签名是否存在。

| 任务类型 | 实现位置 |
|---|---|
| 学习创建一个独立插件包 | 本文的新临时目录，包 `demo_prefix_plugin`，入口 `demo_prefix` |
| 给已有 `dcu` 加能力 | 本仓 `hcu_sglang/`；在对应功能模块实现，沿 `hcu_sglang:register` 接入，并更新 [开发迁移记录](hcu-sglang-development.md) |
| 改开发指引或工具 | 当前方法库 `hcu-sglang-plugin` |

本仓维护正式 HCU 实现；教学 demo 使用独立目录，不添加到本仓根目录。已有 `dcu` 的能力扩展复用当前 wheel。

## 2. 前置条件与工程结构

在已具备 pinned SGLang 依赖的开发环境执行（容器进入方式见 [安装](installation.md)）。Python 3.10+；虽然不使用 GPU，但导入 SGLang 的 common 模块仍需其 Torch 等依赖。先按 [路径配置](configuration.md#1-当前工具的路径配置) 设置 `OFFICIAL`，选择有写权限的新目录；以下 `mktemp` 不覆盖现有文件：

```bash
: "${OFFICIAL:?请按路径配置设置官方源码目录}"
DEMO=$(mktemp -d)
mkdir -p "$DEMO/demo_prefix_plugin" "$DEMO/check"
cd "$DEMO"
# 无 passwd 记录的容器 UID 也使用自己的缓存，不依赖 HOME
export XDG_CACHE_HOME="$DEMO/cache"
export TORCHINDUCTOR_CACHE_DIR="$DEMO/cache/inductor"
export TRITON_CACHE_DIR="$DEMO/cache/triton"
```

创建打包定义。distribution 名、Python 包名、entry point 名各有用途；`SGLANG_PLUGINS` 选择的是 `demo_prefix`：

```bash
cat > pyproject.toml <<'TOML'
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "sglang-prefix-demo"
version = "0.0.1"
requires-python = ">=3.10"

[project.entry-points."sglang.srt.plugins"]
demo_prefix = "demo_prefix_plugin:register"

[tool.setuptools.packages.find]
include = ["demo_prefix_plugin*"]
TOML
```

创建插件。AROUND 收到的 `original_fn` 是当前有效基线，保留参数、返回值与异常；不在导入时执行注册、不手工调用 apply：

```bash
cat > demo_prefix_plugin/__init__.py <<'PYCODE'
import logging
import os

TARGET = "sglang.srt.utils.common.add_prefix"
logger = logging.getLogger("demo_prefix_plugin")


def trace_prefix(original_fn, *args, **kwargs):
    result = original_fn(*args, **kwargs)
    if os.environ.get("DEMO_TRACE_PREFIX", "0") == "1":
        logger.info("[demo-prefix] result=%s", result)
    return result


def register():
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    HookRegistry.register(TARGET, trace_prefix, HookType.AROUND)
PYCODE
```

创建 smoke，验证真实 loader 及目标行为。`_hooks/_patched` 是该固定版本的内部诊断字段，不应当作稳定业务 API；版本升级时同步调整测试：

```bash
cat > check/smoke.py <<'PYCODE'
import io
import logging
import os
from importlib.metadata import entry_points

from sglang.srt import plugins
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.srt.utils import common

TARGET = "sglang.srt.utils.common.add_prefix"
eps = entry_points(group="sglang.srt.plugins")
assert any(ep.name == "demo_prefix" for ep in eps), "缺少安装 metadata"
capture = io.StringIO()
handler = logging.StreamHandler(capture)
log = logging.getLogger("demo_prefix_plugin")
log.setLevel(logging.INFO)
log.addHandler(handler)
plugins.load_plugins()
active = os.environ["EXPECT_PLUGIN"] == "1"
assert (TARGET in HookRegistry._hooks) == active
assert (TARGET in HookRegistry._patched) == active, "目标未应用"
assert common.add_prefix("weight", "layer") == "layer.weight"
assert common.add_prefix(name="weight", prefix="") == "weight"
try:
    common.add_prefix("weight", "layer", unexpected=True)
except TypeError:
    pass
else:
    raise AssertionError("原异常被吞掉")
traced = os.environ.get("DEMO_TRACE_PREFIX") == "1" and active
assert capture.getvalue().count("[demo-prefix]") == (2 if traced else 0)
print("PASS", "plugin=" + str(active), "trace=" + str(traced))
PYCODE
```

## 3. 隔离安装与启停验证

在临时目录做隔离 target 安装，不修改共享 site-packages。`--no-build-isolation` 需要环境已有 setuptools；site 是本次新目录。

```bash
python3 -m pip install --no-deps --no-build-isolation --target "$DEMO/site" .
export PYTHONPATH="$DEMO/site:$OFFICIAL/python"
SGLANG_PLUGINS=demo_prefix DEMO_TRACE_PREFIX=1 EXPECT_PLUGIN=1 python3 check/smoke.py
SGLANG_PLUGINS=demo_prefix DEMO_TRACE_PREFIX=0 EXPECT_PLUGIN=1 python3 check/smoke.py
SGLANG_PLUGINS=__no_general_plugins__ DEMO_TRACE_PREFIX=1 EXPECT_PLUGIN=0 python3 check/smoke.py
```

预期依次是 `PASS plugin=True trace=True`、`PASS plugin=True trace=False`、`PASS plugin=False trace=False`。第一行证明入口发现、目标应用和实际调用都生效；第二行证明功能关闭后保留原行为；第三行证明没有加载 demo 时仍使用基线。每行必须启动新 Python，因为 loader 在进程内只执行一次。

在本次版本中，空 `SGLANG_PLUGINS` 表示加载所有发现的插件，不能用空值测试停用。最后一行的名称应确认没有对应已安装入口。示例不选择 `dcu`，不验证 HCU 服务，也不证明 adapter 组合兼容。

## 4. HCU 功能开发衔接

接入链验证完成后，参考 [开发指南](plugins.md)，使用 [hcu-plugin-dev](../skills/hcu-plugin-dev/SKILL.md)：

1. 写清能力和支持范围，例如只优化某 dtype/shape 的 norm，其他情况回退；确认上游已有 seam，适合时优先使用。
2. 在运行时实现仓新增功能模块，读取真实 caller、返回值、状态和下一消费者。可选 vendor 在选中路径内导入；不能把 demo 的任意参数包装误当成已经满足真实签名。
3. 沿已有顶层 register 接入，核对完整目标合同。已有 `dcu` 可用 [严格加载工具](../tools/validate_plugin_runtime.py)；该工具不能直接拿来验证本文 demo。
4. 先测支持/回退、参数和状态，再测相关 HCU 数值、Graph、TP 和模型请求。涉及 collective、KV、norm 交接时需穿过真实框架边界。
5. 写 [验证记录](../templates/validation-record.md)，说明真实完成范围；需要 wheel 或 embedded 交付时分别验证实际来源和行为。

例如新增 HCU norm 的检查表应包含“支持的 dtype/shape 命中新路径、其他情况仍调用原路径、输出/残差一致、Graph replay 变输入正确”。具体 kernel 和阈值需按实际依赖与模型确定，不能从本字符串例子推出。

## 5. 常见问题与诊断入口

| 现象 | 先检查 | 然后使用 |
|---|---|---|
| 安装失败 | Python、setuptools、目录权限，是否在 demo 根目录 | [安装](installation.md)、[环境](bootstrap.md) |
| 缺少安装 metadata | pip 和运行脚本是否同一 Python、site 是否在 PYTHONPATH | [排障入口](troubleshooting.md) |
| 入口可见但目标未应用 | loader 第一条错误、模块路径、目标名和该版本 API | [排障 Skill](../skills/hcu-plugin-triage/SKILL.md) |
| 目标已应用但行为不对 | 调用是否命中包装、返回值/参数、先前别名与组合 | [排障入口](troubleshooting.md) |

三个对照完成后保留 `$DEMO` 作为学习产物即可；`site` 只通过本 shell 的 PYTHONPATH 使用，关闭该 shell 后不会自动启用。继续真实开发时重新按 [安装手册](installation.md) 设置运行时环境，避免沿用 demo 白名单。
