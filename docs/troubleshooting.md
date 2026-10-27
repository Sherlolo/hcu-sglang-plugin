# 插件故障诊断指南

> **本文用途**：从症状开始选择检查与 Skill；不要求先读完全部规约。已有日志时使用 [hcu-plugin-triage](../skills/hcu-plugin-triage/SKILL.md)，不需要重新走插件创建流程。

## 使用 Skill 执行故障诊断

对应 Skill：[`hcu-plugin-triage`](../skills/hcu-plugin-triage/SKILL.md)。该 Skill 按症状定位首个失败边界并建立最小复现；仅安装/metadata 问题先使用 [安装 Skill](../skills/hcu-plugin-install/SKILL.md)，需要代码修复时衔接开发 Skill。

**输入要求**：版本与模块路径、完整配置、原始日志、请求/响应、复现步骤、基线或启停对照、预期行为及证据目录。

**调用示例**：将占位符替换为实际绝对路径与任务信息，提交给 Agent。

```text
读取 <方法库绝对路径>/skills/hcu-plugin-triage/SKILL.md，按该 Skill 执行。
版本与路径：<official/plugin/可选 adapter>。
执行环境与完整配置：<描述或配方路径>。
日志：<路径>；请求/响应：<路径，非服务问题说明不适用>。
复现步骤与预期行为：<描述>；基线/启停对照：<已有信息>。
证据目录：<本次任务目录>。
定位最早失败层，形成最小复现；根因明确后完成针对性修复与回归验证。
保留原失败记录，诊断配置与原验收配置分别记录。
```

**预期交付**：故障分类、首次分叉与根因、最小复现、修复前后证据及未覆盖范围。权限或资源阻塞按真实状态记录。

## 1. 诊断输入与证据要求

记录实际 Python/模块路径，official/plugin/可选 adapter 版本，完整启动配方，最早错误及各 rank 日志，原始请求和响应。证据放新的 `artifacts/<任务名>/`；保留旧失败，不通过重装、切换参数或覆盖日志抹掉现场。

## 2. 症状分类与诊断路径

| 故障现象 | 初步检查 | 诊断参考与操作 |
|---|---|---|
| pip 安装失败、找不到包 | 是否装了方法库而非实现仓；Python/权限/build 依赖 | [安装步骤](installation.md)、[环境](bootstrap.md) |
| import 成功却没有入口 | 同一 Python 的 distribution entry points；只有 PYTHONPATH 不等于已安装 metadata | [安装检查](installation.md#3-安装入口与加载验证) |
| 有入口但功能不生效 | SGLANG_PLUGINS、实际 module.__file__、register/apply 首条错误、目标合同 | 先严格加载，参考 [最小案例](first-plugin.md) 的实际调用断言 |
| 启动时 TypeError / 参数不识别 | 当前签名、receiver、可选/keyword-only 参数；异常发生在 Hook 前还是内部 | [开发经验](development-methodology.md) 的接口与分发案例 |
| 导入/编译卡住，GPU 尚未工作 | 子进程、JIT 锁、实际 builder 环境和缓存 | [runtime-and-env](../skills/hcu-plugin-dev/references/runtime-and-env.md) |
| 构造 KV / Graph 时失败 | 第一条异常及构造/分发/视图链，不只看最后退出码 | [历史经验](development-methodology.md)、[precision-and-graph](../skills/hcu-plugin-triage/references/precision-and-graph.md) |
| health 200，但乱码、答案错 | 先验证完整最终答案、stop、截断和模型身份，再找首次数值/状态分叉 | 下节诊断案例；[precision-and-graph](../skills/hcu-plugin-triage/references/precision-and-graph.md) |
| eager 正常，Graph 才错 | capture 后换输入/位置/cache 的 replay；比较同一请求 | [precision-and-graph](../skills/hcu-plugin-triage/references/precision-and-graph.md) |
| 多 rank hang / crash | 是否先有 rank 退出；各 rank 最后步骤、collective 次数/group/shape | [distributed-and-performance](../skills/hcu-plugin-triage/references/distributed-and-performance.md) |
| 吞吐下降 | 固定形状、预热、同步、并发、频率，比较实际命中的实现 | 同上性能路线；不要把首次 JIT 当稳态性能 |
| 只有 adapter＋plugin 组合出错 | 固定两仓版本、同目标包装顺序和 fallback 是否绕过适配 | [治理](governance.md)、[配置](configuration.md) |

若严格加载报告只有 `Failed to execute general plugin: dcu`，在同一已配置环境用新进程打开日志，找第一条 traceback：

```bash
python3 - <<'PYCODE'
import logging
logging.basicConfig(level=logging.INFO)
from sglang.srt.plugins import load_plugins
load_plugins()
PYCODE
```

本次框架会记录部分加载错误后继续，所以上面命令退出 0 不表示通过；修复后仍需重跑严格加载合同。典型例子是 AITER 单独使用自己的 JIT 目录，XDG_CACHE_HOME 正确也可能还在访问不可写的默认路径，按 [安装手册](installation.md) 配齐任务缓存。

## 3. 诊断案例：TP4 生成异常与重复归约

这是本次实际发生的案例，详细原始记录见 [V3](v3-runtime-validation.md) §17–§18；下面展示如何使用文档和 Skill，而非要求所有乱码都按这个根因修复。

1. **确认错误是真的。** 保存原始响应。旧检查只搜数字 `2`，乱码里碰巧出现 `2.0.2` 也误过。对确定性问题，按 [响应验收](validation.md#7-确定性生成响应的离线验收) 使用 `validate_generation_response.py` 检查完整答案和正常结束，并回放坏响应确认它被拒绝。
2. **选择排障 Skill。** 读取 `skills/hcu-plugin-triage/SKILL.md`，给它原配方、日志、请求/响应和版本；先看 `precision-and-graph`。关闭 Graph 的诊断仍然错，只说明这次不只依赖 Graph；保留原 Graph 配方作为最终验收项。
3. **缩小到真实边界。** 模型实际 shape/provider 的算子对拍未出现灾难误差，继续追踪 MoE 产出和 FFN 出口。发现旧插件先求和，而当前 FFN 出口又求和。不是直接认定 vendor kernel 有问题。
4. **切到开发 Skill 修复。** 读取 `hcu-plugin-dev` 的 `references/test-contract.md`。回归保留真实 `FfnExit/ResidualStream`，mock 通信但不 mock 出口，TP4 预期 8 得到 32，修复前失败。
5. **最小修复与复验。** 删除旧方法多余的归约，保持当前框架出口所有权；同一回归通过。再用原接受配置分别验证源码插件与 embedded 服务，完整答案正确，保存本次结果并清理自有进程。

如果你的问题在第 2 步已确认仅 Graph 失败，应继续 Graph 路线；若只在 TP>1 出现，重点查看归约和 rank 分支。案例提供查法，不预设根因，也不把三次短回答当成完整准确率测试。

## 4. 修复验收标准

有能复现问题的输入、明确首个失败边界、针对根因的回归、修复后原接受配置的验证结果。关闭优化或降级配置只用于诊断，不替代原要求；无法执行的项保留未验证。按 [验证记录模板](../templates/validation-record.md) 写出结果。

共享节点先确认自有资源；只有用户已授权且节点权限允许时才处理已核实的外部冲突。观察中断后先核验原进程，避免重复加载。具体资源恢复见 [分布式参考](../skills/hcu-plugin-triage/references/distributed-and-performance.md)。
