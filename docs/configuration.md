# 配置管理：路径、开关与运行时状态

> **本文用途**：区分工具参数、SGLang/adapter/plugin 配置的所有权，说明组合运行时的覆盖与记录规则。当前本仓没有配置服务或运行时环境变量注册表；下述运行时规范用于后续实现。

## 1. 当前工具的路径配置

| 名称 | 含义 | 要求 |
|---|---|---|
| `REPO` | 当前方法库 | 执行 L0、工具测试和脚本的位置 |
| `OFFICIAL` | 指定 SGLang 官方工作树 | 固定 SHA；与清单工具的 official-ref 匹配 |
| `FORK` | DAS 来源树 | 审计保留 dirty；不能作为正式运行时 import 来源 |
| `PLUGIN` | 外部含业务实现的 plugin 树 | 当前历史工具针对 `sglang_dcu_plugin` 布局 |
| `ARTIFACTS` | 本次审计/日志输出目录 | 建议位于 `REPO/artifacts/<task>`，不要落到来源树内 |
| `OUTPUT` | 源码回溯新输出树 | 不与任一输入相同、互为父目录或子目录 |

这些是手册中为方便阅读而使用的 shell 变量，不是本库自动读取的环境变量接口。实际 CLI 参数以工具 `--help` 为准；工具不会因为设置了 `PLUGIN` 就自动启用某个运行时插件。

## 2. 运行时配置所有权

| 配置来源 | 所有者 | plugin 接入原则 |
|---|---|---|
| `SGLANG_PLUGINS` 与框架参数 | pinned SGLang | 按该版本 loader/CLI 语义使用 |
| SGLang 自有 `Envs` / typed descriptor | pinned SGLang | 有对应 API 时使用 `.get()` / `.override()`，不在 Hook 中散落重复解析 |
| adapter 的配置/快关 | adapter | 引用其固定版本手册；不复制同名注册表或改写它的默认值 |
| plugin 自有能力开关 | 未来 plugin 实现 | 集中定义、记录类型/默认/支持条件，独立命名并可诊断 |
| NCCL / driver / launcher / OS 变量 | 外部依赖或部署 | 按所有者的接口消费，不擅自改变语义 |

本库目前不实现 `HCU_PLUGIN_*`、`HCU_DISABLE_OPS` 或统一配置文件。示例中旧插件的 `dcu` 来自外部实现仓，不能替换为尚未实现的本仓入口。

## 3. 新增配置的设计检查

| 检查项 | 要给出的答案 |
|---|---|
| 名称与类型 | 谁拥有；布尔/枚举/数值/字符串，取值范围是什么 |
| 默认与覆盖 | 未设置、显式设置和不支持环境下分别采用什么；显式用户值优先，非法或不支持值给出可诊断结果 |
| 生效阶段 | import 前、进程初始化、请求级还是 forward 级；修改后是否需重启 |
| 依赖与组合 | 需要哪个 adapter/vendor/official 版本；另一个仓是否影响同一状态 |
| 观察与回退 | 日志如何显示最终值、来源和选中路径；停用后回到哪条有效基线 |
| 测试 | 默认、显式覆盖、不支持/缺依赖、进程传播与恢复是否覆盖 |

优先用一个含义明确的配置，不增加彼此冲突的多个布尔项。使用目标版本实际存在的 API；不能因为 DAS skill 讲了 RuntimeContext 就假定旧 official 也具备同名接口。

## 4. 配置与状态分层

```mermaid
flowchart LR
    E[启动参数与环境] --> C[解析后的进程配置]
    C --> R[资源/stream/buffer 生命周期]
    R --> F[请求与 per-forward 状态]
```

配置、进程资源和每步状态分开。融合 KV 写入标记属于 batch/layer 的生命周期，不放模块全局跨请求复用；graph 的动态输入不能只在 Python capture 时赋值。若 pinned 版本有 publish/override 机制，按其生命周期写入并记录来源。

平台判定也有所有权：adapter 可能向 SGLang 暴露 CUDA 语义，而真实 torch/编译器仍是 HIP。plugin 的硬件能力判定应使用足够直接且被验证的设备/依赖信息，不能只凭某个 SGLang `is_cuda()` 推断物理设备与 vendor 符号。

## 5. 变更与排障记录

每次配置改动同步默认值说明、支持范围、测试与启动日志示例。发生冲突时先列出“原始值 → 解析值 → 最终值 → 实际分支命中”，再修改实现。验证模板见 [validation-record](../templates/validation-record.md)，详细状态约束见开发 skill 的 [runtime-and-env](../skills/hcu-plugin-dev/references/runtime-and-env.md)。
