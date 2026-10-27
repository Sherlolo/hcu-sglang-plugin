# vLLM 发版 skill 向 SGLang 的迁移分析

> **本文用途**：说明 2026-10-22 从用户指定 `upgrading-vllm-hcu-upstream-first` 中采用的方法、SGLang 改写和未迁移边界。新入口为 [SGLang 升级与发版 skill](../skills/upgrading-sglang-hcu-upstream-first/SKILL.md)，本次不是运行时升级或硬件发版验收。

## 1. 来源与选择依据

来源为历史验证节点同级目录的本地快照；重点读取通用升级/策略/选测/兼容/构建/验收/CI/评测，以及 attention/KV、AITER、FLA 与 SGLang-Qwen 差异方法，并检查原预检脚本。逐文件源字节哈希、迁移落点与排除原因保存在 [source-map.json](../skills/upgrading-sglang-hcu-upstream-first/references/source-map.json)。历史模型专题仅索引，不声称重验其历史日志。

当前仓已具备审计、Hook 检查、源码回溯和排障；缺少贯穿“指定目标→上游接管→真实交付产物→发版闭合”的统一入口，因此新增一个 skill，详细内容按条件加载，不重复复制整套 vLLM references。

## 2. 有用方法与 SGLang 落点

| 来源主题 | 保留的方法 | SGLang 改写/落点 |
|---|---|---|
| upgrade-strategy / upgrade-workflow | 固定任意任务目标、上游先接管、patch-id/range-diff 辅助去重、临时补丁退出条件 | [升级与接管](../skills/upgrading-sglang-hcu-upstream-first/references/upgrade-workflow.md)；沿用三方工具与九类处置 |
| continuous-validation | 改动→owner/消费者→测试、失效扩大覆盖、requested/candidate/last-good | [契约与影响](../skills/upgrading-sglang-hcu-upstream-first/references/contracts-and-impact.md)；无新 selector 或定时任务 |
| compatibility / build-main | 追溯固定源码与兼容准入分离；能力/ABI；隔离安装和父子进程来源 | [兼容与产物](../skills/upgrading-sglang-hcu-upstream-first/references/compatibility-and-artifacts.md)；不继承 vLLM 数值版本/私有 wheel 门槛 |
| plugin-migration / attention-kv | 最小 owner、别名/dispatch 与生命周期、完整数据流 | 目标实际 SGLang seam/Hook/registry、配置解析与 layer boundary；不引入 vLLM Platform/MRV2 |
| AITER / FLA / KV precision | 独立 provider、合法中间值、实际物理格式、变输入 Graph、完整转换成本 | [原生与状态](../skills/upgrading-sglang-hcu-upstream-first/references/native-contracts.md)；NULL 索引和 kernel/manager page 作为需要核验的 ABI |
| validation / ci-release / evaluation | 补丁表＋验收表、实际 wheel/image、真实服务与输出、同条件原始准确率和性能、资源回收 | [发版验收](../skills/upgrading-sglang-hcu-upstream-first/references/release-validation.md)、[升级记录模板](../templates/upgrade-release-record.md)；状态沿用本仓 |
| sglang-qwen35-gap | 不能按算子名字计缺口，要比较实际 dispatch、融合后的完整成本 | 作为跨框架审查方法；2026-09 的候选 gap 和分数不变成本次 SGLang 优化清单 |

同时给现有 migration/dev/triage/backtrack 四个 skill 增加按需路由，保留它们的任务边界。README、阅读指南、环境手册、架构导览、CI/验证和来源导航同步更新。

## 3. 未直接迁移及原因

| 内容 | 不直接迁移的原因 |
|---|---|
| MRV1/MRV2、Platform/CustomOp、vLLM KVCacheSpec/manager/调度私有 API | 框架所有权与生命周期不同，不能机械改前缀为 SGLang |
| VLLM_* 开关、固定厂商 flag、精确 page/shape/default provider | 只能在目标实际源码/安装 ABI 证明适用后采用；不改变当前 SGLang 配置 |
| 原 upgrade_preflight.py 与其测试 | 只扫描 vllm_hcu patch 布局与守卫，不认识 SGLang Hook/seam；现有工具边界已列出，完整 AST 契约自动比较仍是缺口 |
| 具体模型启动命令、历史 pass 数、分数、内存、速度、固定阈值 | 不能迁为新的默认支持/验收结论；按本次 checkpoint/设备/范围重新建立矩阵 |
| 私有镜像仓、runner/CI 变量、依赖 pin、旧 Python floor 与具体 deselect | 本仓尚无上线的硬件发布流水线，按实际项目配置建立，不能把配方说成已运行 |
| 持续 watcher、自动发布、外部 PR/消息 | 不属于本次 skill 迁移的执行范围 |

不复制第三方 runtime/native 实现或整套历史材料。现有 V1 source-inventory/source-manifest 保持快照；本次来源单独记录。用户已有未提交 V3 修改保留，不将其历史测试冒充本次执行。

## 4. 本次验证范围

结果见 [交付记录](delivery-validation.md) 中本次迁移小节。执行 L0、skill 元信息、链接/锚点与现有工具测试；原有 SVG/布局保留，只更新 skill 数量与任务链接。结构验证不能证明实际模型/Graph、wheel/image 发版或 agent 的全部行为，本次不作这些运行结论。
