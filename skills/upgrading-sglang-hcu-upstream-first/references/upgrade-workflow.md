# 升级目标、上游接管与补丁维护边界

> **用途**：升级前决定哪些旧补丁仍需维护。用于用户选定目标的单次迁移；持续跟进只在用户要求时启用。三方审计命令统一维护在 [迁移手册](../../../docs/migration.md)。

## 固定输入

区分实际部署且有证据的基线、源码起点、fork 的官方基点、用户目标和候选。记录 official/fork/plugin/adapter（使用时）的 ref、SHA、dirty、真实安装来源、工具链与模型范围。移动分支解析为当次 SHA；要求最新版本时再核验官方发布信息。历史不完整须说明祖先未核验，不能把 shallow 的 merge-base 失败当作无共同祖先。

分析 official base→fork、base→target、plugin/adapter→target 三类变化。当前审计工具会读取 WORKTREE，不能声称它们只读指定 commit tree；按 [迁移手册](../../../docs/migration.md) 的 HEAD/ref 约束运行。需要其他 ref 时在独立 checkout 准备，不切换或清理用户脏工作树。

## 完整台账

从 entry point/顶层 register 开始，补齐条件 Hook、resolution seam、模型/量化/后端 registry、导入别名、模块复制、启动兼容层、native bindings/JIT overlay、打包/配置和 fork 回移。未触发或已移除的项也有处置行；不能只盘点报错文件。

沿用 [迁移契约](../../sglang-dcu-plugin-migration/references/migration-contract.md) 的九类处置和状态，另在证据中补：稳定行为 ID、旧用途/前提、目标 owner/消费者、上游接管证据、当前唯一实现 owner、关联测试 ID、临时 patch 退出条件。清单重排时保留旧新 ID 映射；不机械保持旧 Hook 数量。

## 上游接管判定

1. 写清旧补丁保护的正确性或性能、启用条件、shape/dtype/layout/provider、模型和拓扑；同一补丁的性能与兼容职责分别判定。
2. 在固定目标查实现、消费者、feature gate 与测试。名称相同、文件相同、PR 合并到更晚 main、patch-id 匹配都只是线索。
3. 在 HCU 实际栈验证可达性，排除旧替换、缓存 dispatch 或 `from ... import ...` 别名遮蔽。标准 ROCm/CUDA 有实现不代表当前 DTK/provider 可用。
4. 隔离候选绕开待移除部分，同时保留必要 adapter，按同条件基线验证功能、正确性和约定性能；无需先完整搬移可能多余的旧补丁来做 A/B。

| 结果 | 处置 |
|---|---|
| 官方完整覆盖且 HCU 实际执行满足标准 | 交回官方 owner，移除重复实现/注册与失效开关，保留行为回归 |
| 通用优化已覆盖，HCU ABI/layout/部分 shape 仍缺 | 保留最小适配，委托官方公共逻辑 |
| 官方没有所需路径或本次有回归 | 保留/改写有证据必要的实现，注明退出条件 |
| 只有静态线索或缺验收条件 | pending；不据此删除必要补丁或宣称接管 |

选择边界依次检查：官方已有行为→目标正式 seam/registry→窄 wrapper/Hook→记录 integration gap 的最小临时差异。不得假造目标没有的接口。已有行为适配由 adapter 负责、新能力由 plugin 负责，按 [治理](../../../docs/governance.md) 拆分；同一目标可有合法组合包装，但兼容转换集中在明确 owner，不重复修同一问题。

## 历史去重与退出

历史齐全且对象可解析时可用 `git patch-id --stable` 找 cherry-pick 重复候选、`git range-diff` 人工核对迁移前后的补丁序列；两者不证明语义等价，不作自动删除/通过条件。跨仓对象准备在隔离分析仓库进行。PR 优化研究使用准确 base/head，不把混入无关分支历史的 merge-base 全差异当作优化补丁。

退出条件写成可测能力，例如“固定目标官方 writer/reader 覆盖要求布局、Graph 与模型准确率且达到性能标准”，不能只写“等下个版本”。删除项同步处理别名、registry、CLI 兼容和共享 helper 消费者；新进程证明旧替换不再生效。

交付使用 [升级记录](../../../templates/upgrade-release-record.md)；本文件不授权更改目标、取消能力或自动发布。
