# SGLang HCU 升级与发版记录

> **用途**：复制到本次任务产物中填写；空白、示例和旧报告不代表通过。配合 [升级 skill](../skills/upgrading-sglang-hcu-upstream-first/SKILL.md)，细粒度行为记录仍用 [验证模板](validation-record.md)。本模板不改变现有 ledger JSON schema。

## 1. 本轮输入与范围

| 项 | 本次值与证据 |
|---|---|
| 日期、需求与必须保留的功能/验收标准来源 | |
| 实际已验证基线与证据（不只写起点 HEAD） | |
| official/fork/plugin 起点 ref、SHA、dirty 与安装来源 | |
| fork 官方基点与祖先核验/历史缺口 | |
| 用户目标 ref、解析 SHA、目标元信息 | |
| adapter（使用时）版本、部署组合、有效 fallback | |
| gfx/卡数/拓扑、Python/Torch/DTK/provider ABI | |
| 模型/checkpoint revision、tokenizer/template、数据/样本 | |
| 交付形态、候选 wheel/hash 或 image/digest、构建配置 | |

## 2. 补丁迁移表

逐条覆盖旧注册/Hook/具名 seam、registry、alias、复制实现、native、build/config。稳定行为 ID 与 [迁移契约](../skills/sglang-dcu-plugin-migration/references/migration-contract.md) 台账条目关联，清单重排保存 ID 映射。处置与状态沿用该契约。

| 行为 ID / 来源位置与前提 | 官方目标 owner / 消费者与变化 | 接管证据 / HCU 剩余差异 | 处置 / 唯一实现 owner / 候选位置 | seam、加载顺序与 fallback | 临时 patch 退出条件 | 验收行 ID / 证据 / 状态 |
|---|---|---|---|---|---|---|

单列枚举总数、处置/状态分布、AST 候选、registered、applied、实际执行与缺口；不把数量相同当完整。

## 3. 本次验收矩阵

状态使用 passed/failed/pending/not-applicable；pending 注明 blocked 或 not-run 与解除条件。阈值在比较前确认。必测 skip/xfail 不能算 passed，不适用须有范围依据。

| 行 ID / 需求与补丁 ID / 必测或可选 | 模型与环境/拓扑 | provider/KV/Graph/prefix/投机及解析后配置 | 基线与候选命令/负载/数据 | 断言与阈值来源 | 实际路径与原始证据 | 状态 / 缺口 |
|---|---|---|---|---|---|---|

native provider 与 fallback、源码与实际交付产物、单仓与 adapter 组合分别留行。记录 collected/executed/skip/xfail 和必测行实际结果；引用旧日志要写明复用条件。

## 4. 测试影响与构建复用

| map revision / 来源 | repo:符号/路径及消费者、布局/native/build 依赖 | 真实 test ID / command / 对应矩阵行 | always-run 或选中理由 | 排除及失效条件 / 实际执行集合 |
|---|---|---|---|---|

记录未映射项与扩大测试范围决定；CI/选择器变化用独立清单核对。缓存/预编译扩展复用写明源码、ABI 与环境不变依据，否则重建。

## 5. 服务、准确率、性能与产物

记录父子进程模块/扩展来源、原始命令和实际解析值、provider/kernel 命中、ready 与请求输出、cache/投机计数增量。准确率保留原始分数/样本数、finish reason/token 数与失败样本；性能保留冷编译/预热和热测量分布。wheel/native resources/许可/hash 或 embedded 清单核验附原始报告。

服务记录启动前资源、端口监听者、自有会话/PID 与创建时间、超时和清理结果；观察中断后核验原进程，再决定是否重启。长期证据位置：填写稳定路径，不能仅依赖 ignored 目录。

## 6. candidate 与 last-good

| 记录类型 | 源码与产物摘要 / 环境 / 支持范围 | 必测结果与证据 | 结论及剩余项 |
|---|---|---|---|
| requested target | | | |
| candidate | | | |
| 原 last-good | | | |
| 新 last-good（本范围全部必测通过后填写） | | | |

交付说明删除/复用的优化、剩余复制实现/临时 patch、未测和恢复方法。只完成代码适配时明确阶段；本记录不自动授权发布。
