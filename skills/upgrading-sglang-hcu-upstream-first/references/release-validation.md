# 发版矩阵、服务与实际交付验证

> **用途**：在升级适配之后判断本次声明范围能否交付。沿用 [验证手册](../../../docs/validation.md) 的 passed/failed/pending/not-applicable；pending 再注明 blocked 或 not-run 及解除条件，不新增冲突的状态枚举。

## 必测矩阵与执行路径

修改前从需求、已有功能与目标差异建立 [矩阵](../../../templates/upgrade-release-record.md)，包含补丁 ID、必测/可选、checkpoint/revision/template、硬件/依赖、TP/PP/DP/EP/多节点等实际拓扑、backend/provider/KV/Graph/prefix/投机、命令、标准来源、原始证据。涉及 adapter 的真实部署组合必须单独验收。

所需 native provider 分别有行；dense 模型不能证明 MoE，fallback 不能通过 native 行。若要求保留 AITER 与 SlimQuant，分别触发它们的真实路径，先核验 SGLang CLI 与优先级，不能复制 vLLM 的选项。关闭 Graph、减少卡数、切换 dtype/backend 的诊断对照不替代原行。

真实服务需要配置解析、模型加载、ready、prefill/decode 与输出检查。保存原始启动命令和解析后配置；“没写 Graph 参数”不等于 eager，正式 seam 解析通过也不等于服务 ready。要求 prefix 时检查实际命中计数增量与正确输出、分叉/页边界；DP 需能归因各 rank。要求投机/MTP 时检查适用 draft 权重、drafted/accepted 增量和非零接受，不能从内嵌 MTP 层推断已启用。

## 模型协议、准确率与性能

基线/候选固定 checkpoint、tokenizer/template、reasoning/tool parser、采样/并发、样本 ID、数据与 token 预算。先探测请求协议是否生效，再解释数值回归；不同 checkpoint 不共用“关闭 thinking”参数。查询实际 `/v1/models` 服务名避免别名 404；保留 content/reasoning、finish reason、token 数、原始分数/样本数及失败样本。

先分清截断、格式与数值差异，不为候选结果改标准。对确定性短文本可按 [离线生成验收](../../../docs/validation.md#7-确定性生成响应的离线验收) 完整匹配最终回答并保存原响应哈希；不能只检查任意位置含正确数字。开放题仍使用其独立评分标准。代码输出归一化另报，保留缩进与原文；只有进行代码执行评测时才在已配置隔离 evaluator 中执行。单题差异不能自动忽略，重复测量按项目规程；没有准确率/性能标准则记录对照与待定项，不能自定可接受退化。

同硬件/拓扑、权重/精度/功能、输入输出分布、并发和客户端，分开冷编译、预热与稳定测量，报告吞吐、TTFT、TPOT/ITL、延迟分布、显存和错误率。按项目重复规程留原数据；规程缺失时多次独立测量并报告分布。单 kernel 更快或另一个框架的历史分数不能证明本次升级无退化。

## 长服务生命周期

记录启动前显存/利用率、端口与现有服务；0% 利用率但显存驻留不等于空闲。启动方式须适配执行器：可保持的前台/PTY 会话或有明确超时和清理责任的控制器，不能假定一次性 shell 的后台进程会存活。保存会话、PID/创建时间、所属进程组和独立日志。

本地 HTTP 探测按进程范围绕过 proxy、设连接/总超时并与服务日志相关联。确认监听者属于本任务后才请求；SSH/控制器观察中断后先核验原进程和健康状态，不能重复加载大模型，也不能把连接超时直接判为 OOM/模型失败。

共享资源的并发准入、用户授权停止冲突、节点权限拒绝与新 attempt 交接统一见 [资源排障](../../hcu-plugin-triage/references/distributed-and-performance.md)。清理仅操作自有会话/进程组及其 workers，验证退出、端口和显存恢复到实测启动前基线（可能非零）；无法确认就明确记录，不 broad pkill 或重启共享节点。网络中断时不得凭旧 PID 清理可能复用的进程。

## CI 与发版形态

本仓现状仍是 [runner 配方](../../../docs/ci.md)，不代表已经部署硬件或镜像 gate。实现 CI 时从当前项目配置确定触发方式、分支、镜像、设备、host/container Python 与测试清单；不复制 vLLM 私有变量、依赖 pin、no-skip 例外或 runner 标签。

真实 job 记录实际 checkout SHA、测试集合、环境和退出码，防止跑旧 ref/空选测。可变 image tag 当轮解析 digest；在交付镜像自己的解释器与依赖下验收，检查必要设备/挂载和 native 扩展。CPU gate 与硬件 gate 分开，硬件未运行不能由 portable 结果替代。CI 凭据/权限与外部发布沿用当前项目及用户授权。

源码、wheel/image、embedded 分开结论；只要求其中一种则注明范围。必测行有 failed/pending、旧补丁仍无处置或资源未确认回收时，报告相应剩余事项，不更新完整范围的 last-good。交付命令、产物来源、原始日志与恢复方法，长期证据不能只留 ignored artifacts。
