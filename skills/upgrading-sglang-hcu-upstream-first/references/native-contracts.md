# 原生算子、缓存状态与 Graph 契约

> **用途**：将 vLLM 原生/状态经验改为 SGLang 的检查问题，供影响分析命中相关路径时读取。本文件不指定某个 vendor 默认路线，也不证明当前硬件支持某个组合。

## Provider 与完整数据流

把 AITER、LightOp、SlimQuant、BoltOPs 等安装产物视为各自 ABI；同名包/同公开版本不保证符号与参数相同。按实际签名、配置查找、shape key 与调用探针核验。MoE、dense GEMM、attention、KV writer、Custom AllReduce 分开记录；启用其中之一不能推断其他都可用。显式要求的 native provider 与安全 fallback 独立验收。

能力不支持时在转换/打包/状态写入前回退到当前有效基线。provider 已选中后的 kernel/状态/非法内存错误不得被 broad except 吞掉；已知能力拒绝须有窄判定。不要安装上游同名包覆盖当前厂商栈来消除 import 错误。

FLA/GDN 对拍使用从合法前序计算产生的中间值，保留 gating、varlen、chunk_indices、initial/final state、输出 backing 和 dtype。随机 `w/u` 未必是合法 WY 中间值。显式审查 NULL state index、slot 索引起点与哨兵；来源 vLLM 的 0 与某些 SGLang provider 的 -1 是条件 ABI 案例，必须按本次调用者和安装签名确认，不能全局替换。预分配输出需要 copy 的成本属于候选路径。

## KV 与状态所有权

区分 checkpoint 权重格式、CLI KV 名称、解析配置和实际物理存储；FP8 字样不能证明 E4M3/E5M2 或 packed MLA 格式。先做 writer→gather/dequant→attention reader，再做 Graph/cache 与模型准确率。writer round-trip 通过不证明 decode 支持。

检查实际 layout、stride、storage offset、padding、scale、slot mapping、页边界与跨调用覆盖。重复 slot 的竞争要与布局错误分开；先唯一 slot 验证确定语义。kernel page、调度页和 conv/Mamba state page 分属不同 owner；不把 vLLM hybrid 页尺寸/allocator API 搬来改 SGLang，也不为满足一个 kernel 断言重做逐层大分配。

metadata 持久 buffer 的容量与有效长度分开：增长后传有效 view，覆盖 batch 缩小、prefill/decode 转换、不同请求结束时序；不把 capture 初始容量/零填充尾当有效工作项。conv/Mamba 状态检查权重布局、dtype、更新/restore 和请求隔离，首次分叉前先核对输入相同。

## Graph 与 collective

capture 后保持静态地址，写入新的输入与 rank-specific 值再 replay；从当轮值计算 reference，检查输出和 residual/state。覆盖动态 batch、边界形状和适用并发；成功 capture、有限输出、固定输入 replay 都不充分。collective 的 eager、注册 buffer、copy-in 分开验收；来源中的 AITER 构造 flag 仅作调查线索，不能未经 SGLang 实际调用点核实就设默认。

当目标使用 Dynamo/编译时，避免在可见 Python wrapper 用动态 token 维作白名单判断导致范围特化；仅在目标已有合适的注册算子边界才考虑 concrete-shape dispatch，同时验证 FakeTensor/动态维和 unsupported-shape 回退。不能凭 vLLM CustomOp 示例在 SGLang 假造注册方式。Graph 中所需索引/metadata 按目标约束在 capture 前准备，不能临时引入 host 同步。

## 性能判定

在真实模型 shape/dtype/头比例、合法中间值、当前实际 provider 上对照，而非只与裸官方 kernel 比。包含 layout 转换、repeat/gather/scatter、cast、临时分配和 copy 到官方 backing；eager 与 Graph replay 分开测，排名可能相反。按本次标准决定默认路由，不复制来源的 5% 阈值或固定 shape 结论。

模型与服务层门槛见 [发版验收](release-validation.md)，原有首次分叉方法见 [排障 Skill](../../hcu-plugin-triage/SKILL.md)。
