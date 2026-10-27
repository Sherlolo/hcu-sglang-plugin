# 测试契约：真实边界、反例与有效 oracle

> **用途**：按实际改动选择能发现错误的测试，避免接线或孤立算子全绿掩盖模型调用链失败。开发闭环见 [历史经验](../../../docs/development-methodology.md)，分层结论见 [验证手册](../../../docs/validation.md)。

## 1. 从改动选择回归入口

| 改动面 | 回归应保留的真实结构 | 关键断言 |
|---|---|---|
| API/Hook/seam | 顶层注册链、公共分发、实际 reader/readout、绑定 receiver | 预期非空 targets 全量 apply；新可选/keyword-only 参数与返回值完整传递 |
| KV pool/布局 | 构造器→descriptor→写/移/读→backend→native consumer | pointer、stride、byte span、不同 K/V head_dim、页边界与备用 buffer |
| residual/norm | 合法 initial/written/pending、entry wrapper、模型实际 NormReadout/NormQuantReadout | residual 更新一次、pending 被消费、共享 plan 不被修改、额外张量不丢失 |
| collective | 生产者局部贡献→真实 FFN exit/下一消费者 | TP1 和本次 TP>1、空/非空；sum 的 owner、group、次数及最终值 |
| cache/消费者 dtype | canonical buffer 与消费者 cast/视图分开 | dtype/device、源 buffer 替换、memo 复用/失效、其他消费者不受破坏 |
| Graph | 同一真实 kernel/状态链的 eager、capture、变输入 replay | 改输入、位置/slot、cache 后读取新数据；K 的每个 head 只处理一次 |
| 配置/能力选择 | 实际启停和所声明的 adapter/plugin 组合 | 分支命中、完整参数、回退保留当前有效基线 |
| 安装/交付 | 指定源码、隔离 wheel/镜像、embedded | 包数据与 ABI、真实模块路径、外部入口隔离、同一行为合同 |

并非每次改动都跑表中所有行。按影响面选测并记录理由；只对当前需求明确涉及的模型、部署组合和产物给结论。

## 2. Mock 的边界

mock 昂贵硬件或通信时，保留被验证的框架生命周期。只直接调用内部 Hook 会漏掉公共 reader 在 Hook 前抛错；`__new__` 手填 buffer 会漏掉真实构造中的 descriptor 错误。pending residual 用生产者的 `record` 创建，不把 written 状态与独立 hidden 硬凑在一起。

使用实际 readout。NormReadout 与 NormQuantReadout 即使 quant_format 为空也可能调用不同 norm 签名；第三个位置 None、显式/默认 bool 与 keyword-only 都需按调用点覆盖。静态 signature bind 只是候选检查；加 `**kwargs` 后丢弃数据不能算兼容。

归约测试不能停在 TP1 的 router/expert mock。当前社区边界若负责 sum，专家方法应返回局部贡献；用真实 ExitPolicy/FfnExit 验证完整结果和 collective 次数。V3 中预期 8、旧结果 32 的反例比“all_reduce 被调用过”更能验证合同。

## 3. Oracle 与修复前反例

先让同一回归在修复前失败，再保留最小变更和修复后的结果。方法机械搬移与语义修改分开；AST/源码或 `sync_deepseek_methods --check` 只证明所定义的文本映射，不代表行为等价。

探针记录实际对象/文件、实现选择、dtype、shape 与模式。社区算子通过不代表实际 vendor 分支通过；其他 M/专家数失败也不能直接替当前模型定因。调用点和 provider 不一致时先修正探针。

对量化算法区分理想 FP32 oracle、同量化输入/权重 oracle 与安装的 native 基线。运输层应保持原生结果与物理字节；原生近似误差不能靠放宽运输阈值掩盖，也不能把与 native 一致当成完整模型准确率。

验收器也需要真实失败样本。保存原始 JSON，验证数字/关键词偶然出现、控制字符、未闭合思考段或 length 截断都会失败；正常已知答案仍应通过。确定性短响应使用 [validate_generation_response.py](../../../tools/validate_generation_response.py)，开放题另选评分标准。

## 4. 状态与 Graph

输入 token、位置、cache 初态与采样对齐后，比较首次“输入相同、输出不同”的边界。状态等价与 batch-invariance 分开；基础路径、真实 prefix hit、decode hit、跨页/滑窗/淘汰与重算分别验证。用例名称不是命中证据。

Graph replay 不执行 Python forward hooks；capture 中不能以 `.item()` 等主机同步保存证据。探针应证明命中且保留原问题；在 capture 外读动态状态，用变输入/变状态 replay 对照。仅 eager 正确或关闭 Graph 后遇到其他错误，都不能给原 Graph 行通过。

## 5. 服务、基准与交接

服务需分别给初始化、owned ready、真实生成及清理证据。相同 prompt 的短响应、health=200、20 个 Graph 捕获完成或单 kernel 对拍，不替代其他层。资源、协议、构建环境与形态闭环统一见 [历史经验](../../../docs/development-methodology.md)。

bench 固定硬件/driver、依赖、SHA、输入输出分布、并发与正确性，分离冷编译/预热/稳定测量，报告相对基线。未涉及性能要求的修复不凭日志中的绝对 tok/s 宣称加速。

## 抽取方法与模块别名

将类方法抽为普通函数时，执行构造路径检查拥有类、receiver 与 `super`/`__class__`；保留装饰器行为，class/static/property 描述符另声明绑定。常量重写须区分标识符、属性名、字符串和注释，重复执行仍应保持结果；本地同名变量应先审阅作用域。对源类拆分拒绝猜测。

验证 post-apply 时预先导入实际消费者，检查旧别名是否刷新；同时断言缺失属性和无关模块不被写入。只检查 patch 后重新 import 会漏掉旧引用。本轮修复前后执行证据见 [实施记录](../../../docs/v3-20261026-development.md#3-故障复现与工具修正)。


### 设备专用优化的输入准入

若宿主平台探测用于选择设备 kernel，还需检查实际输入设备。参照 [CP remap 案例](../../../docs/development-methodology.md#11-设备能力与张量准入)，用 HCU 主机上的 CPU 输入验证参考路径不导入设备依赖，再用真实 HCU 输入验证优化分支及相关 Graph 数值；不能只 mock 平台为非 HCU 绕过该边界。


### 缓存池类与通信器集成

当差异覆盖构造、布局、状态和多个协作方法时，应以一致的对象契约接入，保留原拥有类的闭包，核对派生类继承和预导入消费者。可复用的官方类/utility 先比对实现及实际依赖，再验证完整构造、缓冲和下一消费者；不能只替换缺失常量来使导入通过。参见 [CP 父池与预算案例](../../../docs/v3-20261026-development.md#72-cp-父池容量与握手接线)。

通信可用性按实际生命周期判断。当前 PyNCCL 的默认 disabled 与作用域内启用分别检查，并断言真实 collective 数值；native all-reduce、PyNCCL 和模型 PD 传输的证据分别记录。新增 GPU 单测沿用显式 `HCU_PLUGIN_RUN_GPU_TESTS=1` 准入，默认 CPU 检查不触发这些设备数值用例。


### 描述符协议与复制计划

涉及 PD/分片缓存时，从公开状态生产入口检查编解码、peer 发布与最终 byte effects。对照应覆盖源/目标不同排序、同层不同 family、未拥有项、组件类型的重复次序、元数据数量和 ABI 错配；错误注册在任何写入前拒绝，重新注册撤销旧 verdict。目标组件已映射时，对应指针、stride、索引和描述符都沿同一个目标索引。

复制计划的参考结果从身份与原始缓冲独立计算。检查每个批次沿用同一注册快照，验证异常时仍等待已经启动的写入完成。可用有界 CPU engine 做局部反例，再以真实 HTTP/ZMQ、可用原生 transport 和实际缓冲复验；按各 lane 的实际范围记录，见 [描述符案例](../../../docs/development-methodology.md#13-描述符身份与位置式校验冲突)。


### Attention 类图与原生构建边界

若迁移涉及基类新增状态，检查派生类的真实基类与 class closure；模块别名更新不能重建已有继承关系。CP 空查询回归保留 prepare/prefetch/wait 到早退的顺序，后端每条独立索引读取路径分别检查且不改写共享 metadata。相关实证见 [类图案例](../../../docs/development-methodology.md#14-attention-类图与空查询通信顺序)。

JIT 故障核对最终编译器 include 顺序及其进入缓存键的时点，不只检查 extra include 或资源哈希。native dtype 拒绝先区分数值格式与 packed-byte carrier，用实际 provider 对照 bytes/pointer/stride；数值转换与 view 不是同一种操作。记录 oracle 的边界，详见 [头文件案例](../../../docs/development-methodology.md#15-jit-实际头文件优先级) 和 [原生接口案例](../../../docs/development-methodology.md#16-原生算子-provider-与-fp8-carrier)。


### CP 缓存的读取与写入所有权

涉及紧凑 staging 时，分别核对读 getter 与持久写 getter；非 owner 可能仍须产生复制的 latent、索引或 pair state。调用点、Python carrier 与 native 无写入分支须一起验证，不能只把 tensor 改成 None。真实池测试保留共享 staging 的前后字节，设备测试覆盖有/无主 KV 的 eager 与变输入 Graph；物理 page-major payload/scale 按各自跨度核对。实证见 [写入所有权案例](../../../docs/development-methodology.md#18-缓存读取-staging-与持久写入目标)。


### 前向记录与后端依赖图

后端新增状态跨越构造器、metadata 类型、候选发布/消费与执行记录时，核对完整依赖图及类型身份；class REPLACE 后仍需检查原派生类的实际基类。对新增 ForwardBatch 字段检查声明、默认、实际构造、dataclass 复制及序列化，区分它与冻结 ServerArgs；见 [运行记录案例](../../../docs/development-methodology.md#19-前向批次-schema-与复制生命周期)。

保留真实 pool/allocator 的映射绑定及公开 metadata 初始化，数学探针不能用手填 core 替代这些边界。native logits 的 oracle 应明确编码、FP32/BF16 累积及候选可见范围；对齐适配放在实际 producer→consumer 边界，底层非法输入仍按合同拒绝。相关实证见 [候选评分案例](../../../docs/development-methodology.md#20-候选评分的行对齐与消费边界) 与 [容器分片案例](../../../docs/development-methodology.md#21-python-序列分片与-tensor-高级索引)。真实 CP>1 验证同时核对拓扑、空计数参与、独立通信器及销毁；分片节点方法通过不能代替模型引擎或流式生命周期验收。

环境变量迁移测试使用当前官方 `envs` 对象及实际 descriptor，通过环境变量驱动分支；不要用 `raising=False` 给旧模块补不存在的配置属性，否则会掩盖真实导入错误。见 [环境变量对象案例](../../../docs/development-methodology.md#22-环境变量对象与测试替身的接口一致性)。

CP 数值对照出现偏差时，先比较实际 Q/KV/索引与有效长度，再以同卡、相同原生批次形状隔离 provider 的形状依赖；独立 FP32 参考与原生完整批次对照保留各自结论。显式记录参考分块选项，不通过隐藏更换参考或放宽阈值覆盖原始失败。见 [批次形状案例](../../../docs/development-methodology.md#23-原生计算批次形状与数值参考)。

服务与产物来源验证同时固定工作目录和 Python 搜索路径；预检进程与实际启动方式不同，应检查派生进程是否会优先找到当前目录中的旧包。记录实际模块路径及启动目录，不能只由分发名或预检结果推断模型进程来源。参见 [产物与来源案例](../../../docs/development-methodology.md#8-交付产物与模块来源验证)。
