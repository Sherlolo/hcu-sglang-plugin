# 接口契约、测试影响与验证基线

> **用途**：减少无关重复测试，同时防止静默失效。输入是本轮固定的基线、目标、候选及其完整差异；不依赖 vLLM 的扫描格式或测试 registry。

## 契约先于守卫更新

先查目标真实定义、调用者、导入别名、缓存时序与注册阶段。比较参数顺序/类别/默认值、注解、装饰器、async、函数体及所调用 helper；类拆分、条件定义、继承和动态 alias 未解析时记缺口。AST 相同不证明外部依赖或数据所有权一致；AST 变化也不直接等于 break。不能只改签名白名单/源码摘要来消除报错。

SGLang 特别检查：ServerArgs 的实际 record 类型及解析/冻结阶段、typed config bag、resolution hooks、模型 layer boundary/residual、attention/KV 元数据、投机状态与并行组。目标可能没有 V3 的 API，按实际源码核实；实践线索见 [V3 手册](../../../docs/v3-runtime-validation.md)。

当前工具的边界：

- [审计与清单工具](../../../docs/migration.md) 产生静态候选，不能替代 runtime。
- [validate_hook_signatures.py](../../../tools/validate_hook_signatures.py) 会加载外部插件，仅覆盖 models REPLACE 子集，默认值数量相同不代表值相同；空结果不是兼容证明。
- [validate_plugin_runtime.py](../../../tools/validate_plugin_runtime.py) 使用 `--plugin-package hcu_sglang` 按当前 HCU 包/框架接口检查（旧包需另行显式选择），覆盖非空加载、预期集合、apply、软错误、来源及具名 seam；不支持的新布局需适配工具并验证，不能靠删预期目标绕过。它不证明每条模型路径实际执行或 native ABI 正确。
- vLLM 原 `upgrade_preflight.py` 扫描 `vllm_hcu/patch/TARGET_MODULE`，不适配 SGLang，本次不复制。完整 AST 默认值/函数体对照当前仍需人工或专门工具，不能写成已自动覆盖。

## 测试影响映射

复用项目已有格式，在 [升级记录](../../../templates/upgrade-release-record.md) 关联测试：映射版本/来源、repo:路径/符号、调用者与依赖、测试 ID/真实命令、环境与功能、always-run 理由、排除/失效条件。

收集 official、plugin、adapter、provider、build/config 与测试自身的差异；沿消费者、别名、共享布局读写端、状态与构建依赖展开。选择受影响用例与 always-run 的并集。启动/注册、来源检查和代表性数据流是基础候选，具体列表从当前实现推导，不照搬历史测试 ID。

以下变化扩大到相关子系统或本次完整矩阵：owner/动态 dispatch 变化、native ABI/布局或核心生命周期变化、映射过期/未映射、模型范围变化、依赖图无法复核。选择器或 CI 自身变化时用独立审核的完整清单检查漏选、空集合、skip/xfail；不能相信选择器自己的“零影响”。记录选中、排除、实际收集/执行集合和理由。

## 快速反馈与完整验收

按静态/portable→真实导入与 native→代表性 Graph/cache/投机/分布式交互→目标完整矩阵取得反馈。CPU mocks 不替代真实 loader。必测用例 skip/xfail 保留原因并记 pending/failed，不能汇总成发版通过；有依据的不适用子项在矩阵中单独登记。

复用结果须证明相关源码、依赖、模型/数据、配置、测试和环境未变并引用原日志；否则重跑受影响部分。构建缓存键包含实际相关源码/生成器/头文件、编译参数、Python ABI、Torch/DTK/provider/编译器与 gfx；影响不明则失效，不把命中缓存当验证完成。

## requested target、candidate 与 last-good

三者分别记录。last-good 包含官方/plugin/adapter 来源、wheel/image 内容摘要、ABI/provider/硬件拓扑、checkpoint、支持功能范围及全部必测证据。只有相应范围完整 passed 才更新；CPU 通过最多更新 CPU 阶段记录。失败候选不覆盖旧 last-good，也不要求同一个插件继续支持未声明的全部历史版本。

稳定可复现的跨版本故障可在独立 checkout 二分；good/bad 须同条件成立。中间提交无法构建或依赖不满足记不可测，不能机械判为推理 bad。
