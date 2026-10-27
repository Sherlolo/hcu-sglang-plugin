# 环境变量与运行时状态

> **用途**：在新增配置或修改状态生命周期时核对所有者、版本与可复用状态。完整参数边界见 [配置管理](../../../docs/configuration.md)。

## 1. 来源与版本适用性

从 DAS 的 env-var-conventions 与 sglang-runtime-context 提取，应用前核对目标 official SHA 是否具有相同 API；DAS 中存在的 RuntimeContext API 不代表旧 official 一定有。

## 2. 环境与覆盖

SGLang 自有 `SGLANG_*` 使用其版本支持的 Envs/typed descriptor API，读取 `.get()`，测试用 `.override()` 恢复状态；不要新增 legacy `SGL_*` 别名或在 plugin 热路径散落 raw getenv。外部驱动/launcher 的 NCCL、RANK、PATH 等变量仍由其所有者定义。plugin 自有开关在外部实现仓集中定义，登记名称/类型/默认/优先级/生效阶段/关闭路径，不在本库假造运行时配置。

显式用户值优先；日志要说明最终值与来源。不要为注入一个配置字段改写 official 源码；如果当前 seam 不足，台账记录 integration gap。

## 3. 状态生命周期

配置解析、进程全局资源与 per-forward 状态分开；如果目标版本存在 publish/override/config bag，则遵循它的生命周期，不在 publish 后私改原始 ServerArgs。多进程/角色要分别验证配置传播。融合 KV 标记等 per-forward 数据不能在模块全局跨请求复用，Graph replay 的动态 tensor 状态不能只在 Python capture 阶段赋值。

组合使用时，adapter 的平台语义可能与真实 HIP 能力不同。配置读取不能重复注册另一仓的同名项；日志记录最终来源。per-forward 标记和 Graph 动态输入分别验证生命周期，fallback 保留当前有效适配基线。

## 4. 导入与构建环境也是契约

Python 启动兼容层只安装轻量注册/延迟 finder；不要在 usercustomize/sitecustomize 中提前导入 Torch、AITER 或启动 JIT。用实际解释器及 `ninja --version` 一类嵌套子进程核对可及时退出，避免父子进程递归进入同一编译锁。

按 pinned 源码选择构建工具链，在真实 builder 的环境中检查 `sys.executable`/realpath、PATH、cargo/rustc 等版本。某些 builder 会前置解释器目录，shell 的版本输出不足为证。先预编译真实 bindings，再加载大模型；仅设置任务级 wrapper/cache，保留全局工具、HOME、源码和 lock。

记录扩展 fingerprint、解释器/平台和 Torch ABI；已编译缓存只在兼容条件满足时复用，不自动宣称进入 wheel 或可跨 ABI 分发。失败保留原日志，不通过切换实现或删依赖绕过。V3 的锁递归与 Cargo 选择案例见 [历史经验](../../../docs/development-methodology.md#6-构建进程环境与工具链一致性)。
