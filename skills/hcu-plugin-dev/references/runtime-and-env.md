# 环境变量与运行时状态

> **用途**：在新增配置或修改状态生命周期时核对所有者、版本与可复用状态。完整参数边界见 [配置管理](../../../docs/configuration.md)。

## 1. 来源与版本适用性

从 DAS 的 env-var-conventions 与 sglang-runtime-context 提取，应用前核对目标 official SHA 是否具有相同 API；DAS 中存在的 RuntimeContext API 不代表旧 official 一定有。

## 2. 环境与覆盖

SGLang 自有 `SGLANG_*` 使用其版本支持的 Envs/typed descriptor API，读取 `.get()`，测试用 `.override()` 恢复状态；不要新增 legacy `SGL_*` 别名或在 plugin 热路径散落 raw getenv。外部驱动/launcher 的 NCCL、RANK、PATH 等变量仍由其所有者定义。plugin 自有开关在实现所属仓集中定义；当前本仓入口为 `hcu_sglang/config.py` / `hcu_sglang/environ.py`，登记名称、类型、默认、优先级、生效阶段及关闭路径，未实现接口不写成可用配置。

显式用户值优先；日志要说明最终值与来源。不要为注入一个配置字段改写 official 源码；如果当前 seam 不足，台账记录 integration gap。

## 3. 状态生命周期

配置解析、进程全局资源与 per-forward 状态分开；如果目标版本存在 publish/override/config bag，则遵循它的生命周期，不在 publish 后私改原始 ServerArgs。多进程/角色要分别验证配置传播。融合 KV 标记等 per-forward 数据不能在模块全局跨请求复用，Graph replay 的动态 tensor 状态不能只在 Python capture 阶段赋值。

组合使用时，adapter 的平台语义可能与真实 HIP 能力不同。配置读取不能重复注册另一仓的同名项；日志记录最终来源。per-forward 标记和 Graph 动态输入分别验证生命周期，fallback 保留当前有效适配基线。

## 4. 导入与构建环境也是契约

Python 启动兼容层只安装轻量注册/延迟 finder；不要在 usercustomize/sitecustomize 中提前导入 Torch、AITER 或启动 JIT。用实际解释器及 `ninja --version` 一类嵌套子进程核对可及时退出，避免父子进程递归进入同一编译锁。

按 pinned 源码选择构建工具链，在真实 builder 的环境中检查 `sys.executable`/realpath、PATH、cargo/rustc 等版本。某些 builder 会前置解释器目录，shell 的版本输出不足为证。先预编译真实 bindings，再加载大模型；仅设置任务级 wrapper/cache，保留全局工具、HOME、源码和 lock。

记录扩展 fingerprint、解释器/平台和 Torch ABI；已编译缓存只在兼容条件满足时复用，不自动宣称进入 wheel 或可跨 ABI 分发。失败保留原日志，不通过切换实现或删依赖绕过。V3 的锁递归与 Cargo 选择案例见 [历史经验](../../../docs/development-methodology.md#6-构建进程环境与工具链一致性)。


## 5. 缺少上游输入字段时的进程配置

来源DAS比固定官方版本多一个CLI字段时，先核对原始记录和发布后的namespace是否都具有该字段。不能只添加 argparse 选项后让记录构造丢弃它，也不能向冻结seed补写属性。若采用插件进程配置桥接，必须连接CLI消费、明确的覆盖/禁用规则、spawn继承、bootstrap及实际组构造，并声明程序化调用入口与每进程配置边界。

用真实CLI parser/record、冷子进程、公共bootstrap和真实组构造检查整条链，验证显式选项、默认、禁用优先级及失败传播；参数成功不代表传输后端命中。严格后端策略应保留来源异常或拒绝条件，不以回落到另一个communicator伪装完成。当前例子与限制见[后端参数](../../../docs/configuration.md#当前-custom-all-reduce-参数)。

固定版本的ServerArgs.__post_init__可能只构造未解析seed；CLI预检须检查实际launcher是否调用resolve_once，并显式执行同一阶段。resolved_view可读取seed，不能据此自称validated/effective配置；Node/平台/模型约束的拒绝结果需保存。旧快照phase说明错误时修正共享分析范围，保留原始产物，不修改原始模型响应或重新评分标签。
