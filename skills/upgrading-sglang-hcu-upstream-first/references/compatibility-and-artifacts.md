# 兼容准入与交付产物来源

> **用途**：决定是否复用构建、如何接受兼容依赖及验证实际交付内容。适用于声明交付的 wheel、镜像或 embedded；不强制每个任务同时交付三种形态。

## 固定实验与运行准入分别表达

构建/实验固定源码 SHA、完整包版本、wheel SHA256、镜像 digest，供追溯复现。运行时准入从目标项目的版本策略、必要 API/native 能力与实际 ABI 推导；除项目已有明确要求，不新增单个完整 wheel 字符串/SHA allowlist。

若声明 release-line 兼容，使用项目版本解析规则比较完整数值组件与 epoch，不用 `startswith` 把 `0.x.10` 当作 `0.x.1`，明确 RC/dev/post/local 后缀策略。移动分支保存解析 SHA 与源码元信息，不凭分支名猜版本。兼容政策随本轮目标重新核验，不复制 vLLM 的版本号、DTK 最低值或厂商构建时间门槛。

需要特定厂商修复时，可表达为已核验的 ABI 分支＋最低修复构建＋能力测试；元信息不足则不能证明修复存在。版本是预检，还需 Python/Torch/DTK、gfx、扩展符号/签名、dtype/layout、Graph 和模型门槛。新 ABI 不能仅靠放宽版本守卫接受。

## 复用还是构建

1. 已有目标产物：检查内容来源、目标接口、native 能力与 ABI，并在隔离安装验证；无害后缀不同不等于必须重建，但来源不明也不能证明包含指定目标。
2. 缺兼容产物、需要改源码或缺 native 能力：按目标实际 build files/工具链最小构建。不能假定旧 DTK 路径、gfx flags、扩展名称或系统库 workaround 仍需要。
3. 复用预编译扩展须有 ABI 未变的依据，记录原扩展来源与构建条件。最终用实际 wheel/image 测试，editable 通过另记。

发布产物的版本与内容必须对应：SCM 后缀指向旧 HEAD 而源码脏的 wheel 是试验构建，不能称为 clean-commit 产物。用干净提交边界构建或提供明确内容摘要/dirty patch 追溯；提交按任务授权，不清理用户工作。

## 隔离来源与资源

在源码目录之外安装和测试交付 wheel；检查 `PYTHONPATH`、editable `.pth`/metadata 与用户 site，不只看 pip 的版本。父进程和实际 spawn/辅助子进程分别记录 `sglang`、plugin、相关 provider 的 `__file__` 与元信息；子进程不能重新 prepend 源码树。环境处理按当前 runner，测试命令与解析后配置一并保存。

[严格 loader](../../../tools/validate_plugin_runtime.py) 检查框架和插件来源；provider/native `.so` 的实际来源与加载行为另留证。entry-point allowlist 取自目标真实语义，不照搬 vLLM 的 `VLLM_PLUGINS` 经验为 SGLang 开关。

native JIT overlay/头文件/源码/许可属于交付内容。依据实际依赖清点 `.cu/.cpp/.cuh/.h/.hpp` 等资源，比较源码到 wheel/embedded 的清单和哈希；不能只看 Python import。保留来源与第三方许可，版本或资源数以本次枚举为准。

embedded 走 [回溯手册](../../../docs/backtracking.md)：干净输入、生成保护、SHA256；屏蔽同名外部插件及其他 entry points，在输出树独立证明加载与模型路径。生成成功不意味着等价于整个历史 fork。

产物记录至少含源码与 dirty/补丁摘要、Python ABI、Torch/DTK/provider/编译器/gfx、构建选项、依赖、wheel/hash 或 image/digest、安装位置、测试命令与日志。镜像可变 tag 在当轮解析一次 digest，未来任务可选择新产物；本 Skill 不绑定私有镜像仓或 runner 名称。
