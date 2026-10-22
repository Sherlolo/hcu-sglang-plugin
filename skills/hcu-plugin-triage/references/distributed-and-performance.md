# 分布式、crash 与性能定位

> **用途**：定位 rank 分叉、crash 和热路径退化，排除环境与冷编译混淆。按 [验证手册](../../../docs/validation.md) 保存环境、组合与结果。

提取自 DAS debug-distributed-hang、debug-cuda-crash、sglang-bisect-ci-regression、generate-profile 和 sglang-profile-analyse。

## 定位与证据

- 先查各 rank 是否 OOM/异常退出，收集 watchdog/Python 栈与最后错误，再判断 collective 卡住。记录 TP/PP/DP/EP、rank 时间线、最后 collective 类型/shape/顺序、seq_len/batch/cache 状态；二分首个 rank 分叉点，既查 shape 又查分支是否一致。
- 优先用已有 py-spy/watchdog/ROCm/HCU 工具，先读环境权限与版本。CUDA coredump/cuda-gdb 只在对应 CUDA 环境适用；不复制驱动专属命令作为 HCU 指令。不要停止不属于本任务的服务来制造空闲。
- crash 尽量在调用前保存必要输入 metadata 和最小可重放数据，关注 shape/dtype/stride/device/NaN；某方法依赖模块或分布式状态时，输入 tensor dump 不等于可独立重放。
- 性能退化先对齐硬件/driver/DTK/轮子/各仓 SHA，分开冷编译、预热和稳定测量；再用环境支持的 profiler 看 CPU dispatch、kernel、通信和同步占比。profiling 结果与正常 benchmark 分开报告，避免探针开销污染结论。
- 回归二分先找相同测试配置下最后 pass / 首次 fail，并确认不是换了卡型/runner/依赖。对固定失败签名在独立 checkout/worktree 中缩小提交范围；不在用户脏树 checkout/reset。无法复现或缺 GPU 时记 pending，不能强行定责某提交。

组合回归记录两仓 SHA、实际导入路径、加载时序和配置来源；在同一硬件/输入下隔离新变化，而不是一次更换多个依赖后推断某仓有问题。
