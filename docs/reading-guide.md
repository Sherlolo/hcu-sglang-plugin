# 阅读指南

| 你要完成的任务 | 阅读顺序 |
|---|---|
| 第一次接触仓库 | README → architecture → governance → bootstrap |
| 新增 plugin 能力 | skills/hcu-plugin-dev → plugins → validation |
| 从 DAS fork 抽取或升级 | skills/sglang-dcu-plugin-migration → migration → templates |
| 回溯成源码内嵌仓库 | skills/hcu-plugin-backtrack → backtracking |
| 精度/graph/hang/性能回归 | skills/hcu-plugin-triage → 对应 references → validation |
| 维护方法库 | extraction-map → source-inventory.json → ci → delivery-validation |

表内文档均在本目录，skill 在仓库根目录的 skills 下。细节只加载当前任务所需文件。

关键不变量：官方工作树不被日常插件开发改写；注册函数必须从 entry point 可达；AST 候选不等于 active；同名不等于同语义；eager 成功不等于 graph replay 成功；CPU 通过不等于 GPU 等价；历史测试数字不继承为当前结论。回溯是显式例外：它只修改新生成的输出树。

失效信号：上游 SHA/目标路径/签名变化、register 调用链变化、缺可选依赖却在 import 阶段崩溃、快关没有实际命中、缓存重复写入、collective 分支在 rank 间不一致。遇到这些信号，先重做对应契约验证。
