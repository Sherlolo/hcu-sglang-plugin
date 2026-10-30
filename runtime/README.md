# runtime/ — sglang-hcu-plugin 插件运行时

方法库仓内承载的插件运行时代码（初版；按约定运行时最终会迁移到独立工程仓）。

- `sglang-hcu-plugin/`：可 `pip install` 的插件包。entry point 组
  `sglang.srt.plugins`，入口名 `hcu_plugin`。安装后由 sglang 在
  ServerArgs 构造之前自动加载，无需改任何启动参数。

## 安装验证（容器内）

```bash
pip install --no-deps /home/hcu-sglang-plugin/runtime/sglang-hcu-plugin
python3 -c "from importlib.metadata import entry_points; \
  eps=entry_points(group='sglang.srt.plugins'); print([(e.name,e.value) for e in eps])"
```

启动 sglang 服务后，日志中应出现：

```
[sglang-hcu-plugin 0.1.0] loaded pid=...
[sglang-hcu-plugin 0.1.0] ServerArgs ready: model=... tp=... attention_backend=...
[sglang-hcu-plugin 0.1.0] adapter shadow sgl_kernel -> /home/adapter-das/sgl_kernel/__init__.py
```

## 开关

| 环境变量 | 默认 | 作用 |
|---|---|---|
| `HCU_PLUGIN_BANNER` | `1` | 置 `0` 关闭两类横幅输出 |
| `HCU_PLUGIN_TRACE_ADD_PREFIX` | `0` | 置 `1` 打印 `add_prefix` 调用踪迹（hook 链路最小行为证明） |

卸载：`pip uninstall sglang-hcu-plugin`。
