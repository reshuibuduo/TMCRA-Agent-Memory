# DeepSeek Harness 的 TMCRA 纯本地记忆插件

这个 DSH 插件把原生 `agent/pre-step` 与 `session/event` 生命周期接到同一台电脑上的 TMCRA。

先安装纯本地运行时，再生成共享接入配置：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-local.ps1
node .\integrations\local-agent-hooks\scripts\configure.mjs --runtime-config .\.tmcra\config\runtime\local-runtime.json
```

完成依赖、类型、生命周期测试和打包后，把插件加入指定 Harness Profile：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-deepseek-harness-local.ps1 `
  -PackageDirectory D:\tmcra-packages
```

Shell：

```bash
TMCRA_DSH_PACKAGE_DIRECTORY="$HOME/tmcra-packages" \
bash scripts/install-deepseek-harness-local.sh
```

脚本依次执行 `npm ci`、类型检查、Harness 生命周期测试、构建、`npm pack` 和 `dsh plugin --profile web add`。插件默认读取 `~/.tmcra/local-integration.json`，也可以通过 `configPath` 指向另一份纯本地接入配置。Harness 预览版可能无法正确处理带空格或非 ASCII 字符的压缩包路径，因此打包目录必须使用短的 ASCII 路径。

每一个被 Harness 接受的人类回合都会执行：

- 解析与其他本地 TMCRA 接入一致的项目身份；
- 在第一次模型调用前召回用户全局与当前项目证据；
- 以插件来源、明确不可信的数据消息注入召回内容；
- 以 `user` 主体写入用户问题；
- 用实际 Harness agent/subagent 身份写入可见回答；
- 本地写入失败时进入 outbox，在下一回合重试。

这个包不含 TMCRA 账户、套餐、设备授权、托管作用域或生产 API 地址。

生命周期和打包内容已针对 `@deepseek-ai/dsh-agent-loop` `0.1.0-rc.6` 验收。最后的 `dsh plugin add` 仍要求电脑已经安装 DSH CLI；找不到 CLI 时安装器会停止。
