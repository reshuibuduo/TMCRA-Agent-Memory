# 本地工具接入

仓库里的自动接入统一访问本机回环 API，并共用一套项目身份规则。任何本地适配器都不需要 TMCRA 账号或 TMCRA 签发的 API Key。

## 一轮对话的统一顺序

```text
当前用户问题
  -> 解析项目身份
  -> 重试本地待写任务
  -> 召回用户全局 + 当前项目记忆
  -> 把证据作为不可信数据注入
  -> 写入脱敏后的 USER 来源记录
  -> 宿主模型与工具循环
  -> 写入可见的 ASSISTANT 来源记录
```

项目 ID 依次根据 `.tmcra/project.json`、Git origin、Git 根目录或规范化工作目录生成。Session 只承担项目内部来源追踪。在同一个仓库中打开两个工具时，它们会共享项目记忆，同时保留各自的软件、Session、角色和 Agent 身份。

## 支持状态

| 宿主 | 自动链路 | 安装方式 | 当前证据 |
| --- | --- | --- | --- |
| Codex | 回答前召回；用户与 Agent 分开写入 | 一条命令 | Node 契约测试和真实本地 FastAPI 跨软件 E2E |
| DeepSeek Harness | 原生 `agent/pre-step` 召回、`session/event`（`turn/end`）写回 | 源码构建安装器；技术预览 | 真实 Harness AgentLoop 双会话测试、类型检查、构建和包审计 |
| Claude Code | 共用 Hook 实现与配置清单 | 手动注册 | 共用 Hook 契约和真实本地 FastAPI 跨软件 E2E |
| ZCode | 共用 Hook 实现与配置清单 | 手动注册 | 共用 Hook 契约测试；当前宿主打包流程仍待验收 |
| 其他宿主 | 本地 REST API | 由宿主决定 | 按统一顺序接入；未测试生命周期入口前不宣称自动接入 |

## Codex

先在一个终端启动 TMCRA，再开一个终端执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-codex-local.ps1
```

Shell：

```bash
bash scripts/install-codex-local.sh
```

脚本会创建 `~/.tmcra/local-integration.json`。这个文件只保存本地 bearer token 的文件路径，不保存 token 值。随后脚本把当前仓库注册为 Codex 本地 marketplace，开启 Hooks，并安装 `tmcra-local-memory`。

重启 Codex 后打开 `/hooks`，逐项检查四个命令并主动授予信任。安装器不能代替用户信任 Hook。

## DeepSeek Harness

需要本机已有 DSH CLI、Node.js `22.19.0` 或更新版本，以及 npm。

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install-deepseek-harness-local.ps1 `
  -PackageDirectory D:\tmcra-packages
```

脚本会配置共享的回环客户端、安装锁定依赖、运行类型检查和 Harness 生命周期测试、构建压缩包、通过 `npm pack` 审核包内容，最后调用 `dsh plugin --profile web add`。

尚未安装 DSH CLI 的机器也可以先完成全部构建验收：PowerShell 增加 `-SkipDshInstall`，Shell 设置 `TMCRA_SKIP_DSH_PLUGIN_INSTALL=1`。脚本会生成并核验压缩包，然后停止，不会伪装成宿主安装成功。

DeepSeek Harness 仍是预览依赖。即使当前插件已经通过 `0.1.0-rc.6` 生命周期测试，本仓库仍把它标成技术预览。

## Claude Code 与 ZCode

共用实现和清单位于：

- `integrations/local-agent-hooks/hooks/claude-hooks.json`
- `integrations/local-agent-hooks/hooks/zcode-hooks.json`
- `integrations/local-agent-hooks/hooks/run_hook.mjs`

它们与 Codex 使用同一份纯本地配置和数据契约。当前公开版尚未在干净环境验收这两个宿主的最新插件打包流程，因此不宣称一键安装。

## 故障行为

- 召回失败默认不会阻断宿主回答。
- 消息写入失败会保存到 `~/.tmcra/integrations/outbox`，由下一次生命周期事件重试。
- Hook 诊断日志不保存问题、回答、bearer token 或 API Key。
- 客户端会在读取本地 bearer token 前拒绝任何非回环地址。
- 待处理轮次、失败写入队列和诊断目录及文件均限制为当前操作系统用户专用。
