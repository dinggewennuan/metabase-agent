# Metabase Agent Python

这是一个本地可运行的 Python Metabase Agent。它可以把自然语言问题解析成语义查询流程，支持 Metabase Metric，也支持 BigQuery/MongoDB 表源上的基础聚合查询，并输出回答、执行步骤、Query Plan、Metabase Program 和 Query Result。

## 1. 安装依赖

进入项目目录：

```bash
cd /Users/ks/akool/metabase-agent-python
```

安装依赖：

```bash
uv sync --extra dev
```

## 2. 最方便的页面测试方式

启动本地页面：

```bash
uv run metabase-agent web
```

浏览器打开：

```text
http://127.0.0.1:8765
```

页面默认状态读取 `.env` 里的 `AGENT_DRY_RUN`。dry-run 本地样例不需要 OpenAI Key 或 Metabase Key；真实模式会读取 `.env`，页面不会展示密钥。

可以输入：

```text
business_data 下fs_times 最近7天的每天的数据count
```

点击 `Ask` 后，页面会显示：

- Agent 回答
- 对话记录和 `session_id`
- 执行步骤 Trace
- Query Plan
- Metabase Program
- Query Result

## 3. CLI 命令行测试

dry-run 测试：

```bash
uv run metabase-agent ask "上周收入趋势怎么样？" --dry-run
```

表聚合 dry-run 测试：

```bash
uv run metabase-agent ask "business_data 下orders 最近7天的每天的数据count" --dry-run
```

查看帮助：

```bash
uv run metabase-agent --help
uv run metabase-agent ask --help
```

## 4. API 测试

先启动服务：

```bash
uv run metabase-agent web
```

然后另开一个终端请求 API：

```bash
curl -X POST http://127.0.0.1:8765/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"business_data 下fs_times 最近7天的每天的数据count","dry_run":true,"session_id":"api-demo"}'
```

预期会返回包含这些字段的 JSON：

```json
{
  "answer": "...Total Revenue...",
  "query_plan": {...},
  "program": {...},
  "query_result": {...},
  "trace": [...],
  "session_id": "api-demo"
}
```

连续对话记忆测试：

```bash
curl -X POST http://127.0.0.1:8765/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"上周收入趋势怎么样？","dry_run":true,"session_id":"memory-demo"}'

curl -X POST http://127.0.0.1:8765/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"我上次问的内容是什么？","dry_run":true,"session_id":"memory-demo"}'
```

## 5. 自动化测试

运行单元测试：

```bash
uv run pytest
```

编译检查：

```bash
uv run python -m compileall src tests
```

当前已覆盖：

- 页面首页加载
- `/api/ask` dry-run 请求
- LangGraph dry-run workflow
- 中文语义解析
- 会话记忆
- BigQuery schema 表列表
- 表级 count/sum/avg/min/max 聚合
- 最近 N 天时间过滤和按天分组
- Metric 选择
- Metric 路径回答合成（数值 / 趋势 / 环比对比 / 明细）
- Query Program 构造
- Policy 校验
- 只读 SQL 策略（拦截写操作与 BigQuery scripting 关键词）
- 工具循环 Agent（AGENT_MODE=tools）：工具分发、迭代上限、SQL 审批挂起与恢复
- 双 wire 协议工具调用 transport（chat_completions / responses）
- 真流式 SSE 逐节点进度事件
- CLI dry-run 管道与 tools 模式分支

## 6. 使用真实 Metabase / OpenAI

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env`：

```env
OPENAI_API_KEY=你的 OpenAI Key
OPENAI_BASE_URL=https://ai.akool.icu
OPENAI_MODEL=gpt-5
OPENAI_WIRE_API=responses
METABASE_BASE_URL=https://akool.metabaseapp.com
METABASE_API_KEY=你的 Metabase API Key
AGENT_DRY_RUN=false
AGENT_MODE=pipeline
```

`AGENT_MODE` 控制执行方式：

- `pipeline`（默认）：固定规则管道，意图解析后路由到对应节点，行为确定、可离线 dry-run。
- `tools`：LLM 工具循环 Agent，模型自行调用 `list_databases`/`list_tables`/`list_fields`/`run_aggregation`/`run_sql` 等只读工具并综合作答；需要 `OPENAI_API_KEY`，缺 key 时自动回退到 `pipeline`。所有工具仍复用相同的只读 SQL 校验、查询策略和人工审批闸口。

如果你使用的是兼容 OpenAI 协议的模型服务，把 `OPENAI_BASE_URL` 改成你的服务地址，例如：

```env
OPENAI_API_KEY=你的兼容服务 Key
OPENAI_BASE_URL=https://your-openai-compatible-host/v1
OPENAI_MODEL=你的模型名
OPENAI_WIRE_API=responses
```

然后运行：

```bash
uv run metabase-agent web
```

或者：

```bash
uv run metabase-agent ask "上周收入趋势怎么样？"
```

真实查询示例：

```bash
uv run metabase-agent ask "查询BigQuery-GA 下business_data都有什么表"
uv run metabase-agent ask "查询BigQuery-GA 下business_data 的 usr_users 表有多少条数据？"
uv run metabase-agent ask "business_data 下fs_times 最近7天的每天的数据count"
```

注意：本地真实查询需要你的 Metabase API Key 具备对应 Metric/Table 的只读权限。程序会优先尝试 Metabase Agent v2 query；如果当前实例没有 v2 endpoint，会 fallback 到 v1 construct-query + execute。

## 7. 部署运行

开发/内网试用：

```bash
uv sync --extra dev
uv run metabase-agent web --host 127.0.0.1 --port 8765
```

服务化运行可以直接使用 uvicorn：

```bash
uv run uvicorn metabase_agent.api.app:app --host 0.0.0.0 --port 8765
```

建议：

- `.env` 只放在服务器本地，不提交到 git。
- Metabase API Key 使用只读权限。
- 内部网络入口加认证或网关限制（生产建议设置 `AGENT_API_TOKEN`）。
- 如果真实查询偶发 429/502/503/504，客户端会对 GET/HEAD 做短暂重试；POST 不会自动重试，避免重复执行。
- **状态后端与多 worker**：默认 `AGENT_STORE=memory`（进程内 + 本地 JSON），只适合单 worker；用 `--workers N`（N>1）时审批/会话状态会在进程间分裂。要跑多 worker，设 `AGENT_STORE=sqlite` 并把 `AGENT_STATE_PATH` 指向一个共享的 `.db` 文件（如 `/var/lib/metabase-agent/state.db`），各 worker 通过 SQLite（WAL）共享会话记忆、挂起审批与表上下文；可选 `AGENT_SESSION_TTL_SECONDS` 自动清理过期会话。
- **Docker**：`docker build -t metabase-agent . && docker run --env-file .env -p 8765:8765 metabase-agent`。

## 8. 常见问题

如果端口被占用，可以换端口：

```bash
uv run metabase-agent web --port 8766
```

如果想停止服务，在启动服务的终端按：

```text
Ctrl+C
```

如果页面没有返回结果，先用 dry-run 确认本地流程：

```bash
uv run metabase-agent ask "上周收入趋势怎么样？" --dry-run
```

如果真实模式失败，优先检查：

- `.env` 是否存在
- `OPENAI_BASE_URL` 是否是兼容 OpenAI 协议的 `/v1` 地址
- `METABASE_BASE_URL` 是否正确
- `METABASE_API_KEY` 是否有权限
- `AGENT_DRY_RUN=false` 是否已设置

如果 `business_data 下fs_times 最近7天的每天的数据count` 不能自动选择正确时间字段，可以在问题中补充字段名，例如：

```text
business_data 下fs_times 最近7天的每天的数据count，时间字段 last_sub_upgrade_time
```
