# Metabase Agent Python

本地可运行的 Python Metabase 分析 Agent。把自然语言问题解析成语义查询流程，支持 Metabase Metric，也支持 BigQuery/MongoDB 表源上的只读聚合与只读 SQL，输出**回答 + 执行 Trace + Query Plan + Metabase Program + Query Result**。

提供两种执行内核：

- **pipeline**（默认）：确定性的规则管道，可完全离线 dry-run，行为可复现。
- **tools**：LLM 工具循环 Agent，模型自行调用只读工具并综合作答，复用同一套安全闸口。

入口齐全：Web 页面、HTTP API（含 SSE 流式）、CLI。

---

## 目录

- [项目亮点](#项目亮点)
- [架构](#架构)
- [执行流程](#执行流程)
- [安装](#安装)
- [CLI 命令](#cli-命令)
- [Web 页面与 API](#web-页面与-api)
- [配置项](#配置项)
- [两种执行模式](#两种执行模式)
- [安全与策略](#安全与策略)
- [测试](#测试)
- [部署](#部署)
- [改进点 / Roadmap](#改进点--roadmap)
- [常见问题](#常见问题)

---

## 项目亮点

- **双内核**：规则管道（确定、可离线、可测）与 LLM 工具循环（灵活、自主探查）共存，靠 `AGENT_MODE` 一键切换，无 key 自动回退 pipeline。
- **安全优先**：只读 SQL 校验（字面量/注释/反引号掩码 + 单语句检测 + scripting 黑名单）、查询策略（limit ≤ 200、聚合函数白名单）、**SQL 执行前人工审批**闸口；三个入口（API/SSE/CLI）共用同一套闸口。审批是"取出即消费"的原子操作（并发批准不会重复执行），且**批准绑定被 review 的内容**——执行前按指纹比对，内容有偏移会重新进入审批。
- **全程可观测**：每一步都进 Trace；SSE 逐节点 / 逐工具推送进度，前端实时渲染处理过程。
- **会话能力**：按 `session_id` 的多轮记忆、待审批 SQL 与表上下文持久化，重启不丢。
- **多 worker 就绪**：状态后端可插拔，`AGENT_STORE=sqlite`（WAL）让多 worker 共享会话/审批/表上下文，支持 session TTL。
- **OpenAI 兼容网关友好**：三种 wire 协议（`chat_completions` 走 SDK、`chat_completions_httpx` 同端点裸 httpx、`responses` 走裸 httpx），适配自建/代理网关——包括 WAF 拦 SDK 请求指纹的网关。
- **工程化完备**：209 个单测、ruff、CI（含 Docker build 校验）、多阶段非 root Dockerfile、`ping` 连通性自检。

---

## 架构

分层模块（`src/metabase_agent/`）：

```
┌──────────────────────────────────────────────────────────────────────┐
│ 入口层                                                                 │
│   cli/app.py        Typer CLI：ask / version / ping / web              │
│   api/app.py        FastAPI：/、/api/config、/api/sessions、           │
│                     /api/ask、/api/ask/stream + 内嵌单页前端           │
│   api/store.py      会话状态后端（SqliteStore，多 worker）             │
│   api/static/       index.html（对话工作台前端）                       │
├──────────────────────────────────────────────────────────────────────┤
│ 编排层 agent/                                                          │
│   graph.py          LangGraph 管道（pipeline 内核）                    │
│   tool_loop.py      LLM 工具循环（tools 内核）+ 重复调用检测           │
│   tools.py          只读工具集 + dispatch（list_*/run_aggregation/run_sql）│
│   metadata_flow.py  库/表/字段元数据 + 表聚合处理                      │
│   sql_review.py     SQL 审批/预览/执行意图判定                         │
│   clarify.py        澄清建议话术                                       │
│   state.py/trace.py/metadata.py/dry_run.py                            │
├──────────────────────────────────────────────────────────────────────┤
│ 语义层 semantics/                                                      │
│   intent_parser.py  规则意图解析（正则 + 中文词表）                    │
│   llm_intent.py     LLM 意图分类                                       │
│   llm_client.py     统一 LLM 访问 + 双 wire 协议 tool-calling transport│
│   sql_explainer.py  SQL 解读（LLM + 结构化降级）                       │
│   business_glossary.py 业务术语归一                                    │
├──────────────────────────────────────────────────────────────────────┤
│ 查询/策略层                                                            │
│   query/query_planner.py / query_program_builder.py                  │
│   query/bigquery_report_sql.py + templates/monthly_usage_report.sql  │
│   metrics/metric_resolver.py   policy/query_policy.py                 │
├──────────────────────────────────────────────────────────────────────┤
│ 外部工具层                                                             │
│   tools/metabase/client.py   Metabase HTTP 客户端（重试/v2→v1 fallback）│
├──────────────────────────────────────────────────────────────────────┤
│ 配置层  config/settings.py（pydantic-settings，读 .env）              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 执行流程

### pipeline 内核（LangGraph）

```
question
  │
  ▼
parse ── 规则解析 + （真实模式）LLM 意图分类
  │
  ├─ sql_explanation ─→ LLM 解读 / 结构化降级摘要 ─→ END
  ├─ native_sql ─────→ 只读校验 → 人工审批 → POST /api/dataset ─→ END
  ├─ bigquery_sql ───→ 月度用量报表 SQL 模板 → 审批 → 执行 ─→ END
  ├─ database_metadata → 库/表/字段元数据；表聚合走
  │                      v2 /api/agent/v2/query（404 → v1 construct+execute）─→ END
  └─ search → inspect → plan → build_program → policy → execute → answer ─→ END
                                                （Metric 路径：数值/趋势/环比/明细）
```

### tools 内核（工具循环）

```
question + 历史
  │
  ▼
┌─────────────────────────────────────────────┐
│ iter_tool_loop（最多 N 轮，重复调用检测）   │
│   LLM ─→ 工具调用?                           │
│     ├─ list_databases / list_tables /        │
│     │   list_fields / run_aggregation        │ ← 自动执行（只读 + policy）
│     │      └─ 结果回填 messages，继续循环     │
│     ├─ run_sql ─→ 挂起，等待人工授权 ────────┼─→ requires_approval
│     │                （审批后恢复执行）        │
│     └─ 文本回答 ─────────────────────────────┼─→ completed
└─────────────────────────────────────────────┘
```

审批挂起时，完整消息历史写入会话状态（`PENDING_SQL_APPROVALS`，可落 SQLite）；下一次带 `decision=approve` 的请求恢复循环并执行已批 SQL。

---

## 安装

需要 [uv](https://github.com/astral-sh/uv)。

```bash
uv sync --extra dev
```

---

## CLI 命令

入口注册为 `metabase-agent`，统一用 `uv run metabase-agent <命令>`。共 4 个命令：

| 命令 | 作用 | 关键参数 |
|---|---|---|
| `ask` | 单次提问 | `question`（位置，必填）、`--dry-run` |
| `web` | 启动 Web/API 服务 | `--host`（默认 127.0.0.1）、`--port`（默认 8765） |
| `ping` | 连通性自检（Metabase + OpenAI 网关） | 无；任一子系统不通则退出码非 0 |
| `version` | 打印版本号 | 无 |

```bash
# 提问（dry-run 离线样例，无需任何 key）
uv run metabase-agent ask "上周收入趋势怎么样？" --dry-run
uv run metabase-agent ask "business_data 下orders 最近7天的每天的数据count" --dry-run

# 真实模式（需 .env 配好 key、AGENT_DRY_RUN=false）
uv run metabase-agent ask "查询BigQuery-GA 下business_data都有什么表"

# 连通性自检（部署后/换网关时先跑这个）
uv run metabase-agent ping

# 版本 / 帮助
uv run metabase-agent version
uv run metabase-agent --help
uv run metabase-agent ask --help
```

---

## Web 页面与 API

启动：

```bash
uv run metabase-agent web
# 浏览器打开 http://127.0.0.1:8765
```

页面展示：Agent 回答、对话记录与 `session_id`、执行 Trace、Query Plan、Metabase Program、Query Result，结果可复制 / 导出 CSV·Excel。

### API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 单页前端 |
| GET | `/api/config` | 返回 `default_dry_run` |
| GET | `/api/sessions/{session_id}` | 读取会话记忆 |
| POST | `/api/ask` | 同步提问 |
| POST | `/api/ask/stream` | SSE 流式（逐节点/逐工具进度 + final） |

```bash
curl -X POST http://127.0.0.1:8765/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"business_data 下fs_times 最近7天的每天的数据count","dry_run":true,"session_id":"api-demo"}'
```

返回 JSON 含 `answer / query_plan / program / query_result / trace / session_id / memory`。

连续对话（同一 `session_id` 即可记忆）：

```bash
curl -s -X POST http://127.0.0.1:8765/api/ask -H "Content-Type: application/json" \
  -d '{"question":"上周收入趋势怎么样？","dry_run":true,"session_id":"memory-demo"}'
curl -s -X POST http://127.0.0.1:8765/api/ask -H "Content-Type: application/json" \
  -d '{"question":"我上次问的内容是什么？","dry_run":true,"session_id":"memory-demo"}'
```

配置了 `AGENT_API_TOKEN` 时，`/api/ask`、`/api/ask/stream`、`/api/sessions` 需带 `X-Agent-Token` 头。

---

## 配置项

复制模板后编辑：

```bash
cp .env.example .env
```

| 变量 | 默认 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | 空 | LLM key；空则仅 pipeline + 不调 LLM |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI 兼容网关地址 |
| `OPENAI_MODEL` | `gpt-5` | 模型名 |
| `OPENAI_WIRE_API` | `chat_completions` | `chat_completions`（SDK）、`chat_completions_httpx`（同端点裸 httpx，网关 WAF 拦 SDK 指纹返回 403 时用）或 `responses`（裸 httpx） |
| `OPENAI_TIMEOUT` | `120` | LLM 请求超时（秒） |
| `SILICONFLOW_API_KEY` | 空 | SiliconFlow embedding key；`AGENT_EMBEDDING_PROVIDER=siliconflow` 时必填 |
| `SILICONFLOW_BASE_URL` | `https://api.siliconflow.cn/v1` | SiliconFlow API base URL |
| `METABASE_BASE_URL` | 空 | Metabase 实例地址 |
| `METABASE_API_KEY` | 空 | 只读权限的 Metabase API Key |
| `AGENT_DRY_RUN` | `true` | true=本地样例（离线、无需 key） |
| `AGENT_MODE` | `pipeline` | `pipeline` 或 `tools` |
| `AGENT_API_TOKEN` | 空 | 设置后 /api 需带 `X-Agent-Token` |
| `AGENT_REQUIRE_TOKEN` | `false` | true 且未配 token 时直接拒绝所有 /api 请求 |
| `AGENT_STORE` | `memory` | 会话/短期记忆后端：`memory`（单 worker）、`sqlite`（多 worker 共享）或 `mongodb`（存 MongoDB，集合自动创建） |
| `AGENT_STORE_MONGODB_URI` | 空 | `AGENT_STORE=mongodb` 的连接串；空则回退 `AGENT_MONGODB_URI` |
| `AGENT_STORE_MONGODB_DATABASE` | `metabase_agent_sessions` | MongoDB 会话库名 |
| `AGENT_STATE_PATH` | `.metabase_agent_state.json` | 审批/表上下文文件；sqlite 模式指向 `.db` |
| `AGENT_MEMORY_PATH` | `.metabase_agent_memory.json` | 会话记忆文件（memory 后端） |
| `AGENT_SESSION_TTL_SECONDS` | `0` | 会话过期秒数，0=不过期（仅 sqlite） |
| `AGENT_CHECKPOINT_BACKEND` | `none` | LangGraph checkpoint 后端；`mongodb` 启用 MongoDBSaver |
| `AGENT_CHECKPOINT_MONGODB_URI` | 空 | MongoDBSaver URI；空时复用 `AGENT_MONGODB_URI` |
| `AGENT_CHECKPOINT_MONGODB_DATABASE` | `metabase_agent_checkpoints` | MongoDBSaver database |
| `AGENT_CHECKPOINT_TTL_SECONDS` | `0` | checkpoint TTL 秒数，0=不过期 |
| `AGENT_TENANT_ID` | `default` | 长期记忆租户 ID |
| `AGENT_USER_ID` | 空 | 长期记忆用户 ID；API 为空时默认用 `session_id` |
| `AGENT_LONG_TERM_MEMORY_ENABLED` | `false` | 是否启用 MongoDB + pgvector 长期记忆 |
| `AGENT_MEMORY_LLM_EXTRACTOR` | `false` | 每轮问答后用 LLM 提取候选记忆（语义偏好 + procedural 规则提案，规则一律 `pending_review`）；需 `OPENAI_API_KEY` |
| `AGENT_MONGODB_URI` | 空 | 长期记忆 MongoDB URI |
| `AGENT_MONGODB_DATABASE` | `metabase_agent` | 长期记忆 MongoDB database |
| `AGENT_MEMORY_COLLECTION` | `agent_memories` | 长期记忆集合 |
| `AGENT_PGVECTOR_DSN` | 空 | pgvector PostgreSQL DSN |
| `AGENT_PGVECTOR_TABLE` | `memory_embeddings` | pgvector 表名 |
| `AGENT_PGVECTOR_AUTO_CREATE` | `true` | 启动时自动建库/扩展/表/索引（需建库权限）；false 则手动跑 `scripts/init_pgvector_memory.py` |
| `AGENT_EMBEDDING_PROVIDER` | `hash` | `hash`（离线/测试）、`openai` 或 `siliconflow` |
| `AGENT_EMBEDDING_MODEL` | `text-embedding-3-small` | embedding 模型；SiliconFlow 可用 `BAAI/bge-m3` |
| `AGENT_EMBEDDING_DIMENSIONS` | `1536` | embedding 维度；pgvector 表也必须使用相同维度 |
| `AGENT_SKILLS_ENABLED` | `true` | 是否启用 `skills/*/SKILL.md` 上下文注入 |
| `AGENT_SKILLS_PATH` | `skills` | skills 根目录 |
| `AGENT_SKILLS_MAX_CHARS` | `6000` | 每次注入 skills 上下文最大字符数 |
| `METABASE_BIGQUERY_DATABASE_ID` | `19` | BigQuery 库的 Metabase database id |
| `AGENT_REPORT_RANGE_START` / `_END_EXCLUSIVE` / `_TIMEZONE` | `2025-11-01` / `2026-05-01` / `US/Pacific` | 月度用量报表的时间范围与时区 |

---

## 两种执行模式

- **`pipeline`（默认）**：意图解析后路由到固定节点。行为确定、可离线 dry-run、便于测试与审计。
- **`tools`**：LLM 通过 function calling 自行调用 `list_databases`/`list_tables`/`list_fields`/`run_aggregation`/`run_sql`，拿到结果后综合作答。需要 `OPENAI_API_KEY`；缺 key 或 dry-run 时自动走 pipeline。所有工具复用相同的只读校验、查询策略与人工审批闸口。

> dry-run 始终走 pipeline（保证“本地样例、不调 LLM、可复现”），即使 `AGENT_MODE=tools`。

---

## 安全与策略

- **只读 SQL 校验**：必须以 `SELECT`/`WITH` 开头、单语句；对字面量/注释/反引号做掩码后再比对关键词黑名单（`INSERT/UPDATE/DELETE/DROP/...` 及 BigQuery scripting：`CALL/EXECUTE/EXPORT/LOAD/DECLARE`）。中文注释/中文字符串字面量不影响 SQL 完整性（只有字面量/注释之外的中文才被视为自然语言尾巴截断）。
- **查询策略**（`policy/query_policy.py`）：`limit ≤ 200`、聚合函数白名单、source 类型校验；sum/avg 自动选字段时排除主键/外键。
- **人工审批**：生成的 SQL / 结构化查询执行前需显式授权（`decision=approve`）；粘贴新 SQL 不会被误判为对上一条的批准；拒绝短语优先于批准短语（"不要执行"/"do not execute" 永远是拒绝）；批准是原子的 take-once 操作，且绑定被 review 内容的指纹（`agent/sql_review.py:program_fingerprint`），执行内容与审批内容不一致时会重新进入审批。tools 模式挂起时同时记录目标 `database_name`，恢复执行不会落到默认库。
- **鉴权**：`AGENT_API_TOKEN`（常量时间比较）+ 可选 `AGENT_REQUIRE_TOKEN`；真实模式无 token 启动时打印告警。注意当前是单一共享 token，多用户隔离尚未实现。
- **错误脱敏**：`/api/ask` 失败只返回异常类名，完整异常进服务端日志。
- **重试**：Metabase 客户端对 GET/HEAD 在 429/502/503/504 时短暂重试；POST 不自动重试，避免重复执行。

---

## 测试

```bash
uv run pytest                       # 209 单测
uv run ruff check src tests scripts # lint
uv run python -m compileall src tests
```

覆盖范围（节选）：页面加载、`/api/ask` dry-run、LangGraph workflow、中文语义解析、会话记忆、表级聚合、最近 N 天过滤与按天分组、Metric 路径回答合成、只读 SQL 策略、工具循环（分发/迭代上限/重复检测/审批挂起恢复）、双 wire 协议 transport、SSE 流式、CLI（ask/version/ping）、SQLite 状态共享与 TTL。

---

## 部署

开发/内网：

```bash
uv run metabase-agent web --host 127.0.0.1 --port 8765
```

服务化（uvicorn）：

```bash
uv run uvicorn metabase_agent.api.app:app --host 0.0.0.0 --port 8765
```

Docker（多阶段构建、非 root 运行、自带 HEALTHCHECK；镜像内含 `skills/`）：

```bash
docker build -t metabase-agent .
# 状态文件默认写到容器内 /app/data，挂卷持久化：
docker run --env-file .env -p 8765:8765 -v metabase-agent-data:/app/data metabase-agent
```

建议：

- `.env` 只放服务器本地，不提交 git；Metabase Key 用只读权限；入口设 `AGENT_API_TOKEN`（生产建议 `AGENT_REQUIRE_TOKEN=true`）。
- **单 worker（默认）**：`AGENT_STORE=memory`，状态在进程内 + 本地 JSON。
- **多 worker（单机）**：设 `AGENT_STORE=sqlite`，`AGENT_STATE_PATH` 指向共享 `.db`（如 `/var/lib/metabase-agent/state.db`），各 worker 经 SQLite（WAL）共享会话/审批/表上下文；可选 `AGENT_SESSION_TTL_SECONDS` 清理过期会话。
- **跨主机多副本**：设 `AGENT_STORE=mongodb` + `AGENT_STORE_MONGODB_URI`，会话/审批/表上下文存 MongoDB（集合与索引自动创建，审批 `claim` 为原子 take-once）；tools 模式的短期记忆走这条即可跨副本共享。

---

## 改进点 / Roadmap

已知限制与后续方向：

- **真实网关 tools 联调**：`/responses` 的 tool-calling 形状按 OpenAI 规范实现，换自建网关后建议先 `uv run metabase-agent ping` 验证连通，再端到端验证 tools 模式。
- **pipeline 语义解析是正则**：`intent_parser` 依赖中文正则与词表，表述偏差易落入兜底分支；tools 模式是其长期替代。
- **跨主机多副本**：当前 SQLite 后端覆盖单机多进程；跨主机需要 Redis/外部 DB 后端（`api/store.py` 接口已抽象，可加 `RedisStore`）。
- **报表为模板驱动的单一报表**：月度 web/api 用量报表是固定模板 + 参数；更多报表建议继续模板化或落成 Metabase saved question。
- **CLI tools 审批**：CLI 为单次执行，触发 `run_sql` 审批时只打印 SQL，不支持交互式授权（请用 Web 端）。
- **业务默认值**：`METABASE_BIGQUERY_DATABASE_ID`、报表日期等仍是 akool 部署默认值，开源/换环境前应清空。
- **CI/Docker 实跑**：CI 工作流（lint + pytest + docker build）与 Dockerfile 已就绪，需推到远端 + 启 Docker daemon 验证。
- **多用户鉴权**：当前是单一共享 `AGENT_API_TOKEN`，持 token 者可读写任意 session 与任意 tenant/user 的长期记忆；多用户场景需要把身份绑定到凭证（per-tenant token 或接入认证系统），禁止请求体直接指定 tenant_id/user_id。
- **只读校验的纵深防御**：应用层校验是手写词法器，建议生产侧同时用只读 BigQuery 服务账号 + `maximum_bytes_billed`，或引入 `sqlglot` 做 AST 级断言。
- **长期记忆**：已加入 MongoDB 结构化 memory repository、pgvector 向量索引接口、skills 解析注入；默认关闭长期记忆，详见 `docs/memory-skills-implementation.md`。
- **pgvector 初始化**：默认 `AGENT_PGVECTOR_AUTO_CREATE=true`，启动时自动建库/扩展/表/索引（连接账号需有建库权限）；也可手动 `uv run python scripts/init_pgvector_memory.py`。启用后跑 `uv run metabase-agent ping` 会额外自检 `memory.mongodb / memory.embedding / memory.pgvector` 是否连通，避免"配了不生效"。

---

## 常见问题

换端口：

```bash
uv run metabase-agent web --port 8766
```

停止服务：启动终端按 `Ctrl+C`。

页面无返回，先用 dry-run 确认本地流程：

```bash
uv run metabase-agent ask "上周收入趋势怎么样？" --dry-run
```

真实模式失败，优先检查：`.env` 是否存在、`OPENAI_BASE_URL` 是否兼容 `/v1`、`METABASE_BASE_URL`/`METABASE_API_KEY` 是否正确有权限、`AGENT_DRY_RUN=false` 是否设置——或直接 `uv run metabase-agent ping`。

时间字段选不对时，在问题里补字段名：

```text
business_data 下fs_times 最近7天的每天的数据count，时间字段 last_sub_upgrade_time
```
