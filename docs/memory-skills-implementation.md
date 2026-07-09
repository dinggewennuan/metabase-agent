# Memory + Skills 实现说明

本项目已在现有 `pipeline` / `tools` 双内核上接入长期记忆和 skills 上下文，不强制切换到 DeepAgents runtime。

## 设计分工

| 能力 | 实现 |
|---|---|
| 短期 session 历史 | 保留现有 JSON/SQLite session store |
| LangGraph checkpoint | 可选 `MongoDBSaver`，按 `session_id` 作为 `thread_id` |
| 结构化长期记忆 | `metabase_agent.memory`，MongoDB repository |
| 向量检索 | `PgVectorIndex`，pgvector 只保存 embedding 与 memory id |
| skills | `skills/*/SKILL.md`，轻量 `SkillRegistry` 解析和注入 |
| prompt 注入 | API/CLI 请求开始时加载 memory context + skills context |

MongoDB 是长期记忆事实源；pgvector 是语义召回索引。pgvector 返回 memory id 后，再回 MongoDB 取完整 memory record。

## 配置项

默认长期记忆关闭，skills 开启但目录不存在时自动为空。

```env
AGENT_TENANT_ID=default
AGENT_USER_ID=

AGENT_CHECKPOINT_BACKEND=none
AGENT_CHECKPOINT_MONGODB_URI=
AGENT_CHECKPOINT_MONGODB_DATABASE=metabase_agent_checkpoints
AGENT_CHECKPOINT_TTL_SECONDS=0

AGENT_LONG_TERM_MEMORY_ENABLED=false
AGENT_MONGODB_URI=mongodb://127.0.0.1:27017
AGENT_MONGODB_DATABASE=metabase_agent
AGENT_MEMORY_COLLECTION=agent_memories

AGENT_PGVECTOR_DSN=postgresql://user:pass@127.0.0.1:5432/metabase_agent
AGENT_PGVECTOR_TABLE=memory_embeddings

AGENT_EMBEDDING_PROVIDER=hash
AGENT_EMBEDDING_MODEL=text-embedding-3-small
AGENT_EMBEDDING_DIMENSIONS=1536

SILICONFLOW_API_KEY=
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1

AGENT_SKILLS_ENABLED=true
AGENT_SKILLS_PATH=skills
AGENT_SKILLS_MAX_CHARS=6000
```

生产建议：

```env
AGENT_LONG_TERM_MEMORY_ENABLED=true
AGENT_EMBEDDING_PROVIDER=openai
```

如果 OpenAI embedding 暂不可用，可以切到 SiliconFlow：

```env
AGENT_LONG_TERM_MEMORY_ENABLED=true
AGENT_EMBEDDING_PROVIDER=siliconflow
AGENT_EMBEDDING_MODEL=BAAI/bge-m3
AGENT_EMBEDDING_DIMENSIONS=1024
SILICONFLOW_API_KEY=你的 key
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
```

SiliconFlow provider 会按如下请求生成 embedding：

```bash
curl --location 'https://api.siliconflow.cn/v1/embeddings' \
  --header 'Authorization: Bearer ${SILICONFLOW_API_KEY}' \
  --header 'Content-Type: application/json' \
  --data '{"input":"Hello, world!","model":"BAAI/bge-m3"}'
```

`AGENT_USER_ID` 为空时，API 会默认使用 `session_id` 作为 user id。多用户系统应该从认证身份传入 `user_id`。

`AGENT_CHECKPOINT_BACKEND=mongodb` 时，API 会用 LangGraph `MongoDBSaver` 保存 graph checkpoint；它解决的是同一个 `session_id/thread_id` 内的 graph 状态恢复，不等同于长期记忆。长期记忆仍由 `AGENT_LONG_TERM_MEMORY_ENABLED` 控制。

## pgvector DDL

根据 embedding 维度创建表。`text-embedding-3-small` 默认是 1536 维；如果使用本地 hash provider，也会按 `AGENT_EMBEDDING_DIMENSIONS` 生成同维度向量，仅用于测试和离线开发。

推荐直接运行初始化脚本：

```bash
uv run python scripts/init_pgvector_memory.py
```

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memory_embeddings (
  id text PRIMARY KEY,
  tenant_id text NOT NULL,
  user_id text NOT NULL,
  scope text NOT NULL,
  memory_type text NOT NULL,
  memory_id text NOT NULL,
  content text NOT NULL,
  embedding vector(1536) NOT NULL,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
  status text NOT NULL,
  created_at timestamptz NOT NULL,
  updated_at timestamptz NOT NULL
);

CREATE INDEX IF NOT EXISTS memory_embeddings_vector_idx
ON memory_embeddings
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS memory_embeddings_filter_idx
ON memory_embeddings (tenant_id, user_id, memory_type, status);
```

如果显式把 hash provider 改成 64 维：

```sql
embedding vector(64) NOT NULL
```

同时设置：

```env
AGENT_EMBEDDING_DIMENSIONS=64
```

## 当前写入策略

两层提取器并存：

**规则提取器（始终开启，确定性基线）**：

- 用户说“以后/记住/默认/偏好”且提到中文时，写 `profile.language`。
- 用户表达“简洁/直接/详细/工程”等风格时，写 `profile.answer_style`。
- 查询计划里出现 database/schema/table 时，写默认数据库、schema 和最近表上下文。
- 查询结束后写一条 `episodic` 分析事件。

**LLM 提取器（`AGENT_MEMORY_LLM_EXTRACTOR=true` 且配置了 `OPENAI_API_KEY` 时启用）**，实现 `memory/extractor.py`，走的正是规划中的链路：

```text
本轮问答 -> LLM 提取候选（仅 semantic/procedural，json_mode）
        -> 代码规则过滤（置信度 >= 0.6、长度、凭据模式拒绝、key 字符集校验）
        -> procedural 一律强制 pending_review（无论模型声称什么状态）
        -> _upsert_candidate 去重/冲突 -> MongoDB -> pgvector
```

要点：

- 能识别否定语义与自由格式偏好（例如“users_20xx 这种表之后基本不怎么关注了”会被提为
  `rule.*` 的 procedural 提案，等待 `POST /api/memories/{id}/status` 审核激活）。
- LLM 提取失败时自动降级，只写规则提取器的候选，不影响本轮请求。
- 冲突保护：同 key 内容相同的重复提案只刷新 `last_seen`，绝不把 ACTIVE 规则降级；
  与 ACTIVE 规则内容冲突的提案写入 `<key>.conflict.<hash>`（metadata 带 `conflicts_with`），
  原规则保持不变，由人工裁决。

## 当前检索策略

请求开始时：

1. key 查询 semantic profile slots：
   - `profile.language`
   - `profile.answer_style`
   - `profile.default_database`
   - `profile.default_schema`
   - `analytics.table_context`
2. 查询 active procedural rules。
3. 对当前问题生成 embedding。
4. 用 pgvector 检索相关 semantic/episodic memory ids。
5. 从 MongoDB repository 读取完整 memory records。
6. 渲染成 `memory_context` 注入 prompt。

## 人工审核与管理接口

Procedural memory 建议先写成 `pending_review`，再由人工或后台规则确认后改成 `active`。当前 API 已提供基础管理入口：

```http
GET /api/memories?tenant_id=default&user_id=u1&memory_type=procedural&status=pending_review
POST /api/memories
POST /api/memories/{memory_id}/status
```

写入一条待审核规则：

```json
{
  "tenant_id": "default",
  "user_id": "u1",
  "memory_type": "procedural",
  "key": "rule.sql.require_approval",
  "content": "执行 SQL 前必须先让用户确认。",
  "status": "pending_review",
  "confidence": 0.9
}
```

审核通过：

```json
{
  "tenant_id": "default",
  "user_id": "u1",
  "status": "active"
}
```

`active` procedural memory 会在下一次请求开始时通过 key/list 查询加载进 `memory_context`，不走向量检索。

## Skills

默认 skills：

- `skills/metabase-analysis`
- `skills/sql-safety`
- `skills/agent-memory`

Agent 启动时只扫描 `SKILL.md` 的 frontmatter：

```yaml
---
name: sql-safety
description: Use this skill for SQL review, SQL approval, read-only policy...
---
```

请求时按问题关键词匹配 skill，读取完整 `SKILL.md` 并注入 prompt。当前实现是轻量版 progressive loading，后续可以迁移到 DeepAgents 原生 skills backend。

## 代码入口

- memory 模型：`src/metabase_agent/memory/models.py`
- MongoDB repository：`src/metabase_agent/memory/repository.py`
- pgvector index：`src/metabase_agent/memory/vector.py`
- memory manager：`src/metabase_agent/memory/manager.py`
- skills registry：`src/metabase_agent/skills/registry.py`
- tools prompt 注入：`src/metabase_agent/agent/tool_loop.py`
- API 上下文加载与 MongoDBSaver checkpoint：`src/metabase_agent/api/app.py`
- pgvector 初始化：`scripts/init_pgvector_memory.py`
- CLI 上下文加载：`src/metabase_agent/cli/app.py`
