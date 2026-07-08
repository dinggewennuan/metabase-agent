# Agent Memory 设计笔记：LangGraph + MongoDB

记录日期：2026-07-03

本文整理前面对 Agent、LangGraph memory、MongoDB 持久化、长期记忆分类、向量检索、去重冲突判断、Prompt 注入，以及 HelloAgents 项目 memory 管理方式的讨论。

重点结论先放前面：

1. `MongoDBSaver` 和 `MongoDBStore` 都是“存储”，但职责完全不同。
2. `MongoDBSaver` 负责 graph checkpoint，也就是 thread/session 内的短期状态恢复。
3. `MongoDBStore` 负责长期记忆，也就是跨 session、跨 thread 的用户/组织/业务记忆。
4. 使用 `MongoDBStore` 后，LangGraph 不会自动知道什么该记住，也不会自动把记忆塞回 Prompt。
5. 生产级长期记忆必须自己设计：提取、判断、去重、冲突处理、写入、检索、Prompt 注入、过期、审计。
6. 长期记忆通常分三类：Semantic Memory、Episodic Memory、Procedural Memory。
7. 是否写入哪类记忆，核心不是数据库决定的，而是“LLM 提议 + 代码规则校验 + 旧记忆比对 + 写入策略”共同决定。
8. 向量检索不是所有记忆都需要；稳定字段优先 key 查询，开放语义内容才用 vector search。

---

## 1. LangGraph 中 memory 的两层含义

LangGraph 里讨论 memory 时，容易混淆两个层级：

1. Short-term memory，也叫 thread-level memory。
2. Long-term memory，也叫 cross-thread memory。

### 1.1 Short-term memory：短期记忆

短期记忆是某个 graph thread 内的状态。

可以理解为：

- 当前会话走到哪一步。
- 当前 graph state 里有哪些 messages。
- 工具调用结果是否已经回来。
- 中断后能不能恢复。
- 多轮对话里前几轮的消息是否还在。

在 LangGraph 中，它通常由 checkpointer/saver 保存。

常见实现：

- `InMemorySaver`
- `PostgresSaver`
- `MongoDBSaver`
- `RedisSaver` 或其他后端实现

`InMemorySaver` 只适合开发和测试，因为进程一重启，状态就丢了。

生产环境应该使用数据库后端，比如 Postgres、MongoDB、Redis。

### 1.2 Long-term memory：长期记忆

长期记忆不是某个 thread 的状态，而是跨会话、跨 thread、跨时间保存的信息。

可以理解为：

- 用户偏好。
- 用户身份画像。
- 用户长期目标。
- 项目背景。
- 过去发生过的重要事件。
- Agent 的工作规范。
- 对某个租户、团队、用户长期有效的配置。

在 LangGraph 中，它通常由 store 保存。

例如：

- `MongoDBStore`
- `PostgresStore`
- `InMemoryStore`
- 自定义 Store

### 1.3 两者区别

| 维度 | Saver / Checkpointer | Store |
|---|---|---|
| 典型实现 | `MongoDBSaver` | `MongoDBStore` |
| 负责内容 | graph checkpoint | long-term memory |
| 生命周期 | thread/session 级别 | user/tenant/app 级别 |
| 查询方式 | 按 thread/checkpoint 恢复 | 按 namespace/key/vector 查询 |
| 是否自动注入 Prompt | 通常 graph state 自带 | 不会，需要你自己检索并注入 |
| 用途 | 断点续跑、会话恢复、多轮状态 | 偏好、画像、事实、经验、规则 |

一句话：

`MongoDBSaver` 解决“这个对话进行到哪了”；`MongoDBStore` 解决“这个用户/业务长期要记住什么”。

---

## 2. MongoDBSaver 和 MongoDBStore 的区别

用户问题是：`mongodb_saver` 和 `mongodb_store` 都是存储，区别是什么？

答案是：它们都是落 MongoDB，但业务语义不同。

### 2.1 MongoDBSaver

`MongoDBSaver` 是 checkpointer。

它保存的是 LangGraph graph 执行状态。

典型保存内容包括：

- thread id
- checkpoint id
- graph state
- messages
- node 执行进度
- pending task
- interrupt 信息
- checkpoint metadata

它服务的是 graph runtime。

你通常不应该把它当作“用户长期记忆库”来查询。

示例用途：

```python
graph = builder.compile(checkpointer=mongodb_saver)

result = graph.invoke(
    input,
    config={"configurable": {"thread_id": "session_123"}},
)
```

当下一次用同一个 `thread_id` 调用时，LangGraph 可以恢复这个 thread 的上下文。

### 2.2 MongoDBStore

`MongoDBStore` 是 long-term store。

它保存的是你自己定义的长期记忆对象。

典型保存内容包括：

- 用户偏好
- 用户画像
- 项目事实
- 历史事件摘要
- Agent 操作规范
- 可检索知识
- 向量 embedding
- memory metadata

它服务的是你的业务 memory 系统。

示例用途：

```python
namespace = ("tenant", tenant_id, "user", user_id, "semantic")

store.put(
    namespace,
    key="profile.answer_style",
    value={
        "content": "用户偏好简洁、直接的中文技术回答",
        "confidence": 0.92,
        "updated_at": now,
    },
)
```

读取时：

```python
memory = store.get(namespace, "profile.answer_style")
```

或语义搜索：

```python
memories = store.search(
    namespace,
    query="用户喜欢什么回答风格？",
    limit=5,
)
```

### 2.3 为什么不能只用 Saver

因为 saver 保存的是 thread 内状态。

如果用户开启新 session，新 `thread_id` 下，旧 thread 的 messages 不一定应该全部带过来。

长期记忆应该经过筛选、压缩、结构化、去重和权限隔离，而不是把所有历史对话原样塞给模型。

---

## 3. tenant_id 是什么

`tenant_id` 是租户 ID。

它用于多租户系统中的数据隔离。

如果你的系统只有个人用户，可以简单理解：

- `tenant_id` = 组织 ID
- `user_id` = 用户 ID

例如 SaaS 系统：

```text
tenant_id = company_akool
user_id   = user_123
```

同一个用户可能属于不同团队，同一个团队有多个用户。

长期记忆必须考虑隔离范围：

| 范围 | 示例 namespace | 说明 |
|---|---|---|
| 用户级 | `("tenant", tenant_id, "user", user_id, "semantic")` | 某个用户自己的偏好 |
| 租户级 | `("tenant", tenant_id, "org", "semantic")` | 整个组织通用信息 |
| 应用级 | `("app", app_id, "procedural")` | 某个 Agent 通用规则 |
| 项目级 | `("tenant", tenant_id, "project", project_id, "semantic")` | 某个项目背景 |

设计原则：

1. 用户个人偏好不要污染整个组织。
2. 组织规则不要只存在某个用户下面。
3. 跨租户绝不能混查。
4. namespace 应该让你一眼看出记忆的作用范围。

---

## 4. namespace 是什么

前面提到：

```python
semantic_ns = ("tenant", tenant_id, "user", user_id, "semantic")
```

这是一个 namespace。

它不是 MongoDB 原生概念，而是 LangGraph Store 层用来组织记忆的逻辑路径。

可以把它理解成“记忆目录”。

例如：

```text
tenant/acme/user/u_001/semantic
tenant/acme/user/u_001/episodic
tenant/acme/user/u_001/procedural
tenant/acme/org/semantic
tenant/acme/project/p_123/semantic
```

namespace 的作用：

- 数据隔离。
- 缩小搜索范围。
- 避免不同用户记忆混在一起。
- 支持不同类型记忆分开存储。
- 支持 key 查询和 vector search 的范围限定。

---

## 5. 三类长期记忆

长期记忆常分为三类：

1. Semantic Memory
2. Episodic Memory
3. Procedural Memory

### 5.1 Semantic Memory：语义记忆 / 事实记忆

Semantic Memory 保存的是长期稳定的事实、偏好、画像、关系。

它回答的问题是：

“用户是谁？用户偏好什么？业务事实是什么？”

示例：

```json
{
  "memory_type": "semantic",
  "subject": "user",
  "predicate": "answer_style",
  "value": "prefers concise Chinese technical answers",
  "content": "用户偏好简洁、直接的中文技术回答",
  "confidence": 0.92
}
```

适合写入 semantic memory 的内容：

- 用户喜欢中文回答。
- 用户希望技术解释直接、详细。
- 用户使用 MongoDB 存储 LangGraph memory。
- 用户项目使用 Python。
- 用户更关心工程实现而不是纯理论。
- 某个项目使用 MongoDB 8.3.4。

不适合写入 semantic memory 的内容：

- “用户刚才问了一个问题。”
- “这轮对话里用户说继续。”
- “用户今天执行了某个一次性命令。”

Semantic Memory 通常是长期有效的。

### 5.2 Episodic Memory：情节记忆 / 事件记忆

Episodic Memory 保存的是发生过的事件、经历、任务过程。

它回答的问题是：

“以前发生过什么？做过什么？结果是什么？”

示例：

```json
{
  "memory_type": "episodic",
  "event": "mongodb_migration",
  "content": "2026-07-03 将本地 MongoDB 从 5.0.21 迁移到 8.3.4，删除了 fsweb.quotarecords 和 facewebs.quotarecords 后补建索引成功。",
  "time": "2026-07-03T02:00:00+08:00",
  "entities": ["mongodb", "fsweb.quotarecords", "facewebs.quotarecords"],
  "outcome": "success"
}
```

适合写入 episodic memory 的内容：

- 某次迁移做了什么。
- 某个 bug 如何排查。
- 用户曾确认删除某两个集合。
- 某个部署过程的关键结果。
- 某次任务中的失败原因和解决方式。

Episodic Memory 不一定永久有效，但对后续排查、复盘、连续任务很有价值。

### 5.3 Procedural Memory：程序性记忆 / 行为规范

Procedural Memory 保存的是 Agent 应该如何做事的规则、流程、偏好约束。

它回答的问题是：

“以后遇到类似任务应该怎么做？”

示例：

```json
{
  "memory_type": "procedural",
  "rule": "mongodb_migration_safety",
  "content": "迁移 MongoDB 前必须先保留旧数据目录备份；删除旧数据目录前必须确认新库数据和索引验证通过。",
  "status": "active",
  "confidence": 0.95
}
```

适合写入 procedural memory 的内容：

- 用户要求生产操作前先备份。
- 用户偏好直接执行，但删除数据前要确认。
- Agent 在该项目中应优先使用 MongoDB 8.3.4。
- 对某类任务必须跑哪些验证步骤。
- 某个团队的代码规范。

Procedural Memory 风险最高，因为它会改变 Agent 后续行为。

所以建议：

```text
LLM 提议 procedural 记忆
-> 先进入 pending_review
-> 规则或人工确认
-> active 后才自动注入 Prompt
```

---

## 6. 三类长期记忆如何区分

核心判断问题：

```text
这条信息是在描述“事实”、描述“事件”，还是描述“以后怎么做”？
```

### 6.1 判断表

| 类型 | 判断问题 | 示例 |
|---|---|---|
| Semantic | 这是不是长期稳定的事实或偏好？ | 用户偏好中文技术回答 |
| Episodic | 这是不是某次发生过的事件？ | 2026-07-03 完成 MongoDB 迁移 |
| Procedural | 这是不是以后指导 Agent 行为的规则？ | 删除生产数据前必须先确认 |

### 6.2 更细规则

如果句子可以改写成：

```text
用户/项目/组织 是/有/偏好/使用 ...
```

通常是 semantic。

例如：

```text
用户使用 MongoDB 作为 LangGraph 长期记忆存储。
```

如果句子可以改写成：

```text
在某个时间，发生了某件事，结果是 ...
```

通常是 episodic。

例如：

```text
2026-07-03，用户将 MongoDB 5.0 数据迁移到 8.3.4。
```

如果句子可以改写成：

```text
以后遇到 X，要 Y。
```

通常是 procedural。

例如：

```text
以后涉及删除数据库目录时，需要先确认目标路径和服务状态。
```

---

## 7. 是否让大模型判断

可以让大模型判断，但不能只靠大模型。

推荐架构是：

```text
用户对话
-> LLM 提取候选记忆
-> 代码规则过滤
-> 查旧记忆做去重/冲突判断
-> 按类型写入 Store
```

LLM 适合做：

- 从自然语言中提取候选记忆。
- 判断记忆类型。
- 生成结构化 JSON。
- 生成摘要。
- 判断语义相似。
- 解释冲突原因。

代码规则适合做：

- 过滤低价值内容。
- 校验 JSON schema。
- 检查 confidence 阈值。
- 检查是否含敏感信息。
- 控制 namespace。
- 控制 key。
- 控制是否自动写入。
- 对 procedural memory 做人工确认门槛。

数据库适合做：

- 精确查询。
- 向量检索。
- 去重检索。
- 版本保存。
- metadata 过滤。
- last_seen 更新。

不能把“记忆管理”完全交给模型，因为模型可能：

- 把临时信息当长期信息。
- 把用户一次性指令变成永久规则。
- 写入错误事实。
- 写入过度隐私内容。
- 重复写入相似记忆。
- 覆盖掉更可信的旧记忆。

---

## 8. 推荐的候选记忆 JSON 结构

### 8.1 Semantic 示例

```json
{
  "memory_type": "semantic",
  "subject": "user",
  "predicate": "answer_style",
  "value": "prefers concise Chinese technical answers",
  "content": "用户偏好简洁、直接的中文技术回答",
  "confidence": 0.92,
  "source": "conversation",
  "evidence": "用户多次要求详细但直接地解释 LangGraph memory 原理",
  "created_at": "2026-07-03T02:10:00+08:00",
  "updated_at": "2026-07-03T02:10:00+08:00",
  "last_seen": "2026-07-03T02:10:00+08:00"
}
```

字段解释：

| 字段 | 含义 |
|---|---|
| `memory_type` | 记忆类型，semantic/episodic/procedural |
| `subject` | 记忆主体，通常是 user/project/org/agent |
| `predicate` | 属性名或关系名 |
| `value` | 结构化值，方便程序读取 |
| `content` | 给 LLM 看的自然语言版本 |
| `confidence` | 置信度 |
| `source` | 来源 |
| `evidence` | 证据，说明为什么写入 |
| `created_at` | 首次创建时间 |
| `updated_at` | 最近更新时间 |
| `last_seen` | 最近一次在对话中被观察到或确认的时间 |

### 8.2 Episodic 示例

```json
{
  "memory_type": "episodic",
  "event_type": "database_migration",
  "title": "MongoDB 5.0 到 8.3.4 迁移",
  "content": "用户在本地将 MongoDB Community 5.0.21 数据迁移到 8.3.4，因磁盘不足删除 fsweb.quotarecords 和 facewebs.quotarecords 后完成索引补建。",
  "entities": ["mongodb", "fsweb.quotarecords", "facewebs.quotarecords"],
  "outcome": "success",
  "time": "2026-07-03T02:00:00+08:00",
  "confidence": 0.98
}
```

### 8.3 Procedural 示例

```json
{
  "memory_type": "procedural",
  "rule_id": "database_destructive_ops_confirm",
  "content": "涉及删除数据库目录、集合或 Docker 数据时，必须先确认目标、影响范围和可恢复性。",
  "trigger": "database destructive operation",
  "action": "confirm before execution",
  "status": "pending_review",
  "confidence": 0.9,
  "created_at": "2026-07-03T02:10:00+08:00"
}
```

---

## 9. profile / semantic slot 是什么

前面提到：

```python
semantic_ns = ("tenant", tenant_id, "user", user_id, "semantic")
```

然后：

```text
精确查：适合 profile / semantic slot
```

### 9.1 profile

profile 是用户画像。

它通常由多个稳定字段组成：

```json
{
  "language": "zh-CN",
  "answer_style": "direct_detailed_technical",
  "timezone": "Asia/Shanghai",
  "preferred_stack": ["Python", "MongoDB", "LangGraph"],
  "risk_preference": "confirm_before_destructive_ops"
}
```

这些字段通常不需要向量搜索。

它们有明确 key，可以直接读取。

### 9.2 semantic slot

semantic slot 是一个稳定语义槽位。

例如：

```text
profile.answer_style
profile.language
project.default_database
project.agent_framework
org.security_policy
```

这些都是可以直接算出 key 的内容。

示例：

```python
key = "profile.answer_style"
memory = store.get(semantic_ns, key)
```

### 9.3 为什么 slot 适合 key 查询

因为你在写入前已经知道它是什么字段。

例如用户说：

```text
以后你都用中文回答我。
```

这不是一个开放语义搜索问题，而是一个明确偏好字段：

```text
profile.language = zh-CN
```

所以应该直接写入固定 key。

---

## 10. 什么时候用 key 查询，什么时候用向量检索

### 10.1 key 查询适合什么

如果你能提前确定 key，就用 key 查询。

适合：

- 用户语言偏好。
- 回答风格。
- 用户时区。
- 默认数据库。
- 默认项目。
- 明确配置。
- 稳定 profile 字段。
- procedural rule id。

示例：

```python
store.get(
    ("tenant", tenant_id, "user", user_id, "semantic"),
    "profile.answer_style",
)
```

优点：

- 快。
- 准。
- 不依赖 embedding。
- 不会召回无关内容。
- 成本低。

### 10.2 向量检索适合什么

如果你不知道具体 key，只知道当前问题语义，就用向量检索。

适合：

- 历史事件回忆。
- 模糊项目背景。
- 用户曾经说过的需求。
- 过去 bug 排查记录。
- 非结构化偏好。
- 业务知识片段。
- 与当前任务语义相关的记忆。

示例：

```python
store.search(
    ("tenant", tenant_id, "user", user_id, "episodic"),
    query="之前 MongoDB 迁移遇到过什么问题？",
    limit=5,
)
```

优点：

- 能处理模糊语义。
- 能召回相似事件。
- 适合非结构化内容。

缺点：

- 可能召回不相关内容。
- 需要 embedding。
- 需要向量索引。
- 需要阈值和 rerank。

### 10.3 混合策略

生产推荐混合：

```text
固定 profile / rules -> key 查询
相关历史 / 知识片段 -> vector search
最后按 relevance、recency、confidence、status 排序
```

伪代码：

```python
def load_memories_for_prompt(store, tenant_id, user_id, query):
    semantic_ns = ("tenant", tenant_id, "user", user_id, "semantic")
    episodic_ns = ("tenant", tenant_id, "user", user_id, "episodic")
    procedural_ns = ("tenant", tenant_id, "user", user_id, "procedural")

    profile = [
        store.get(semantic_ns, "profile.language"),
        store.get(semantic_ns, "profile.answer_style"),
        store.get(semantic_ns, "profile.tech_stack"),
    ]

    related_events = store.search(
        episodic_ns,
        query=query,
        limit=5,
    )

    active_rules = store.search(
        procedural_ns,
        query=query,
        filter={"status": "active"},
        limit=3,
    )

    return build_memory_context(profile, related_events, active_rules)
```

---

## 11. 什么时候写入记忆

不是每轮对话都应该写入长期记忆。

推荐写入时机：

1. 用户明确表达长期偏好。
2. 用户纠正 Agent 行为。
3. 用户提供稳定背景。
4. 完成一个重要任务后。
5. 出现可复用的经验。
6. 用户明确说“记住”。
7. 用户确认某个规则以后都适用。
8. 会话结束时做总结提取。

### 11.1 实时写入

适合：

- 用户明确说“以后都这样”。
- 用户说“记住我喜欢中文”。
- 用户提供重要身份信息。

优点：

- 及时。
- 不容易丢。

缺点：

- 容易误写。
- 需要更严格规则。

### 11.2 会话结束写入

适合：

- 总结本 session。
- 提取 episodic memory。
- 合并重复偏好。
- 生成任务复盘。

优点：

- 信息更完整。
- 更容易判断重要性。

缺点：

- 如果 session 异常结束，可能丢失候选。

### 11.3 混合写入

推荐：

```text
明确偏好/规则 -> 实时提取
任务结果/过程经验 -> session end 提取
低置信候选 -> pending_review
```

---

## 12. 如何判断并写入哪类记忆

推荐 pipeline：

```text
messages
-> memory_extractor
-> candidate memories
-> schema validation
-> type-specific filters
-> duplicate/conflict check
-> write/update/pending/drop
```

### 12.1 LLM 提取候选

System prompt 可以要求模型只输出 JSON：

```text
你是记忆提取器。
从对话中提取值得长期保存的记忆。
只提取对未来有用、稳定、可复用的信息。
不要提取临时指令、寒暄、一次性问题。

记忆类型：
- semantic：稳定事实、偏好、画像
- episodic：发生过的重要事件
- procedural：未来行为规则

输出 JSON array。
每个对象必须包含：
memory_type, content, confidence, rationale
```

### 12.2 代码规则过滤

伪代码：

```python
def filter_candidate(candidate):
    if candidate["confidence"] < 0.75:
        return "drop"

    if len(candidate["content"]) < 8:
        return "drop"

    if contains_secret(candidate["content"]):
        return "drop"

    if candidate["memory_type"] == "procedural":
        return "pending_review"

    if is_temporary_instruction(candidate["content"]):
        return "drop"

    return "accept"
```

### 12.3 按类型写入

Semantic：

- 能映射到 profile slot，就写固定 key。
- 不能映射，就写 hash key 或 semantic note。
- 需要支持更新和覆盖。

Episodic：

- 通常 append。
- 用事件 ID 或内容 hash 作为 key。
- 保留时间、实体、结果。

Procedural：

- 默认 pending_review。
- 规则确认后 active。
- active 才能自动注入 Prompt。

---

## 13. 查旧记忆做去重和冲突判断

前面流程里有一步：

```text
查旧记忆做去重/冲突判断
```

这是长期记忆系统最关键的部分之一。

### 13.1 为什么要查旧记忆

如果不查旧记忆，会出现：

- 同一偏好重复写 100 次。
- 用户偏好从“详细回答”变成“简洁回答”，旧记忆仍然存在。
- 同一事件重复保存。
- Prompt 注入越来越长。
- 新旧记忆冲突，模型不知道听谁的。

### 13.2 Semantic 去重

Semantic 常用两种方式：

1. slot key 精确查。
2. 向量相似查。

#### 13.2.1 slot key 精确查

例如：

```python
key = "profile.answer_style"
old = store.get(semantic_ns, key)
```

如果旧值存在：

- 内容相同：更新 `last_seen`。
- 内容更具体：merge。
- 内容冲突：按新证据、时间、置信度处理。

示例：

旧记忆：

```json
{
  "content": "用户喜欢详细中文技术解释",
  "value": "detailed Chinese technical answers",
  "confidence": 0.86
}
```

新候选：

```json
{
  "content": "用户偏好简洁、直接的中文技术回答",
  "value": "concise Chinese technical answers",
  "confidence": 0.92
}
```

这是冲突。

处理方式：

```text
如果新候选来自用户明确纠正，且置信度更高，则覆盖旧值。
旧值可以进入 history 或 superseded 状态。
```

#### 13.2.2 向量相似查

对于非 slot semantic notes：

```python
similar = store.search(
    semantic_ns,
    query=candidate["content"],
    limit=5,
)
```

如果相似度很高：

- 相同：不新增，只更新 last_seen。
- 包含关系：合并。
- 冲突：进入 conflict resolution。

### 13.3 Episodic 去重

Episodic 是事件记忆，不一定要覆盖。

但需要防止同一事件重复保存。

去重依据：

- event_type
- time window
- entities
- outcome
- content hash
- vector similarity

伪代码：

```python
def dedupe_episodic(candidate):
    event_key = make_event_key(
        candidate["event_type"],
        candidate["time"][:10],
        candidate["entities"],
    )

    old = store.get(episodic_ns, event_key)
    if old:
        return merge_event(old, candidate)

    similar = store.search(
        episodic_ns,
        query=candidate["content"],
        limit=3,
    )

    if top_score(similar) > 0.9:
        return update_last_seen(similar[0])

    return create_new(candidate)
```

### 13.4 Procedural 去重

Procedural 最容易造成行为污染。

推荐：

- rule_id 精确查。
- 同 trigger 下只允许少量 active 规则。
- 新规则默认 `pending_review`。
- 与 active 规则冲突时不自动覆盖。

伪代码：

```python
def process_procedural(candidate):
    rule_id = normalize_rule_id(candidate)
    old = store.get(procedural_ns, rule_id)

    if old and old["content"] == candidate["content"]:
        old["last_seen"] = now()
        return store.put(procedural_ns, rule_id, old)

    if old and old["content"] != candidate["content"]:
        return create_pending_conflict(candidate, old)

    candidate["status"] = "pending_review"
    return store.put(procedural_ns, rule_id, candidate)
```

---

## 14. 冲突判断原则

冲突不是简单“新覆盖旧”。

推荐按下面因素判断：

1. 用户是否明确纠正。
2. 新信息是否更近。
3. 新信息置信度是否更高。
4. 旧信息是否被多次确认。
5. 记忆类型是否允许自动覆盖。
6. 是否影响安全或关键行为。

### 14.1 Semantic 冲突处理

可自动覆盖的例子：

```text
旧：用户偏好英文回答
新：用户明确说以后用中文回答
```

应更新为中文。

不可轻易覆盖的例子：

```text
旧：用户生产数据库地址是 A
新：一次对话里提到数据库地址 B
```

应进入 pending 或保留两个候选，等用户确认。

### 14.2 Episodic 冲突处理

Episodic 一般不覆盖，而是 append 或修正。

例如：

```text
旧：迁移失败，原因是磁盘不足
新：后来删除两个大集合后迁移成功
```

应该合并成一个事件链：

```text
initial_status = failed
failure_reason = out_of_disk
resolution = deleted two large collections and rebuilt indexes
final_status = success
```

### 14.3 Procedural 冲突处理

Procedural 冲突必须谨慎。

例如：

```text
旧规则：删除数据前必须确认
新候选：以后直接删除不用确认
```

不应自动覆盖。

应进入：

```text
pending_review
```

---

## 15. last_seen 是什么

`last_seen` 表示这条记忆最近一次被观察到、使用到或被确认的时间。

它不是创建时间。

例如：

```json
{
  "content": "用户偏好中文技术回答",
  "created_at": "2026-06-01T10:00:00+08:00",
  "updated_at": "2026-06-01T10:00:00+08:00",
  "last_seen": "2026-07-03T02:10:00+08:00"
}
```

含义：

- `created_at`：第一次写入。
- `updated_at`：内容最后一次修改。
- `last_seen`：最近一次被再次确认或命中。

用途：

- 判断记忆是否仍然活跃。
- 排序时提高近期确认过的记忆。
- 清理长期未出现的低置信记忆。
- 统计哪些记忆真正有用。

---

## 16. Procedural Memory 的人工确认

前面建议：

```text
procedural 不要自动激活：
LLM 提议 -> pending_review
人工/规则确认 -> active
```

### 16.1 为什么需要确认

Procedural Memory 会影响 Agent 以后怎么做事。

如果错误写入，后果比 semantic/episodic 更严重。

例如错误规则：

```text
以后执行 rm -rf 不需要确认。
```

这种规则绝不能自动 active。

### 16.2 什么时候人工确认

需要人工确认的情况：

- 涉及删除、覆盖、迁移生产数据。
- 涉及权限、安全、隐私。
- 涉及付款、计费、账户。
- 涉及长期改变 Agent 行为。
- 与已有 active rule 冲突。
- LLM confidence 不够高。
- 用户没有明确说“以后都这样”。

可以自动确认的情况：

- 低风险格式偏好。
- 用户明确表达并多次重复。
- 已有规则的轻微补充。
- 规则只影响当前用户，不影响组织。

### 16.3 如何确认

方式一：管理后台

```text
pending_review 列表
-> 管理员查看内容、证据、影响范围
-> approve/reject/edit
```

方式二：对话内确认

```text
我理解你希望以后都用中文回答技术问题。是否记为长期偏好？
```

方式三：规则自动确认

```python
if candidate["memory_type"] == "procedural":
    if is_low_risk(candidate) and user_explicitly_said_always(candidate):
        candidate["status"] = "active"
    else:
        candidate["status"] = "pending_review"
```

---

## 17. 如何主动加入下个 session 的 Prompt 上下文

`MongoDBStore` 不会自动注入 Prompt。

你要在 graph 的某个节点里显式做：

```text
收到用户输入
-> 根据 tenant_id/user_id/thread_id 确定 namespace
-> key 查询 profile/rules
-> vector search 查询相关 episodic/semantic notes
-> 排序、过滤、压缩
-> 构造 memory context
-> 放入 system prompt 或 messages
-> 调 LLM
```

### 17.1 推荐 graph 节点

典型节点：

```text
load_memory_node
-> call_model_node
-> extract_memory_node
-> write_memory_node
```

伪代码：

```python
def load_memory_node(state, config, store):
    tenant_id = config["configurable"]["tenant_id"]
    user_id = config["configurable"]["user_id"]
    user_query = state["messages"][-1].content

    memory_context = load_memories_for_prompt(
        store=store,
        tenant_id=tenant_id,
        user_id=user_id,
        query=user_query,
    )

    return {
        "memory_context": memory_context,
    }
```

### 17.2 Prompt 注入格式

不要把所有 memory 原样塞进去。

推荐结构化注入：

```text
Known long-term memory:

User profile:
- Preferred language: Chinese
- Answer style: direct, technical, detailed when requested

Relevant past events:
- 2026-07-03 migrated local MongoDB from 5.0.21 to 8.3.4.
- User chose to delete fsweb.quotarecords and facewebs.quotarecords.

Active working rules:
- Confirm before destructive database operations.
```

### 17.3 注入顺序

推荐顺序：

1. system policy
2. developer/app instruction
3. active procedural memory
4. user profile semantic memory
5. relevant episodic memory
6. current conversation messages

注意：

Procedural memory 如果进入 Prompt，本质上类似“软 developer instruction”，必须可信。

---

## 18. 向量检索的数据如何存储

向量检索需要保存两类东西：

1. 原始文本或可读内容。
2. embedding 向量。

示例 MongoDB 文档：

```json
{
  "_id": "mem_01J...",
  "tenant_id": "tenant_001",
  "user_id": "user_123",
  "namespace": ["tenant", "tenant_001", "user", "user_123", "episodic"],
  "memory_type": "episodic",
  "content": "2026-07-03 用户完成 MongoDB 5.0 到 8.3.4 迁移。",
  "embedding": [0.0123, -0.0456, 0.0789],
  "metadata": {
    "event_type": "database_migration",
    "entities": ["mongodb"],
    "outcome": "success",
    "confidence": 0.98,
    "status": "active"
  },
  "created_at": "2026-07-03T02:00:00+08:00",
  "updated_at": "2026-07-03T02:00:00+08:00",
  "last_seen": "2026-07-03T02:00:00+08:00"
}
```

向量索引通常建在 `embedding` 字段上。

检索时：

```text
当前 query
-> 生成 query embedding
-> 在同 namespace / tenant / user 范围内做 vector search
-> 返回 topK
-> metadata 过滤
-> rerank
-> 注入 Prompt
```

---

## 19. MongoDB Vector Search 说明

前面本地验证过：

- 普通本地 MongoDB Community 5.0 不支持 `$vectorSearch`。
- 本地安装 MongoDB Community 8.3.4 后，普通 `27017` 服务仍不是 Atlas Vector Search 环境。
- `$vectorSearch` 本地可通过 Atlas CLI local deployment 跑，但它依赖 Docker。
- Atlas 云也支持 MongoDB Atlas Vector Search。

关键区别：

| 环境 | 是否适合普通数据存储 | 是否支持 `$vectorSearch` |
|---|---|---|
| MongoDB Community 8.3 native `27017` | 是 | 普通本地服务不等于 Atlas Vector Search |
| Atlas Local deployment | 是 | 支持，依赖 Docker |
| MongoDB Atlas Cloud | 是 | 支持 |

所以对当前机器：

```text
27017 = native MongoDB 8.3.4，适合 MongoDBSaver/MongoDBStore 普通存储
27018 = Atlas Local Vector Search，之前创建过，但 Docker 关闭后不可用
```

如果 Agent memory 需要向量检索，有三种选择：

1. 使用 Atlas Cloud Vector Search。
2. 使用 Atlas Local + Docker。
3. MongoDB 只存 memory 文档，向量检索交给 Chroma、Qdrant、Milvus、pgvector、Redis Vector 等开源向量库。

### 19.1 是否必须用 MongoDB Vector Search

不必须。

MongoDBStore 可以用于长期记忆的结构化存储。

向量检索可以单独用另一个向量库。

架构可以是：

```text
MongoDB: memory metadata + content + status
Vector DB: memory_id + embedding
```

检索时：

```text
Vector DB search -> memory_id list -> MongoDB fetch full documents
```

这种方式工程上也很常见。

---

## 20. LangGraph 中的实现逻辑

LangGraph 不会自动做长期记忆写入和读取。

你需要自己在 graph 中加入节点。

### 20.1 推荐 graph 流程

```text
START
-> load_memory
-> call_model
-> maybe_tools
-> extract_memory
-> write_memory
-> END
```

也可以把 `extract_memory/write_memory` 放在后台任务。

### 20.2 State 设计

```python
from typing import TypedDict, Any

class AgentState(TypedDict):
    messages: list
    memory_context: str
    candidate_memories: list[dict[str, Any]]
```

### 20.3 load_memory 节点

```python
def load_memory(state: AgentState, config, store):
    tenant_id = config["configurable"]["tenant_id"]
    user_id = config["configurable"]["user_id"]
    query = state["messages"][-1].content

    semantic_ns = ("tenant", tenant_id, "user", user_id, "semantic")
    episodic_ns = ("tenant", tenant_id, "user", user_id, "episodic")
    procedural_ns = ("tenant", tenant_id, "user", user_id, "procedural")

    profile_items = [
        store.get(semantic_ns, "profile.language"),
        store.get(semantic_ns, "profile.answer_style"),
    ]

    related_events = store.search(episodic_ns, query=query, limit=5)
    active_rules = store.search(
        procedural_ns,
        query=query,
        filter={"status": "active"},
        limit=3,
    )

    memory_context = render_memory_context(
        profile_items=profile_items,
        related_events=related_events,
        active_rules=active_rules,
    )

    return {"memory_context": memory_context}
```

### 20.4 call_model 节点

```python
def call_model(state: AgentState):
    system = f"""
You are a helpful technical agent.

Long-term memory:
{state.get("memory_context", "")}
"""

    messages = [{"role": "system", "content": system}] + state["messages"]
    response = llm.invoke(messages)

    return {"messages": state["messages"] + [response]}
```

### 20.5 extract_memory 节点

```python
def extract_memory(state: AgentState):
    recent_messages = state["messages"][-8:]

    candidates = memory_extractor_llm.invoke({
        "messages": recent_messages,
        "schema": MEMORY_SCHEMA,
    })

    return {"candidate_memories": candidates}
```

### 20.6 write_memory 节点

```python
def write_memory(state: AgentState, config, store):
    tenant_id = config["configurable"]["tenant_id"]
    user_id = config["configurable"]["user_id"]

    for candidate in state.get("candidate_memories", []):
        decision = validate_and_route(candidate)

        if decision == "drop":
            continue

        if candidate["memory_type"] == "semantic":
            upsert_semantic(store, tenant_id, user_id, candidate)

        elif candidate["memory_type"] == "episodic":
            upsert_episodic(store, tenant_id, user_id, candidate)

        elif candidate["memory_type"] == "procedural":
            propose_procedural(store, tenant_id, user_id, candidate)

    return {}
```

---

## 21. 示例：完整写入逻辑

### 21.1 Semantic 写入

```python
def upsert_semantic(store, tenant_id, user_id, candidate):
    ns = ("tenant", tenant_id, "user", user_id, "semantic")

    slot = classify_semantic_slot(candidate)

    if slot:
        key = f"profile.{slot}"
        old = store.get(ns, key)

        if old is None:
            store.put(ns, key, enrich(candidate))
            return

        merged = resolve_semantic_conflict(old.value, candidate)
        store.put(ns, key, merged)
        return

    similar = store.search(ns, query=candidate["content"], limit=5)

    if is_duplicate(similar, candidate):
        update_last_seen(store, similar[0])
        return

    key = "note." + content_hash(candidate["content"])
    store.put(ns, key, enrich(candidate))
```

### 21.2 Episodic 写入

```python
def upsert_episodic(store, tenant_id, user_id, candidate):
    ns = ("tenant", tenant_id, "user", user_id, "episodic")

    key = make_event_key(candidate)
    old = store.get(ns, key)

    if old:
        merged = merge_episode(old.value, candidate)
        store.put(ns, key, merged)
        return

    similar = store.search(ns, query=candidate["content"], limit=3)

    if is_same_event(similar, candidate):
        merged = merge_episode(similar[0].value, candidate)
        store.put(ns, similar[0].key, merged)
        return

    store.put(ns, key, enrich(candidate))
```

### 21.3 Procedural 写入

```python
def propose_procedural(store, tenant_id, user_id, candidate):
    ns = ("tenant", tenant_id, "user", user_id, "procedural")

    rule_id = normalize_rule_id(candidate)
    old = store.get(ns, rule_id)

    candidate = enrich(candidate)
    candidate["status"] = "pending_review"

    if old:
        if is_same_rule(old.value, candidate):
            old.value["last_seen"] = now()
            store.put(ns, rule_id, old.value)
            return

        candidate["conflicts_with"] = rule_id
        candidate["status"] = "pending_review"

    store.put(ns, rule_id, candidate)
```

---

## 22. Memory 注入不要做什么

不要：

1. 把所有历史对话都塞进 Prompt。
2. 把低置信 memory 塞进 Prompt。
3. 把 pending_review procedural memory 塞进 Prompt。
4. 不做 namespace 隔离。
5. 不做去重。
6. 不做时间和可信度排序。
7. 把用户临时命令保存成永久偏好。
8. 把工具输出里的敏感数据写入长期记忆。
9. 用 vector search 代替所有 key 查询。
10. 用 Saver 代替 Store。

---

## 23. HelloAgents 项目的 memory 管理方式

前文讨论过 HelloAgents 主分支。

它的 memory 管理方式更接近：

```text
session persistence
conversation history
context engineering
history compression / summarization
tool result management
```

而不是 LangGraph `MongoDBStore` 这种严格意义的长期记忆系统。

### 23.1 它更像短期/会话记忆

项目关注点通常是：

- 保存会话。
- 恢复会话上下文。
- 管理 messages。
- 控制上下文长度。
- 需要时压缩历史。
- 把相关历史喂给 Agent。

这类能力很重要，但它不等价于长期记忆。

### 23.2 和 LangGraph 长期记忆的区别

| 维度 | HelloAgents 类会话管理 | LangGraph Store 长期记忆 |
|---|---|---|
| 主要对象 | session/messages | memory objects |
| 结构化程度 | 偏消息流 | 高度结构化 |
| 是否三类记忆 | 通常没有严格区分 | semantic/episodic/procedural |
| 是否跨 session | 可以，但多是历史恢复 | 明确跨 session |
| 是否做去重冲突 | 通常较弱 | 应该显式实现 |
| 是否有 namespace | 不一定 | 必须设计 |
| 是否向量检索 | 可选 | 常用于长期记忆召回 |

### 23.3 如果把 HelloAgents 升级为长期记忆系统

可以加四层：

1. Memory Extractor：从 session messages 提取候选记忆。
2. Memory Store：把候选写到 MongoDBStore 或其他 store。
3. Memory Retriever：新 session 开始时检索相关记忆。
4. Memory Injector：把记忆渲染进 system prompt。

流程：

```text
HelloAgents session messages
-> summarizer
-> memory extractor
-> semantic/episodic/procedural classifier
-> dedupe/conflict resolver
-> store
-> next session retriever
-> prompt builder
```

---

## 24. 对当前用户场景的建议

你当前关注的是：

- LangGraph production memory。
- MongoDB 后端。
- 长期记忆如何设计。
- 向量检索如何落地。

建议架构：

```text
MongoDBSaver:
  保存 LangGraph checkpoint
  用于 thread/session 恢复

MongoDBStore:
  保存长期记忆对象
  semantic / episodic / procedural 分 namespace

Vector Search:
  如果使用 Atlas Cloud 或 Atlas Local，则 embedding 存 MongoDB 并建 vector index
  如果不用 Atlas，则 MongoDB 保存 memory 文档，向量库单独用 Qdrant/Chroma/pgvector
```

### 24.1 namespace 建议

```python
semantic_ns = ("tenant", tenant_id, "user", user_id, "semantic")
episodic_ns = ("tenant", tenant_id, "user", user_id, "episodic")
procedural_ns = ("tenant", tenant_id, "user", user_id, "procedural")

org_semantic_ns = ("tenant", tenant_id, "org", "semantic")
project_semantic_ns = ("tenant", tenant_id, "project", project_id, "semantic")
```

### 24.2 key 建议

Semantic profile：

```text
profile.language
profile.answer_style
profile.timezone
profile.tech_stack
profile.default_database
profile.risk_preference
```

Episodic：

```text
event.{event_type}.{date}.{hash}
```

Procedural：

```text
rule.{normalized_trigger}.{hash}
```

### 24.3 写入策略建议

| 类型 | 默认写入策略 |
|---|---|
| Semantic slot | 自动 upsert |
| Semantic note | 相似度去重后写入 |
| Episodic | 任务完成后写入 |
| Procedural | 默认 pending_review |

### 24.4 Prompt 注入建议

每次调用模型前：

1. 固定 key 读取 profile。
2. 读取 active procedural rules。
3. 用当前 query 向量检索 episodic/semantic notes。
4. 按 relevance + confidence + recency 排序。
5. 限制 token budget。
6. 渲染成短小结构化 memory context。

---

## 25. 最小可行实现

如果先做 MVP，不要一开始做太复杂。

第一版建议：

1. 只做 semantic profile。
2. 支持 5 到 10 个固定 slot。
3. 支持 key 查询。
4. 每轮或 session end 提取偏好。
5. 暂时不做 vector search。
6. procedural 全部 pending，不自动生效。

第二版：

1. 加 episodic memory。
2. 任务完成后写事件摘要。
3. 用向量检索召回相关事件。

第三版：

1. 加 procedural active/pending。
2. 加管理后台或确认流程。
3. 加冲突审计。
4. 加 memory usage telemetry。

---

## 26. 参考资料

- LangChain / LangGraph memory concepts: https://docs.langchain.com/oss/python/concepts/memory
- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph stores: https://docs.langchain.com/oss/python/langgraph/stores
- MongoDB `$vectorSearch`: https://www.mongodb.com/docs/manual/reference/operator/aggregation/vectorSearch/
- HelloAgents main branch: https://github.com/jjyaoao/HelloAgents/tree/main

