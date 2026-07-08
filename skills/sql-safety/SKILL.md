---
name: sql-safety
description: Use this skill for SQL review, SQL approval, read-only policy, native SQL execution, BigQuery SQL, destructive database operation risk, or query safety.
---

# sql-safety

## Rules

1. Only SELECT/WITH SQL can be executed.
2. Native SQL execution requires explicit user approval.
3. A new pasted SQL statement must be treated as a new request, not as approval of an older pending SQL.
4. Never execute INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, MERGE, CALL, EXECUTE, EXPORT, LOAD, or scripting statements.
5. For generated SQL, explain whether the displayed SQL is exact executable SQL or only an equivalent preview.

## Approval language

Treat explicit approval such as `确认执行`, `同意执行`, `approve`, or `execute` as approval only when the message does not contain a new SQL statement.

