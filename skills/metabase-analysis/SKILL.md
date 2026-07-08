---
name: metabase-analysis
description: Use this skill for Metabase analytics questions about databases, schemas, tables, fields, metrics, counts, trends, aggregations, query planning, and data口径.
---

# metabase-analysis

## Workflow

1. Identify whether the user asks for metadata, an aggregation, a metric, SQL explanation, or native SQL execution.
2. Prefer metadata tools before execution: list databases, list tables, then list fields.
3. Prefer `run_aggregation` for count/sum/avg/min/max over native SQL when the request can be represented safely.
4. Use `run_sql` only when the request cannot be represented by existing aggregation tools.
5. Always state database, schema/table, time range, aggregation, and unit in the final answer.
6. If table, schema, date field, or metric口径 is ambiguous, ask a clarification question.

## Output expectations

- Answer in concise Chinese.
- Do not invent row counts or field names.
- If a tool returns `not_found`, explain what was missing and suggest the next concrete query.

