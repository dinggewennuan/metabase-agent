"""Database/table metadata intents, split out of the former 244-line graph node."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import httpx

from metabase_agent.agent.clarify import (
    _database_clarification_suggestions,
    _table_candidate_suggestions,
    _table_clarification_suggestions,
)
from metabase_agent.agent.dry_run import (
    _dry_databases,
    _dry_table_fields,
    _dry_table_query_metadata,
    _dry_tables,
)
from metabase_agent.agent.metadata import (
    _database_items,
    _database_names,
    _field_id,
    _field_names,
    _fields,
    _filter_tables_by_schema,
    _find_database,
    _find_field,
    _find_table,
    _first_datetime_field,
    _first_numeric_field,
    _infer_database_name,
    _is_schema_not_found,
    _match_agent_field_id,
    _rank_table_candidates,
    _table_items,
    _table_schema,
)
from metabase_agent.agent.sql_review import (
    _APPROVAL_MISMATCH_NOTE,
    _sql_review_response,
    _table_sql_preview,
    approved_program_mismatch,
)
from metabase_agent.agent.state import AgentState
from metabase_agent.agent.trace import append_trace as _append_trace
from metabase_agent.query.query_program_builder import (
    _table_aggregation_v1_payload,
    build_table_aggregation_program,
    table_aggregation_dataset_payload,
)
from metabase_agent.tools.metabase.client import MetabaseClient


@dataclass
class _MetadataRequest:
    intent: str
    database_name: str
    schema_name: str | None
    table_name: str
    field_name: str
    date_field_name: str
    aggregation_function: str
    relative_days: int | None
    time_grain: str | None
    raw_question: str

    @classmethod
    def from_state(cls, state: AgentState) -> _MetadataRequest:
        parsed = cast(Mapping[str, Any], state.get("parsed_intent", {}))
        return cls(
            intent=str(parsed.get("intent") or ""),
            database_name=str(parsed.get("database_name") or ""),
            schema_name=cast(str | None, parsed.get("schema_name")),
            table_name=str(parsed.get("table_name") or ""),
            field_name=str(parsed.get("field_name") or ""),
            date_field_name=str(parsed.get("date_field_name") or ""),
            aggregation_function=str(parsed.get("aggregation_function") or ""),
            relative_days=cast(int | None, parsed.get("relative_days")),
            time_grain=cast(str | None, parsed.get("time_grain")),
            raw_question=str(parsed.get("raw_question") or state.get("question") or ""),
        )


def run_database_metadata(state: AgentState, client: MetabaseClient) -> AgentState:
    req = _MetadataRequest.from_state(state)
    dry_run = bool(state.get("dry_run"))

    if dry_run:
        databases = _dry_databases()
        dry_database = _find_database(databases, req.database_name)
        dry_database_name = str(dry_database.get("name")) if dry_database else req.database_name
        tables = _dry_tables(dry_database_name)
        trace = _append_trace(state, {"step": "metadata.dry_run", "database_count": len(databases), "table_count": len(tables)})
    else:
        trace = _append_trace(state, {"step": "metabase.request", "endpoint": "GET /api/database"})
        databases = _database_items(client.list_databases())
        trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": "GET /api/database", "database_names": [database.get("name") for database in databases]})
        tables = []

    if req.intent == "database_count":
        return {
            "answer": f"当前可访问的数据库共有 {len(databases)} 个。",
            "query_plan": {"intent": req.intent},
            "query_result": {"status": "completed", "database_count": len(databases), "databases": [database.get("name") for database in databases]},
            "trace": trace,
        }

    if req.intent == "database_list":
        names = [str(database.get("name")) for database in databases]
        return {
            "answer": "当前可访问的数据库：" + "、".join(names),
            "query_plan": {"intent": req.intent},
            "query_result": {"status": "completed", "database_count": len(names), "databases": names},
            "trace": trace,
        }

    database_names = _database_names(databases)

    if req.intent in {"table_field_list", "table_aggregation"} and not req.database_name and not req.schema_name:
        return {
            "answer": f"请先确认要在哪个数据库中查询 `{req.table_name}`。当前可访问数据库：" + "、".join(database_names),
            "query_plan": {"intent": req.intent, "table_name": req.table_name, "requires_clarification": True, "clarification_type": "database"},
            "query_result": {"status": "requires_clarification", "clarification_type": "database", "table_name": req.table_name, "available_databases": database_names, "suggestions": _database_clarification_suggestions(database_names, req.table_name, req.aggregation_function, req.relative_days, req.time_grain)},
            "trace": _append_trace({"trace": trace}, {"step": "metadata.clarify_database", "status": "requires_clarification", "table_name": req.table_name, "available_databases": database_names}),
        }

    if not req.database_name:
        req.database_name = _infer_database_name(databases, req.schema_name)
    database = _find_database(databases, req.database_name)

    if not dry_run:
        if database is None:
            return _database_not_found(req, databases, trace)
        database_id = int(database["id"])
        trace = _append_trace({"trace": trace}, {"step": "metadata.match_database", "status": "matched", "database_id": database_id, "database_name": database.get("name")})
        tables, trace = _load_tables(client, database_id, req.schema_name if req.database_name else None, trace)

    if database is None:
        return _database_not_found(req, databases, trace)

    database_display_name = str(database.get("name", req.database_name))
    if req.schema_name and any(_table_schema(table) for table in tables):
        tables = _filter_tables_by_schema(tables, req.schema_name)
    table_names = [str(table.get("name")) for table in tables if isinstance(table, dict)]
    if req.schema_name:
        trace = _append_trace({"trace": trace}, {"step": "metadata.filter_schema", "schema_name": req.schema_name, "table_count": len(tables), "table_names": table_names})

    if req.intent == "database_table_list":
        return {
            "answer": f"`{database_display_name}`" + (f" 下 `{req.schema_name}`" if req.schema_name else "") + " 的表：" + "、".join(table_names),
            "query_plan": {"intent": req.intent, "database_name": database_display_name, "schema_name": req.schema_name},
            "query_result": {"status": "completed", "database_name": database_display_name, "schema_name": req.schema_name, "table_count": len(table_names), "tables": table_names},
            "trace": trace,
        }

    if req.intent == "table_field_list":
        table = _find_table(tables, req.table_name)
        if table is None:
            return _table_not_found(req, database_display_name, database_names, tables, table_names, trace)
        return _table_field_list(state, client, req, table, trace)

    if req.intent == "table_aggregation":
        table = _find_table(tables, req.table_name)
        if table is None:
            return _table_not_found(req, database_display_name, database_names, tables, table_names, trace)
        return _table_aggregation(state, client, req, database_display_name, database_id if not dry_run else None, table, trace)

    return {
        "answer": f"`{database_display_name}`" + (f" 下 `{req.schema_name}`" if req.schema_name else " 数据库") + f"共有 {len(table_names)} 个表。",
        "query_plan": {"intent": "database_table_count", "database_name": database_display_name, "schema_name": req.schema_name},
        "query_result": {"status": "completed", "database_name": database_display_name, "schema_name": req.schema_name, "table_count": len(table_names), "tables": table_names},
        "trace": trace,
    }


def _database_not_found(req: _MetadataRequest, databases: list[dict[str, Any]], trace: list[dict[str, Any]]) -> AgentState:
    return {
        "answer": f"没有找到名为 `{req.database_name}` 的数据库。当前可访问数据库：" + "、".join(str(database.get("name")) for database in databases),
        "query_plan": {"intent": req.intent, "database_name": req.database_name, "requires_clarification": True},
        "query_result": {"status": "not_found", "database_name": req.database_name, "available_databases": [database.get("name") for database in databases]},
        "trace": _append_trace({"trace": trace}, {"step": "metadata.match_database", "status": "not_found", "target": req.database_name}),
    }


def _load_tables(client: MetabaseClient, database_id: int, schema_name: str | None, trace: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if schema_name:
        try:
            trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": f"GET /api/database/{database_id}/schema/{schema_name}"})
            schema_metadata = client.get_database_schema(database_id, schema_name)
            tables = _table_items(schema_metadata)
            trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": f"GET /api/database/{database_id}/schema/{schema_name}", "table_count": len(tables), "table_names": [table.get("name") for table in tables if isinstance(table, dict)]})
            return tables, trace
        except httpx.HTTPStatusError as exc:
            if not _is_schema_not_found(exc):
                raise
            trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": f"GET /api/database/{database_id}/schema/{schema_name}", "status": "not_found", "fallback": "GET /api/database/{database_id}/metadata"})
    trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": f"GET /api/database/{database_id}/metadata"})
    database_metadata = client.get_database_metadata(database_id)
    tables = database_metadata.get("tables", []) if isinstance(database_metadata, dict) else []
    trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": f"GET /api/database/{database_id}/metadata", "table_count": len(tables), "table_names": [table.get("name") for table in tables if isinstance(table, dict)]})
    return tables, trace


def _table_not_found(req: _MetadataRequest, database_display_name: str, database_names: list[str], tables: list[dict[str, Any]], table_names: list[str], trace: list[dict[str, Any]]) -> AgentState:
    candidates = _rank_table_candidates(tables, req.table_name)
    candidate_names = [str(candidate.get("name")) for candidate, _score in candidates[:5] if candidate.get("name")]
    return {
        "answer": f"在 `{database_display_name}`" + (f" 下 `{req.schema_name}`" if req.schema_name else "") + f" 没有找到名为 `{req.table_name}` 的表。" + ("可能相关的表：" + "、".join(candidate_names) + "。" if candidate_names else "") + "当前可用表：" + "、".join(table_names) + "。你可以选择其中一个表继续提问，或者更改数据库。",
        "query_plan": {"intent": req.intent, "database_name": database_display_name, "schema_name": req.schema_name, "table_name": req.table_name, "requires_clarification": True, "clarification_type": "table"},
        "query_result": {"status": "not_found", "clarification_type": "table", "database_name": database_display_name, "schema_name": req.schema_name, "table_name": req.table_name, "available_tables": table_names, "candidate_tables": candidate_names, "available_databases": database_names, "suggestions": _table_candidate_suggestions(req.intent, candidates, req.schema_name, req.aggregation_function, req.relative_days, req.time_grain) or _table_clarification_suggestions(req.intent, table_names, req.schema_name, req.aggregation_function, req.relative_days, req.time_grain)},
        "trace": _append_trace({"trace": trace}, {"step": "metadata.match_table", "status": "not_found", "target": req.table_name}),
    }


def _table_field_list(state: AgentState, client: MetabaseClient, req: _MetadataRequest, table: dict[str, Any], trace: list[dict[str, Any]]) -> AgentState:
    if state.get("dry_run"):
        field_metadata = _dry_table_fields()
        fields = cast(list[str], field_metadata["fields"])
    else:
        trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": f"GET /api/table/{int(table['id'])}/query_metadata"})
        table_metadata = client.get_table_query_metadata(int(table["id"]))
        fields = _field_names(table_metadata)
        trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": f"GET /api/table/{int(table['id'])}/query_metadata", "field_count": len(fields)})
    table_display_name = str(table.get("name", req.table_name))
    return {
        "answer": f"`{table_display_name}` 表共有 {len(fields)} 个字段：" + "、".join(fields),
        "query_plan": {"intent": req.intent, "table_name": table_display_name},
        "query_result": {"status": "completed", "table_name": table_display_name, "field_count": len(fields), "fields": fields},
        "trace": trace,
    }


def _table_aggregation(state: AgentState, client: MetabaseClient, req: _MetadataRequest, database_display_name: str, database_id: int | None, table: dict[str, Any], trace: list[dict[str, Any]]) -> AgentState:
    table_display_name = str(table.get("name", req.table_name))
    if state.get("dry_run"):
        fields_payload = _dry_table_query_metadata()
        agent_fields: list[dict[str, Any]] = []
    else:
        trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": f"GET /api/table/{int(table['id'])}/query_metadata"})
        fields_payload = client.get_table_query_metadata(int(table["id"]))
        trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": f"GET /api/table/{int(table['id'])}/query_metadata", "field_count": len(_fields(fields_payload))})
        trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": f"GET /api/agent/v1/table/{int(table['id'])}"})
        try:
            agent_table_payload = client.get_table(int(table["id"]))
            agent_fields = _fields(agent_table_payload)
            trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": f"GET /api/agent/v1/table/{int(table['id'])}", "field_count": len(agent_fields)})
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            agent_fields = []
            trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": f"GET /api/agent/v1/table/{int(table['id'])}", "status": "not_found", "fallback": "GET /api/table/{table_id}/query_metadata"})
    fields = _fields(fields_payload)
    field = None
    if req.aggregation_function != "count":
        field = _find_field(fields, req.field_name) if req.field_name else _first_numeric_field(fields)
        if field is None:
            return {
                "answer": f"没有找到可用于 `{req.aggregation_function}` 的字段，请指定数值字段。",
                "query_plan": {"intent": req.intent, "table_name": table_display_name, "aggregation_function": req.aggregation_function, "requires_clarification": True},
                "query_result": {"status": "not_found", "table_name": table_display_name, "available_fields": [str(item.get("name") or item.get("display_name")) for item in fields]},
                "trace": _append_trace({"trace": trace}, {"step": "metadata.match_field", "status": "not_found", "target": req.field_name}),
            }
    field_id = _field_id(field) if field else None
    date_field = None
    if req.relative_days is not None or req.time_grain:
        date_field = _find_field(fields, req.date_field_name) if req.date_field_name else _first_datetime_field(fields)
        if date_field is None:
            return {
                "answer": "没有找到可用于时间过滤/按天分组的日期字段，请指定时间字段。",
                "query_plan": {"intent": req.intent, "table_name": table_display_name, "aggregation_function": req.aggregation_function, "relative_days": req.relative_days, "time_grain": req.time_grain, "requires_clarification": True},
                "query_result": {"status": "not_found", "table_name": table_display_name, "available_fields": [str(item.get("name") or item.get("display_name")) for item in fields]},
                "trace": _append_trace({"trace": trace}, {"step": "metadata.match_date_field", "status": "not_found", "target": req.date_field_name}),
            }
    date_field_id = _field_id(date_field) if date_field else None
    if date_field is not None and date_field_id is None:
        return {
            "answer": f"时间字段 `{req.date_field_name}` 缺少 Metabase field id，不能过滤或分组。",
            "query_plan": {"intent": req.intent, "table_name": table_display_name, "aggregation_function": req.aggregation_function, "requires_clarification": True},
            "query_result": {"status": "not_found", "table_name": table_display_name, "date_field_name": req.date_field_name},
            "trace": _append_trace({"trace": trace}, {"step": "metadata.match_date_field", "status": "missing_id", "target": req.date_field_name}),
        }
    if req.aggregation_function != "count" and field_id is None:
        return {
            "answer": f"字段 `{req.field_name}` 缺少 Metabase field id，不能聚合。",
            "query_plan": {"intent": req.intent, "table_name": table_display_name, "aggregation_function": req.aggregation_function, "requires_clarification": True},
            "query_result": {"status": "not_found", "table_name": table_display_name, "field_name": req.field_name},
            "trace": _append_trace({"trace": trace}, {"step": "metadata.match_field", "status": "missing_id", "target": req.field_name}),
        }
    program = build_table_aggregation_program(int(table["id"]), req.aggregation_function, field_id, date_field_id=date_field_id, relative_days=req.relative_days, time_grain=req.time_grain)
    agent_field_ids: dict[int, str | None] = {}
    if field_id is not None and field:
        agent_field_ids[field_id] = _match_agent_field_id(agent_fields, field)
    if date_field_id is not None and date_field:
        agent_field_ids[date_field_id] = _match_agent_field_id(agent_fields, date_field)
    mapped_agent_field_ids = {key: value for key, value in agent_field_ids.items() if value}
    if mapped_agent_field_ids:
        program["agent_field_ids"] = mapped_agent_field_ids
    query_plan = {"intent": req.intent, "database_name": database_display_name, "schema_name": req.schema_name, "table_name": table_display_name, "field_name": req.field_name or (str(field.get("name") or field.get("display_name")) if field else None), "date_field_name": req.date_field_name or (str(date_field.get("name") or date_field.get("display_name")) if date_field else None), "aggregation_function": req.aggregation_function, "relative_days": req.relative_days, "time_grain": req.time_grain}
    if not state.get("sql_approved") or approved_program_mismatch(state, program):
        sql = _table_sql_preview(req.schema_name, table_display_name, req.aggregation_function, field, date_field, req.relative_days, req.time_grain)
        response = _sql_review_response(sql, {**program, "preview_sql": sql}, query_plan, trace, preview_only=True)
        if state.get("sql_approved"):
            response["answer"] = _APPROVAL_MISMATCH_NOTE + "\n\n" + str(response["answer"])
        return response
    if state.get("dry_run"):
        if date_field_id is not None and req.time_grain:
            date_field_display_name = str(date_field["display_name"] or date_field["name"]) if date_field else "date"
            query_result = {"status": "completed", "data": {"cols": [{"display_name": date_field_display_name}, {"display_name": req.aggregation_function}], "rows": [["2026-05-11", 3], ["2026-05-12", 5]]}, "row_count": 2}
        else:
            query_result = {"status": "completed", "data": {"cols": [{"display_name": req.aggregation_function}], "rows": [[3]]}, "row_count": 1}
    else:
        query_result, trace = _execute_aggregation(client, database_id, program, trace)
    return {
        "answer": _table_aggregation_answer(req, table_display_name, query_result),
        "query_plan": query_plan,
        "program": program,
        "query_result": query_result,
        "trace": trace,
    }


def _table_aggregation_answer(req: _MetadataRequest, table_display_name: str, query_result: Any) -> str:
    base = f"已对 `{table_display_name}` 执行 `{req.aggregation_function}` 聚合。"
    if not _asks_for_growth_analysis(req.raw_question):
        return base
    rows = _result_rows(query_result)
    if len(rows) < 2:
        return base + " 结果不足两天，暂时无法判断最近两天是否增长。"
    previous, latest = rows[-2], rows[-1]
    previous_value = _last_numeric_cell(previous)
    latest_value = _last_numeric_cell(latest)
    if previous_value is None or latest_value is None:
        return base + " 结果中没有可比较的数值列，暂时无法判断最近两天是否增长。"
    delta = latest_value - previous_value
    pct = f"，变化率 {delta / previous_value * 100:+.1f}%" if previous_value else ""
    trend = "增长" if delta > 0 else "下降" if delta < 0 else "持平"
    previous_label = _row_label(previous)
    latest_label = _row_label(latest)
    answer = f"`{table_display_name}` 最近两天总量{trend}：{previous_label} 为 {previous_value:g}，{latest_label} 为 {latest_value:g}，变化 {delta:+g}{pct}。"
    if _asks_for_breakdown(req.raw_question):
        answer += " 当前查询未指定拆分维度，因此只能判断总量；要分析“哪部分增长”，请指定按哪个字段拆分，例如状态、来源、类型、渠道或具体字段名。"
    return answer


def _asks_for_growth_analysis(question: str) -> bool:
    return any(word in question for word in ("增长", "下降", "变多", "变少", "是否增", "是否涨", "变化", "对比", "环比"))


def _asks_for_breakdown(question: str) -> bool:
    return any(word in question for word in ("哪部分", "哪些部分", "哪里", "哪个", "哪些", "来源", "维度"))


def _result_rows(query_result: Any) -> list[Any]:
    if not isinstance(query_result, dict):
        return []
    data = query_result.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        return []
    return data["rows"]


def _last_numeric_cell(row: Any) -> float | None:
    cells = row if isinstance(row, list) else [row]
    for cell in reversed(cells):
        if isinstance(cell, bool):
            continue
        if isinstance(cell, int | float):
            return float(cell)
    return None


def _row_label(row: Any) -> str:
    if isinstance(row, list) and row:
        return str(row[0])
    return "上一项"


def _execute_aggregation(client: MetabaseClient, database_id: int | None, program: dict[str, Any], trace: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]]]:
    query_program = {"source": program["source"], "operations": program["operations"]}
    trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": "POST /api/agent/v2/query", "program": query_program})
    try:
        return client.query(query_program), trace
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code not in {400, 404}:
            raise
        if database_id is not None:
            payload = table_aggregation_dataset_payload(database_id, program)
            trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": "POST /api/agent/v2/query", "status": "failed", "status_code": exc.response.status_code, "fallback": "POST /api/dataset"})
            trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": "POST /api/dataset", "payload": payload})
            return client.execute_mbql_query(payload), trace
        v1_payload = _table_aggregation_v1_payload(program)
        trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": "POST /api/agent/v2/query", "status": "not_found", "fallback": "POST /api/agent/v1/construct-query"})
        trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": "POST /api/agent/v1/construct-query", "payload": v1_payload})
        constructed = client.construct_query_v1(v1_payload)
        trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": "POST /api/agent/v1/execute"})
        return client.execute_query_v1(constructed), trace
