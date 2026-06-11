from __future__ import annotations

from typing import Any, Mapping, cast

import httpx
from langgraph.graph import END, START, StateGraph

from metabase_agent.agent.clarify import _database_clarification_suggestions, _table_candidate_suggestions, _table_clarification_suggestions
from metabase_agent.agent.dry_run import _dry_databases, _dry_metric, _dry_result, _dry_search, _dry_table_fields, _dry_table_query_metadata, _dry_tables
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
    _should_execute_generated_sql,
    _sql_explanation_response,
    _sql_review_response,
    _sql_review_with_optional_explanation,
    _table_sql_preview,
)
from metabase_agent.agent.state import AgentState
from metabase_agent.agent.trace import append_trace as _append_trace
from metabase_agent.config.settings import Settings
from metabase_agent.metrics.metric_resolver import choose_metric
from metabase_agent.policy.query_policy import check_program
from metabase_agent.query.bigquery_report_sql import build_monthly_usage_report_sql, extract_native_sql, is_read_only_sql
from metabase_agent.query.query_planner import build_query_plan
from metabase_agent.query.query_program_builder import _table_aggregation_v1_payload, build_program, build_table_aggregation_program
from metabase_agent.semantics.intent_parser import is_safe_rule_intent_override, parse_intent, wants_sql_explanation
from metabase_agent.semantics.llm_intent import parse_intent_with_llm
from metabase_agent.semantics.sql_explainer import explain_sql_with_llm, structural_sql_summary
from metabase_agent.tools.metabase.client import MetabaseClient


def build_graph(settings: Settings):
    client = MetabaseClient(settings.metabase_base_url, settings.metabase_api_key)
    bigquery_database_id = settings.metabase_bigquery_database_id
    report_range_start = settings.agent_report_range_start
    report_range_end_exclusive = settings.agent_report_range_end_exclusive
    report_timezone = settings.agent_report_timezone

    def _summarize_sql(sql: str, dry_run: bool) -> str:
        if not dry_run:
            try:
                return explain_sql_with_llm(sql, settings)
            except Exception as exc:
                print(f"[metabase-agent] sql.explain.llm failed, using structural summary: {exc}")
        return structural_sql_summary(sql)

    def parse_node(state: AgentState) -> AgentState:
        question = str(state.get("question", ""))
        parsed_intent = dict(parse_intent(question))
        trace = _append_trace(state, {"step": "parse.rule", "question": question, "intent": parsed_intent.get("intent"), "database_name": parsed_intent.get("database_name"), "schema_name": parsed_intent.get("schema_name")})
        if not state.get("dry_run"):
            try:
                llm_intent = parse_intent_with_llm(question, settings)
            except Exception as exc:
                llm_intent = None
                trace = _append_trace({"trace": trace}, {"step": "parse.llm", "status": "failed", "error": str(exc)})
            if llm_intent:
                rule_intent = cast(str | None, parsed_intent.get("intent"))
                llm_intent_name = cast(str | None, llm_intent.get("intent"))
                if parsed_intent.get("intent") in {"native_sql_query", "table_aggregation", "bigquery_usage_report_sql", "sql_explanation"}:
                    parsed_intent.update({key: value for key, value in llm_intent.items() if key in parsed_intent and key != "intent" and value})
                    if is_safe_rule_intent_override(rule_intent, llm_intent_name):
                        parsed_intent["intent"] = llm_intent_name
                    else:
                        parsed_intent["intent"] = rule_intent
                else:
                    parsed_intent.update({key: value for key, value in llm_intent.items() if key in parsed_intent or key == "intent"})
                trace = _append_trace({"trace": trace}, {"step": "parse.llm", "status": "completed", "intent": parsed_intent.get("intent"), "database_name": parsed_intent.get("database_name"), "schema_name": parsed_intent.get("schema_name"), "table_name": parsed_intent.get("table_name")})
        return {"parsed_intent": parsed_intent, "trace": trace}

    def route_after_parse(state: AgentState) -> str:
        parsed = cast(Mapping[str, Any], state.get("parsed_intent", {}))
        if parsed.get("intent") == "sql_explanation":
            return "sql_explanation"
        if parsed.get("intent") == "native_sql_query":
            return "native_sql"
        if parsed.get("intent") == "bigquery_usage_report_sql":
            return "bigquery_sql"
        if parsed.get("intent") in {"database_count", "database_list", "database_table_count", "database_table_list", "table_field_list", "table_aggregation"}:
            return "database_metadata"
        return "search"

    def bigquery_sql_node(state: AgentState) -> AgentState:
        question = str(state.get("question", ""))
        sql = build_monthly_usage_report_sql(report_range_start, report_range_end_exclusive, report_timezone)
        trace = _append_trace(state, {"step": "bigquery.sql.generate", "status": "completed", "report": "monthly_usage_web_api"})
        program = {"type": "bigquery_sql", "database_id": bigquery_database_id, "sql": sql, "execute": _should_execute_generated_sql(question)}
        query_plan = {"intent": "bigquery_usage_report_sql", "range_start": report_range_start, "range_end_exclusive": report_range_end_exclusive, "timezone": report_timezone, "channels": ["web", "api"], "execute": program["execute"]}
        if program["execute"]:
            if not state.get("sql_approved"):
                return _sql_review_response(sql, program, query_plan, trace)
            if not is_read_only_sql(sql):
                return {
                    "answer": "已生成 BigQuery 月度 web/api 用量汇总 SQL，但执行前安全校验未通过：只允许执行只读 SELECT/WITH SQL。",
                    "query_plan": {**query_plan, "blocked": True},
                    "program": program,
                    "query_result": {"status": "blocked", "error": "generated SQL failed read-only policy", "sql": sql},
                    "trace": _append_trace({"trace": trace}, {"step": "bigquery.sql.policy", "status": "blocked"}),
                }
            trace = _append_trace({"trace": trace}, {"step": "bigquery.sql.policy", "status": "passed"})
            if state.get("dry_run"):
                query_result = {"status": "completed", "row_count": 0, "data": {"cols": [], "rows": []}, "dry_run": True, "sql": sql}
            else:
                trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": "POST /api/dataset", "database_id": bigquery_database_id, "source": "generated_bigquery_sql"})
                try:
                    query_result = client.execute_native_query(bigquery_database_id, sql)
                except httpx.HTTPStatusError as exc:
                    return {
                        "answer": f"已生成 BigQuery 月度 web/api 用量汇总 SQL，但执行失败：{exc.response.status_code} {exc.response.reason_phrase}",
                        "query_plan": query_plan,
                        "program": program,
                        "query_result": {"status": "failed", "error": str(exc), "response_text": client.response_text(exc), "sql": sql},
                        "trace": _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": "POST /api/dataset", "status": "failed", "status_code": exc.response.status_code}),
                    }
            return {
                "answer": f"已生成并执行 BigQuery 月度 web/api 用量汇总 SQL，返回 {query_result.get('row_count', 0) if isinstance(query_result, dict) else 0} 行。图片类单位为 count；可统计时长的项目单位为 seconds。",
                "query_plan": query_plan,
                "program": program,
                "query_result": query_result,
                "trace": trace,
            }
        return {
            "answer": "已生成 BigQuery 月度 web/api 用量汇总 SQL。图片类单位为 count；可统计时长的项目单位为 seconds。",
            "query_plan": query_plan,
            "program": program,
            "query_result": {"status": "completed", "unit_policy": {"seconds": "duration_seconds 可用", "count": "只有 generated_count 有意义，例如图片生成"}, "sql": sql},
            "trace": trace,
        }

    def sql_explanation_node(state: AgentState) -> AgentState:
        question = str(state.get("question", ""))
        sql = extract_native_sql(question) or question
        answer = _summarize_sql(sql, bool(state.get("dry_run")))
        return _sql_explanation_response(sql, answer, state.get("trace", []))

    def native_sql_node(state: AgentState) -> AgentState:
        question = str(state.get("question", ""))
        sql = extract_native_sql(question) or ""
        trace = _append_trace(state, {"step": "native_sql.extract", "status": "completed" if sql else "not_found"})
        if not sql:
            return {
                "answer": "没有识别到可执行 SQL。请直接粘贴 SELECT 或 WITH 查询。",
                "query_plan": {"intent": "native_sql_query", "requires_clarification": True},
                "program": {},
                "query_result": {"status": "not_found", "error": "missing SQL"},
                "trace": trace,
            }
        if not is_read_only_sql(sql):
            return {
                "answer": "已拦截：只允许执行只读 SELECT/WITH SQL。",
                "query_plan": {"intent": "native_sql_query", "blocked": True},
                "program": {"type": "native_sql", "database_id": bigquery_database_id, "sql": sql},
                "query_result": {"status": "blocked", "error": "only read-only SELECT/WITH SQL is allowed"},
                "trace": _append_trace({"trace": trace}, {"step": "native_sql.policy", "status": "blocked"}),
            }
        program = {"type": "native_sql", "database_id": bigquery_database_id, "sql": sql}
        if not state.get("sql_approved"):
            query_plan = {"intent": "native_sql_query", "database_name": "BigQuery-GA", "database_id": bigquery_database_id}
            analysis = _summarize_sql(sql, bool(state.get("dry_run"))) if wants_sql_explanation(question) else None
            return _sql_review_with_optional_explanation(sql, program, query_plan, trace, analysis)
        if state.get("dry_run"):
            query_result = {"status": "completed", "row_count": 0, "data": {"cols": [], "rows": []}, "dry_run": True, "sql": sql}
        else:
            trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": "POST /api/dataset", "database_id": bigquery_database_id})
            try:
                query_result = client.execute_native_query(bigquery_database_id, sql)
            except httpx.HTTPStatusError as exc:
                return {
                    "answer": f"SQL 执行失败：{exc.response.status_code} {exc.response.reason_phrase}",
                    "query_plan": {"intent": "native_sql_query", "database_name": "BigQuery-GA", "database_id": bigquery_database_id},
                    "program": program,
                    "query_result": {"status": "failed", "error": str(exc), "response_text": client.response_text(exc), "sql": sql},
                    "trace": _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": "POST /api/dataset", "status": "failed", "status_code": exc.response.status_code}),
                }
        return {
            "answer": f"已执行只读 BigQuery SQL，返回 {query_result.get('row_count', 0) if isinstance(query_result, dict) else 0} 行。",
            "query_plan": {"intent": "native_sql_query", "database_name": "BigQuery-GA", "database_id": bigquery_database_id},
            "program": program,
            "query_result": query_result,
            "trace": trace,
        }

    def database_metadata_node(state: AgentState) -> AgentState:
        parsed = cast(Mapping[str, Any], state.get("parsed_intent", {}))
        intent = str(parsed.get("intent") or "")
        database_name = str(parsed.get("database_name") or "")
        schema_name = cast(str | None, parsed.get("schema_name"))
        table_name = str(parsed.get("table_name") or "")
        field_name = str(parsed.get("field_name") or "")
        date_field_name = str(parsed.get("date_field_name") or "")
        aggregation_function = str(parsed.get("aggregation_function") or "")
        relative_days = cast(int | None, parsed.get("relative_days"))
        time_grain = cast(str | None, parsed.get("time_grain"))
        if state.get("dry_run"):
            databases = _dry_databases()
            dry_database = _find_database(databases, database_name)
            dry_database_name = str(dry_database.get("name")) if dry_database else database_name
            tables = _dry_tables(dry_database_name)
            trace = _append_trace(state, {"step": "metadata.dry_run", "database_count": len(databases), "table_count": len(tables)})
        else:
            trace = _append_trace(state, {"step": "metabase.request", "endpoint": "GET /api/database"})
            databases = _database_items(client.list_databases())
            trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": "GET /api/database", "database_names": [database.get("name") for database in databases]})
            tables: list[dict[str, Any]] = []

        if intent == "database_count":
            return {
                "answer": f"当前可访问的数据库共有 {len(databases)} 个。",
                "query_plan": {"intent": intent},
                "query_result": {"status": "completed", "database_count": len(databases), "databases": [database.get("name") for database in databases]},
                "trace": trace,
            }

        if intent == "database_list":
            names = [str(database.get("name")) for database in databases]
            return {
                "answer": "当前可访问的数据库：" + "、".join(names),
                "query_plan": {"intent": intent},
                "query_result": {"status": "completed", "database_count": len(names), "databases": names},
                "trace": trace,
            }

        database_names = _database_names(databases)

        if intent in {"table_field_list", "table_aggregation"} and not database_name and not schema_name:
            return {
                "answer": f"请先确认要在哪个数据库中查询 `{table_name}`。当前可访问数据库：" + "、".join(database_names),
                "query_plan": {"intent": intent, "table_name": table_name, "requires_clarification": True, "clarification_type": "database"},
                "query_result": {"status": "requires_clarification", "clarification_type": "database", "table_name": table_name, "available_databases": database_names, "suggestions": _database_clarification_suggestions(database_names, table_name, aggregation_function, relative_days, time_grain)},
                "trace": _append_trace({"trace": trace}, {"step": "metadata.clarify_database", "status": "requires_clarification", "table_name": table_name, "available_databases": database_names}),
            }

        if state.get("dry_run"):
            if not database_name:
                database_name = _infer_database_name(databases, schema_name)
            database = _find_database(databases, database_name)
        else:
            if not database_name:
                database_name = _infer_database_name(databases, schema_name)
            database = _find_database(databases, database_name)
            if database is None:
                return {
                    "answer": f"没有找到名为 `{database_name}` 的数据库。当前可访问数据库：" + "、".join(str(database.get("name")) for database in databases),
                    "query_plan": {"intent": intent, "database_name": database_name, "requires_clarification": True},
                    "query_result": {"status": "not_found", "database_name": database_name, "available_databases": [database.get("name") for database in databases]},
                    "trace": _append_trace({"trace": trace}, {"step": "metadata.match_database", "status": "not_found", "target": database_name}),
                }
            database_id = int(database["id"])
            trace = _append_trace({"trace": trace}, {"step": "metadata.match_database", "status": "matched", "database_id": database_id, "database_name": database.get("name")})
            if schema_name and database_name:
                try:
                    trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": f"GET /api/database/{database_id}/schema/{schema_name}"})
                    schema_metadata = client.get_database_schema(database_id, schema_name)
                    tables = _table_items(schema_metadata)
                    trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": f"GET /api/database/{database_id}/schema/{schema_name}", "table_count": len(tables), "table_names": [table.get("name") for table in tables if isinstance(table, dict)]})
                except httpx.HTTPStatusError as exc:
                    if not _is_schema_not_found(exc):
                        raise
                    trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": f"GET /api/database/{database_id}/schema/{schema_name}", "status": "not_found", "fallback": "GET /api/database/{database_id}/metadata"})
                    trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": f"GET /api/database/{database_id}/metadata"})
                    database_metadata = client.get_database_metadata(database_id)
                    tables = database_metadata.get("tables", []) if isinstance(database_metadata, dict) else []
                    trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": f"GET /api/database/{database_id}/metadata", "table_count": len(tables), "table_names": [table.get("name") for table in tables if isinstance(table, dict)]})
            else:
                trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": f"GET /api/database/{database_id}/metadata"})
                database_metadata = client.get_database_metadata(database_id)
                tables = database_metadata.get("tables", []) if isinstance(database_metadata, dict) else []
                trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": f"GET /api/database/{database_id}/metadata", "table_count": len(tables), "table_names": [table.get("name") for table in tables if isinstance(table, dict)]})

        if database is None:
            return {
                "answer": f"没有找到名为 `{database_name}` 的数据库。当前可访问数据库：" + "、".join(str(database.get("name")) for database in databases),
                "query_plan": {"intent": intent, "database_name": database_name, "requires_clarification": True},
                "query_result": {"status": "not_found", "database_name": database_name, "available_databases": [database.get("name") for database in databases]},
                "trace": _append_trace({"trace": trace}, {"step": "metadata.match_database", "status": "not_found", "target": database_name}),
            }

        database_display_name = str(database.get("name", database_name))
        if schema_name and any(_table_schema(table) for table in tables):
            tables = _filter_tables_by_schema(tables, schema_name)
        table_names = [str(table.get("name")) for table in tables if isinstance(table, dict)]
        if schema_name:
            trace = _append_trace({"trace": trace}, {"step": "metadata.filter_schema", "schema_name": schema_name, "table_count": len(tables), "table_names": table_names})

        if intent == "database_table_list":
            return {
                "answer": f"`{database_display_name}`" + (f" 下 `{schema_name}`" if schema_name else "") + " 的表：" + "、".join(table_names),
                "query_plan": {"intent": intent, "database_name": database_display_name, "schema_name": schema_name},
                "query_result": {"status": "completed", "database_name": database_display_name, "schema_name": schema_name, "table_count": len(table_names), "tables": table_names},
                "trace": trace,
            }

        if intent == "table_field_list":
            table = _find_table(tables, table_name)
            if table is None:
                candidates = _rank_table_candidates(tables, table_name)
                candidate_names = [str(candidate.get("name")) for candidate, _score in candidates[:5] if candidate.get("name")]
                return {
                    "answer": f"在 `{database_display_name}`" + (f" 下 `{schema_name}`" if schema_name else "") + f" 没有找到名为 `{table_name}` 的表。" + ("可能相关的表：" + "、".join(candidate_names) + "。" if candidate_names else "") + "当前可用表：" + "、".join(table_names) + "。你可以选择其中一个表继续提问，或者更改数据库。",
                    "query_plan": {"intent": intent, "database_name": database_display_name, "schema_name": schema_name, "table_name": table_name, "requires_clarification": True, "clarification_type": "table"},
                    "query_result": {"status": "not_found", "clarification_type": "table", "database_name": database_display_name, "schema_name": schema_name, "table_name": table_name, "available_tables": table_names, "candidate_tables": candidate_names, "available_databases": database_names, "suggestions": _table_candidate_suggestions(intent, candidates, schema_name, aggregation_function, relative_days, time_grain) or _table_clarification_suggestions(intent, table_names, schema_name, aggregation_function, relative_days, time_grain)},
                    "trace": _append_trace({"trace": trace}, {"step": "metadata.match_table", "status": "not_found", "target": table_name}),
                }
            if state.get("dry_run"):
                field_metadata = _dry_table_fields()
                fields = cast(list[str], field_metadata["fields"])
            else:
                trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": f"GET /api/table/{int(table['id'])}/query_metadata"})
                table_metadata = client.get_table_query_metadata(int(table["id"]))
                fields = _field_names(table_metadata)
                trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": f"GET /api/table/{int(table['id'])}/query_metadata", "field_count": len(fields)})
            table_display_name = str(table.get("name", table_name))
            return {
                "answer": f"`{table_display_name}` 表共有 {len(fields)} 个字段：" + "、".join(fields),
                "query_plan": {"intent": intent, "table_name": table_display_name},
                "query_result": {"status": "completed", "table_name": table_display_name, "field_count": len(fields), "fields": fields},
                "trace": trace,
            }

        if intent == "table_aggregation":
            table = _find_table(tables, table_name)
            if table is None:
                candidates = _rank_table_candidates(tables, table_name)
                candidate_names = [str(candidate.get("name")) for candidate, _score in candidates[:5] if candidate.get("name")]
                return {
                    "answer": f"在 `{database_display_name}`" + (f" 下 `{schema_name}`" if schema_name else "") + f" 没有找到名为 `{table_name}` 的表。" + ("可能相关的表：" + "、".join(candidate_names) + "。" if candidate_names else "") + "当前可用表：" + "、".join(table_names) + "。你可以选择其中一个表继续提问，或者更改数据库。",
                    "query_plan": {"intent": intent, "database_name": database_display_name, "schema_name": schema_name, "table_name": table_name, "requires_clarification": True, "clarification_type": "table"},
                    "query_result": {"status": "not_found", "clarification_type": "table", "database_name": database_display_name, "schema_name": schema_name, "table_name": table_name, "available_tables": table_names, "candidate_tables": candidate_names, "available_databases": database_names, "suggestions": _table_candidate_suggestions(intent, candidates, schema_name, aggregation_function, relative_days, time_grain) or _table_clarification_suggestions(intent, table_names, schema_name, aggregation_function, relative_days, time_grain)},
                    "trace": _append_trace({"trace": trace}, {"step": "metadata.match_table", "status": "not_found", "target": table_name}),
                }
            table_display_name = str(table.get("name", table_name))
            if state.get("dry_run"):
                fields_payload = _dry_table_query_metadata()
                agent_fields: list[dict[str, Any]] = []
            else:
                trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": f"GET /api/table/{int(table['id'])}/query_metadata"})
                fields_payload = client.get_table_query_metadata(int(table["id"]))
                trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": f"GET /api/table/{int(table['id'])}/query_metadata", "field_count": len(_fields(fields_payload))})
                trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": f"GET /api/agent/v1/table/{int(table['id'])}"})
                agent_table_payload = client.get_table(int(table["id"]))
                agent_fields = _fields(agent_table_payload)
                trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": f"GET /api/agent/v1/table/{int(table['id'])}", "field_count": len(agent_fields)})
            fields = _fields(fields_payload)
            field = None
            if aggregation_function != "count":
                field = _find_field(fields, field_name) if field_name else _first_numeric_field(fields)
                if field is None:
                    return {
                        "answer": f"没有找到可用于 `{aggregation_function}` 的字段，请指定数值字段。",
                        "query_plan": {"intent": intent, "table_name": table_display_name, "aggregation_function": aggregation_function, "requires_clarification": True},
                        "query_result": {"status": "not_found", "table_name": table_display_name, "available_fields": [str(item.get("name") or item.get("display_name")) for item in fields]},
                        "trace": _append_trace({"trace": trace}, {"step": "metadata.match_field", "status": "not_found", "target": field_name}),
                    }
            field_id = _field_id(field) if field else None
            date_field = None
            if relative_days is not None or time_grain:
                date_field = _find_field(fields, date_field_name) if date_field_name else _first_datetime_field(fields)
                if date_field is None:
                    return {
                        "answer": "没有找到可用于时间过滤/按天分组的日期字段，请指定时间字段。",
                        "query_plan": {"intent": intent, "table_name": table_display_name, "aggregation_function": aggregation_function, "relative_days": relative_days, "time_grain": time_grain, "requires_clarification": True},
                        "query_result": {"status": "not_found", "table_name": table_display_name, "available_fields": [str(item.get("name") or item.get("display_name")) for item in fields]},
                        "trace": _append_trace({"trace": trace}, {"step": "metadata.match_date_field", "status": "not_found", "target": date_field_name}),
                    }
            date_field_id = _field_id(date_field) if date_field else None
            if date_field is not None and date_field_id is None:
                return {
                    "answer": f"时间字段 `{date_field_name}` 缺少 Metabase field id，不能过滤或分组。",
                    "query_plan": {"intent": intent, "table_name": table_display_name, "aggregation_function": aggregation_function, "requires_clarification": True},
                    "query_result": {"status": "not_found", "table_name": table_display_name, "date_field_name": date_field_name},
                    "trace": _append_trace({"trace": trace}, {"step": "metadata.match_date_field", "status": "missing_id", "target": date_field_name}),
                }
            if aggregation_function != "count" and field_id is None:
                return {
                    "answer": f"字段 `{field_name}` 缺少 Metabase field id，不能聚合。",
                    "query_plan": {"intent": intent, "table_name": table_display_name, "aggregation_function": aggregation_function, "requires_clarification": True},
                    "query_result": {"status": "not_found", "table_name": table_display_name, "field_name": field_name},
                    "trace": _append_trace({"trace": trace}, {"step": "metadata.match_field", "status": "missing_id", "target": field_name}),
                }
            program = build_table_aggregation_program(int(table["id"]), aggregation_function, field_id, date_field_id=date_field_id, relative_days=relative_days, time_grain=time_grain)
            agent_field_ids: dict[int, str | None] = {}
            if field_id is not None and field:
                agent_field_ids[field_id] = _match_agent_field_id(agent_fields, field)
            if date_field_id is not None and date_field:
                agent_field_ids[date_field_id] = _match_agent_field_id(agent_fields, date_field)
            mapped_agent_field_ids = {key: value for key, value in agent_field_ids.items() if value}
            if mapped_agent_field_ids:
                program["agent_field_ids"] = mapped_agent_field_ids
            query_plan = {"intent": intent, "database_name": database_display_name, "schema_name": schema_name, "table_name": table_display_name, "field_name": field_name or (str(field.get("name") or field.get("display_name")) if field else None), "date_field_name": date_field_name or (str(date_field.get("name") or date_field.get("display_name")) if date_field else None), "aggregation_function": aggregation_function, "relative_days": relative_days, "time_grain": time_grain}
            if not state.get("sql_approved"):
                sql = _table_sql_preview(schema_name, table_display_name, aggregation_function, field, date_field, relative_days, time_grain)
                return _sql_review_response(sql, {**program, "preview_sql": sql}, query_plan, trace, preview_only=True)
            if state.get("dry_run"):
                if date_field_id is not None and time_grain:
                    date_field_display_name = str(date_field["display_name"] or date_field["name"]) if date_field else "date"
                    query_result = {"status": "completed", "data": {"cols": [{"display_name": date_field_display_name}, {"display_name": aggregation_function}], "rows": [["2026-05-11", 3], ["2026-05-12", 5]]}, "row_count": 2}
                else:
                    query_result = {"status": "completed", "data": {"cols": [{"display_name": aggregation_function}], "rows": [[3]]}, "row_count": 1}
            else:
                query_program = {"source": program["source"], "operations": program["operations"]}
                trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": "POST /api/agent/v2/query", "program": query_program})
                try:
                    query_result = client.query(query_program)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code != 404:
                        raise
                    v1_payload = _table_aggregation_v1_payload(program)
                    trace = _append_trace({"trace": trace}, {"step": "metabase.response", "endpoint": "POST /api/agent/v2/query", "status": "not_found", "fallback": "POST /api/agent/v1/construct-query"})
                    trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": "POST /api/agent/v1/construct-query", "payload": v1_payload})
                    constructed = client.construct_query_v1(v1_payload)
                    trace = _append_trace({"trace": trace}, {"step": "metabase.request", "endpoint": "POST /api/agent/v1/execute"})
                    query_result = client.execute_query_v1(constructed)
            return {
                "answer": f"已对 `{table_display_name}` 执行 `{aggregation_function}` 聚合。",
                "query_plan": query_plan,
                "program": program,
                "query_result": query_result,
                "trace": trace,
            }

        return {
            "answer": f"`{database_display_name}`" + (f" 下 `{schema_name}`" if schema_name else " 数据库") + f"共有 {len(table_names)} 个表。",
            "query_plan": {"intent": "database_table_count", "database_name": database_display_name, "schema_name": schema_name},
            "query_result": {"status": "completed", "database_name": database_display_name, "schema_name": schema_name, "table_count": len(table_names), "tables": table_names},
            "trace": trace,
        }

    def search_node(state: AgentState) -> AgentState:
        parsed = cast(Mapping[str, Any], state.get("parsed_intent", {}))
        question = str(state.get("question", ""))
        business_terms = cast(list[str], parsed.get("business_terms", []))
        queries = [question, *business_terms]
        result = _dry_search() if state.get("dry_run") else client.search(queries)
        return {"search_result": result, "selected_entity": choose_metric(result, business_terms)}

    def inspect_node(state: AgentState) -> AgentState:
        entity = state.get("selected_entity")
        if not entity:
            return {"inspected_entity": None}
        if state.get("dry_run"):
            return {"inspected_entity": _dry_metric()}
        if entity["type"] == "metric":
            return {"inspected_entity": client.get_metric(int(entity["id"]))}
        return {"inspected_entity": client.get_table(int(entity["id"]))}

    def plan_node(state: AgentState) -> AgentState:
        return {"query_plan": dict(build_query_plan(cast(dict[str, Any], state.get("parsed_intent", {})), state.get("selected_entity")))}

    def build_program_node(state: AgentState) -> AgentState:
        query_plan = cast(dict[str, Any], state.get("query_plan", {}))
        if query_plan.get("requires_clarification"):
            return {"program": {}}
        return {"program": build_program(query_plan, state.get("inspected_entity"))}

    def policy_node(state: AgentState) -> AgentState:
        if not state.get("program"):
            return {"policy_result": {"allowed": False, "reason": "missing query program"}}
        return {"policy_result": dict(check_program(cast(dict[str, Any], state.get("program", {}))))}

    def execute_node(state: AgentState) -> AgentState:
        policy_result = cast(dict[str, Any], state.get("policy_result", {}))
        if not policy_result.get("allowed"):
            return {"query_result": {"status": "blocked", "error": policy_result.get("reason")}}
        if state.get("dry_run"):
            return {"query_result": _dry_result()}
        if not state.get("sql_approved"):
            program = cast(dict[str, Any], state.get("program", {}))
            query_plan = cast(dict[str, Any], state.get("query_plan", {}))
            prompt = "请先 review 这条 Metabase 结构化查询，确认无误后点击授权确认执行，或拒绝本次执行。"
            return {
                "query_plan": {**query_plan, "requires_approval": True},
                "program": {**program, "requires_approval": True},
                "query_result": {"status": "requires_approval", "program": program, "approval_prompt": prompt},
            }
        return {"query_result": client.query(cast(dict[str, Any], state.get("program", {})))}

    def answer_node(state: AgentState) -> AgentState:
        query_plan = cast(dict[str, Any], state.get("query_plan", {}))
        if query_plan.get("requires_clarification"):
            return {"answer": str(query_plan.get("clarification_question", "需要补充查询条件。"))}
        result = cast(dict[str, Any], state.get("query_result", {}))
        if result.get("status") == "requires_approval":
            return {"answer": str(result.get("approval_prompt") or "请先 review 并授权执行该查询。")}
        if result.get("status") != "completed":
            return {"answer": f"查询未完成：{result.get('error', result.get('status'))}"}
        return {
            "answer": f"已基于 Metabase Metric `{query_plan.get('source_name', '')}` 完成查询，返回 {result.get('row_count', 0)} 行。"
        }

    graph = StateGraph(AgentState)
    graph.add_node("parse", parse_node)
    graph.add_node("sql_explanation", sql_explanation_node)
    graph.add_node("native_sql", native_sql_node)
    graph.add_node("bigquery_sql", bigquery_sql_node)
    graph.add_node("database_metadata", database_metadata_node)
    graph.add_node("search", search_node)
    graph.add_node("inspect", inspect_node)
    graph.add_node("plan", plan_node)
    graph.add_node("build_program", build_program_node)
    graph.add_node("policy", policy_node)
    graph.add_node("execute", execute_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "parse")
    graph.add_conditional_edges("parse", route_after_parse, {"sql_explanation": "sql_explanation", "native_sql": "native_sql", "bigquery_sql": "bigquery_sql", "database_metadata": "database_metadata", "search": "search"})
    graph.add_edge("sql_explanation", END)
    graph.add_edge("native_sql", END)
    graph.add_edge("bigquery_sql", END)
    graph.add_edge("database_metadata", END)
    graph.add_edge("search", "inspect")
    graph.add_edge("inspect", "plan")
    graph.add_edge("plan", "build_program")
    graph.add_edge("build_program", "policy")
    graph.add_edge("policy", "execute")
    graph.add_edge("execute", "answer")
    graph.add_edge("answer", END)
    return graph.compile()
