from __future__ import annotations

import logging
from typing import Any, Mapping, cast

import httpx
from langgraph.graph import END, START, StateGraph

from metabase_agent.agent.dry_run import (
    _dry_metric,
    _dry_result,
    _dry_search,
)
from metabase_agent.agent.metadata_flow import run_database_metadata
from metabase_agent.agent.sql_review import (
    _APPROVAL_MISMATCH_NOTE,
    _should_execute_generated_sql,
    _sql_explanation_response,
    _sql_review_response,
    _sql_review_with_optional_explanation,
    approved_program_mismatch,
)
from metabase_agent.agent.state import AgentState
from metabase_agent.agent.trace import append_trace as _append_trace
from metabase_agent.config.settings import Settings
from metabase_agent.metrics.metric_resolver import choose_metric
from metabase_agent.policy.query_policy import check_program
from metabase_agent.query.bigquery_report_sql import (
    build_monthly_usage_report_sql,
    extract_native_sql,
    is_read_only_sql,
)
from metabase_agent.query.query_planner import build_query_plan
from metabase_agent.query.query_program_builder import (
    build_program,
)
from metabase_agent.semantics.intent_parser import (
    is_safe_rule_intent_override,
    parse_intent,
    wants_sql_explanation,
)
from metabase_agent.semantics.llm_intent import parse_intent_with_llm
from metabase_agent.semantics.sql_explainer import (
    explain_sql_with_llm,
    structural_sql_summary,
)
from metabase_agent.tools.metabase.client import MetabaseClient

_logger = logging.getLogger("metabase_agent")


def _metric_answer(intent: str, source_name: str, result: dict[str, Any]) -> str:
    raw_data = result.get("data")
    data = raw_data if isinstance(raw_data, dict) else {}
    raw_rows = data.get("rows")
    rows = cast(list[Any], raw_rows) if isinstance(raw_rows, list) else []
    row_count = result.get("row_count", len(rows))
    if intent == "metric_value" and rows:
        latest = rows[-1]
        if isinstance(latest, list) and latest:
            label = f"（{latest[0]}）" if len(latest) > 1 else ""
            return f"`{source_name}` 当前值为 {latest[-1]}{label}。"
    if intent == "comparison" and len(rows) >= 2:
        previous, current = rows[-2], rows[-1]
        if isinstance(previous, list) and isinstance(current, list) and isinstance(previous[-1], (int, float)) and isinstance(current[-1], (int, float)):
            delta = current[-1] - previous[-1]
            pct = f"，环比 {delta / previous[-1] * 100:+.1f}%" if previous[-1] else ""
            return f"`{source_name}` 对比：{previous[0]} 为 {previous[-1]}，{current[0]} 为 {current[-1]}，变化 {delta:+g}{pct}。"
    if intent == "metric_trend" and rows:
        first, last = rows[0], rows[-1]
        if isinstance(first, list) and isinstance(last, list):
            return f"`{source_name}` 趋势共 {len(rows)} 个数据点：{first[0]} 为 {first[-1]}，最新 {last[0]} 为 {last[-1]}。"
    if intent == "detail_lookup":
        return f"已返回 `{source_name}` 的 {row_count} 行明细。"
    return f"已基于 Metabase Metric `{source_name}` 完成查询，返回 {row_count} 行。"


def build_graph(settings: Settings, checkpointer: Any | None = None):
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
                _logger.warning("sql.explain.llm failed, using structural summary: %s", exc)
        return structural_sql_summary(sql)

    def parse_node(state: AgentState) -> AgentState:
        question = str(state.get("question", ""))
        parsed_intent = dict(parse_intent(question))
        trace = _append_trace(state, {"step": "parse.rule", "question": question, "intent": parsed_intent.get("intent"), "database_name": parsed_intent.get("database_name"), "schema_name": parsed_intent.get("schema_name")})
        if str(state.get("memory_context", "")).strip():
            trace = _append_trace({"trace": trace}, {"step": "memory.context", "status": "loaded"})
        if str(state.get("skills_context", "")).strip():
            trace = _append_trace({"trace": trace}, {"step": "skills.context", "status": "loaded"})
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
            if approved_program_mismatch(state, program):
                response = _sql_review_response(sql, program, query_plan, trace)
                response["answer"] = _APPROVAL_MISMATCH_NOTE + "\n\n" + str(response["answer"])
                return response
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
                "answer": f"已生成并执行 BigQuery 月度 web/api 用量汇总 SQL（统计区间 {report_range_start} ~ {report_range_end_exclusive}（不含），时区 {report_timezone}；如需其他区间请调整 AGENT_REPORT_RANGE_* 配置），返回 {query_result.get('row_count', 0) if isinstance(query_result, dict) else 0} 行。图片类单位为 count；可统计时长的项目单位为 seconds。",
                "query_plan": query_plan,
                "program": program,
                "query_result": query_result,
                "trace": trace,
            }
        return {
            "answer": f"已生成 BigQuery 月度 web/api 用量汇总 SQL（统计区间 {report_range_start} ~ {report_range_end_exclusive}（不含），时区 {report_timezone}；如需其他区间请调整 AGENT_REPORT_RANGE_* 配置）。图片类单位为 count；可统计时长的项目单位为 seconds。",
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
        if approved_program_mismatch(state, program):
            query_plan = {"intent": "native_sql_query", "database_name": "BigQuery-GA", "database_id": bigquery_database_id}
            response = _sql_review_response(sql, program, query_plan, trace)
            response["answer"] = _APPROVAL_MISMATCH_NOTE + "\n\n" + str(response["answer"])
            return response
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
        return run_database_metadata(state, client)

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
        program = cast(dict[str, Any], state.get("program", {}))
        if not state.get("sql_approved") or approved_program_mismatch(state, program):
            query_plan = cast(dict[str, Any], state.get("query_plan", {}))
            prompt = "请先 review 这条 Metabase 结构化查询，确认无误后点击授权确认执行，或拒绝本次执行。"
            if state.get("sql_approved"):
                prompt = _APPROVAL_MISMATCH_NOTE + " " + prompt
            return {
                "query_plan": {**query_plan, "requires_approval": True},
                "program": {**program, "requires_approval": True},
                "query_result": {"status": "requires_approval", "program": program, "approval_prompt": prompt},
            }
        return {"query_result": client.query(program)}

    def answer_node(state: AgentState) -> AgentState:
        query_plan = cast(dict[str, Any], state.get("query_plan", {}))
        if query_plan.get("requires_clarification"):
            return {"answer": str(query_plan.get("clarification_question", "需要补充查询条件。"))}
        result = cast(dict[str, Any], state.get("query_result", {}))
        if result.get("status") == "requires_approval":
            return {"answer": str(result.get("approval_prompt") or "请先 review 并授权执行该查询。")}
        if result.get("status") != "completed":
            return {"answer": f"查询未完成：{result.get('error', result.get('status'))}"}
        return {"answer": _metric_answer(str(query_plan.get("intent") or ""), str(query_plan.get("source_name", "")), result)}

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
    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
