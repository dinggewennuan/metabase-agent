"""SQL review/approval responses, the equivalent-preview SQL builder, and execution-intent detection."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping

from metabase_agent.agent.state import AgentState
from metabase_agent.agent.trace import append_trace

# Presentation-only keys that differ between the reviewed program and the
# rebuilt one; everything else must match or the approval does not apply.
_FINGERPRINT_IGNORED_KEYS = {"requires_approval", "execute", "preview_sql"}


def program_fingerprint(program: Mapping[str, Any] | None) -> str | None:
    """Stable hash of what a program will actually execute.

    Stored alongside a pending approval and re-checked before execution, so an
    approval only ever authorizes the exact query the user reviewed.
    """
    if not isinstance(program, Mapping) or not program:
        return None
    payload = {key: value for key, value in program.items() if key not in _FINGERPRINT_IGNORED_KEYS}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def approved_program_mismatch(state: Mapping[str, Any], program: Mapping[str, Any]) -> bool:
    expected = state.get("approved_program_hash")
    return bool(expected) and program_fingerprint(program) != expected


_APPROVAL_MISMATCH_NOTE = "本次生成的查询与之前审批的内容不一致，已重新进入审批流程。"


def _should_execute_generated_sql(question: str) -> bool:
    lowered = question.lower()
    explicit_execution = any(
        phrase in question or phrase in lowered
        for phrase in (
            "执行sql",
            "执行 sql",
            "执行语句",
            "执行查询",
            "主动执行",
            "自己执行",
            "跑数据",
            "最终数据",
            "数据展示",
            "查询后的数据",
            "获取最终",
        )
    )
    if explicit_execution:
        return True
    asks_for_sql_only = any(phrase in question or phrase in lowered for phrase in ("给出bigquery", "给出 sql", "给出sql", "生成sql", "生成 sql", "统计语句", "查询汇总语句"))
    asks_for_data_result = any(
        phrase in question or phrase in lowered
        for phrase in (
            "查一下",
            "查询一下",
            "查询数据",
            "获取数据",
            "拿到数据",
            "返回数据",
            "展示数据",
            "结果数据",
            "最终结果",
            "查询结果",
            "数据结果",
            "final data",
            "final result",
            "show data",
            "return data",
            "get data",
            "query result",
        )
    )
    return asks_for_data_result and not asks_for_sql_only


def _sql_review_response(sql: str, program: dict[str, Any], query_plan: dict[str, Any], trace: list[dict[str, Any]], *, preview_only: bool = False) -> AgentState:
    if preview_only:
        answer = (
            "下面是本次表聚合的**等价预览 SQL**，仅供 review。实际执行不会直接运行这段文本，"
            "而是通过 Metabase 结构化查询（MBQL）按相同的表/聚合/时间窗口执行，时间窗口以 Metabase 的相对区间语义为准。"
            "确认无误后点击下方授权确认执行，或拒绝本次执行。\n\n```sql\n" + sql + "\n```"
        )
        approval_prompt = "这是等价预览 SQL（实际经 Metabase 结构化查询执行）。确认无误后点击授权确认执行，或拒绝本次执行。"
    else:
        answer = "已生成 SQL，请先人工 review。确认无误后点击下方授权确认执行，或拒绝本次执行。\n\n```sql\n" + sql + "\n```"
        approval_prompt = "请 review SQL。确认无误后点击授权确认执行，或拒绝本次执行。"
    return {
        "answer": answer,
        "query_plan": {**query_plan, "requires_approval": True},
        "program": {**program, "execute": False, "requires_approval": True},
        "query_result": {"status": "requires_approval", "sql": sql, "approval_prompt": approval_prompt, "preview_only": preview_only},
        "trace": append_trace({"trace": trace}, {"step": "sql.review", "status": "requires_approval"}),
    }


def _sql_explanation_response(sql: str, answer: str, trace: list[dict[str, Any]]) -> AgentState:
    tables = sorted(set(re.findall(r"`([^`]+)`", sql)))
    if not tables:
        tables = sorted(set(re.findall(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)", sql, re.IGNORECASE)))
    cte_names = re.findall(r"(?m)^\s*([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", sql)
    return {
        "answer": answer,
        "query_plan": {"intent": "sql_explanation", "execute": False, "table_names": tables, "cte_names": cte_names},
        "program": {"type": "sql_explanation", "execute": False, "sql": sql},
        "query_result": {"status": "completed", "source": "sql_explanation", "sql": sql},
        "trace": append_trace({"trace": trace}, {"step": "sql.explain", "status": "completed", "table_count": len(tables), "cte_count": len(cte_names)}),
    }


def _sql_review_with_optional_explanation(sql: str, program: dict[str, Any], query_plan: dict[str, Any], trace: list[dict[str, Any]], analysis: str | None) -> AgentState:
    response = _sql_review_response(sql, program, query_plan, trace)
    if analysis:
        response["answer"] = analysis + "\n\n请 review SQL。确认无误后点击下方授权确认执行，或拒绝本次执行。\n\n```sql\n" + sql + "\n```"
    return response


def _table_sql_preview(schema_name: str | None, table_name: str, aggregation_function: str, field: dict[str, Any] | None, date_field: dict[str, Any] | None, relative_days: int | None, time_grain: str | None, limit: int = 200) -> str:
    table_ref = f"`{schema_name}.{table_name}`" if schema_name else f"`{table_name}`"
    value_expr = "COUNT(*)"
    if aggregation_function != "count" and field:
        field_name = str(field.get("name") or field.get("display_name"))
        value_expr = f"{aggregation_function.upper()}(`{field_name}`)"
    select_parts: list[str] = []
    group_parts: list[str] = []
    order_parts: list[str] = []
    where_parts: list[str] = []
    if date_field and time_grain:
        date_field_name = str(date_field.get("name") or date_field.get("display_name"))
        bucket = f"TIMESTAMP_TRUNC(`{date_field_name}`, {time_grain.upper()})"
        select_parts.append(f"  {bucket} AS {date_field_name}_{time_grain}")
        group_parts.append(f"{date_field_name}_{time_grain}")
        order_parts.append(f"{date_field_name}_{time_grain} ASC")
    select_parts.append(f"  {value_expr} AS {aggregation_function}")
    if date_field and relative_days is not None:
        date_field_name = str(date_field.get("name") or date_field.get("display_name"))
        where_parts.append(f"`{date_field_name}` >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {relative_days} DAY)")
    sql = "SELECT\n" + ",\n".join(select_parts) + f"\nFROM {table_ref}"
    if where_parts:
        sql += "\nWHERE " + "\n  AND ".join(where_parts)
    if group_parts:
        sql += "\nGROUP BY " + ", ".join(group_parts)
    if order_parts:
        sql += "\nORDER BY " + ", ".join(order_parts)
    sql += f"\nLIMIT {limit}"
    return sql
