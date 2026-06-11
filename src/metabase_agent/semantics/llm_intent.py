from __future__ import annotations

import json
from typing import Any, Literal, cast

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params.reasoning import Reasoning

from metabase_agent.config.settings import Settings

ReasoningEffort = Literal["high", "xhigh"]


def _reasoning_effort(model: str) -> ReasoningEffort:
    normalized = model.lower()
    if normalized.startswith("gpt-5.5"):
        return "xhigh"
    return "high"


def parse_intent_with_llm(question: str, settings: Settings) -> dict[str, Any] | None:
    if not settings.openai_api_key:
        return None

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    prompt = (
        "你是 Metabase 只读分析 Agent 的意图解析器。只返回 JSON。"
        "字段：intent,database_name,schema_name,table_name,field_name,aggregation_function,time_grain。"
        "intent 只能是 database_count,database_list,database_table_count,database_table_list,table_field_list,table_aggregation,bigquery_usage_report_sql,sql_explanation,native_sql_query,metric_trend,metric_value,comparison,detail_lookup。"
        "表或 collection 的 count/sum/avg/min/max 用 table_aggregation，aggregation_function 只能是 count,sum,avg,min,max。"
        "如果用户粘贴 SQL 并要求解释、分析含义、判断图表理解是否正确，用 sql_explanation。"
        "只有用户明确要求执行、运行、获取查询结果或返回数据时，才用 native_sql_query。"
        "注意：BigQuery-GA 这种是 Metabase database；business_data 这种在 'BigQuery-GA 下 business_data' 语境里通常是 schema/dataset，不是 database。"
        f"\n用户问题：{question}"
    )
    if settings.openai_wire_api == "responses":
        response = httpx.post(
            f"{settings.openai_base_url.rstrip('/')}/responses",
            headers={"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"},
            json={"model": settings.openai_model, "input": prompt, "reasoning": Reasoning(effort=_reasoning_effort(settings.openai_model))},
            timeout=180,
        )
        response.raise_for_status()
        return _parse_responses_payload(response.json())

    messages = cast(
        list[ChatCompletionMessageParam],
        [
            {
                "role": "system",
                "content": (
                    "你是 Metabase 只读分析 Agent 的意图解析器。只返回 JSON。"
                    "字段：intent,database_name,schema_name,table_name,field_name,aggregation_function,time_grain。"
                    "intent 只能是 database_count,database_list,database_table_count,database_table_list,table_field_list,table_aggregation,bigquery_usage_report_sql,sql_explanation,native_sql_query,metric_trend,metric_value,comparison,detail_lookup。"
                    "表或 collection 的 count/sum/avg/min/max 用 table_aggregation，aggregation_function 只能是 count,sum,avg,min,max。"
                    "如果用户粘贴 SQL 并要求解释、分析含义、判断图表理解是否正确，用 sql_explanation。"
                    "只有用户明确要求执行、运行、获取查询结果或返回数据时，才用 native_sql_query。"
                    "注意：BigQuery-GA 这种是 Metabase database；business_data 这种在 'BigQuery-GA 下 business_data' 语境里通常是 schema/dataset，不是 database。"
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            reasoning_effort=_reasoning_effort(settings.openai_model),
            temperature=0,
            messages=messages,
            response_format={"type": "json_object"},
        )
    except Exception:
        response = client.chat.completions.create(
            model=settings.openai_model,
            reasoning_effort=_reasoning_effort(settings.openai_model),
            temperature=0,
            messages=messages,
        )
    content = response.choices[0].message.content
    return _parse_json_content(content)


def _parse_json_content(content: str | None) -> dict[str, Any] | None:
    if not content:
        return None
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").removeprefix("json").strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        return None
    return _normalize_intent(parsed)


def _parse_responses_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                return _parse_json_content(str(content.get("text", "")))
    return None


def _normalize_intent(parsed: dict[str, Any]) -> dict[str, Any]:
    aliases = {
        "count_tables": "database_table_count",
        "list_tables": "database_table_list",
        "count_databases": "database_count",
        "list_databases": "database_list",
        "list_fields": "table_field_list",
        "aggregate_table": "table_aggregation",
        "table_count": "table_aggregation",
        "count_rows": "table_aggregation",
        "explain_sql": "sql_explanation",
        "sql_analysis": "sql_explanation",
        "execute_sql": "native_sql_query",
        "run_sql": "native_sql_query",
    }
    intent = parsed.get("intent")
    if isinstance(intent, str):
        parsed["intent"] = aliases.get(intent, intent)
    return parsed
