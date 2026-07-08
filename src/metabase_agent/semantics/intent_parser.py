from __future__ import annotations

import re
from typing import Literal, TypedDict

from metabase_agent.query.bigquery_report_sql import extract_native_sql
from metabase_agent.semantics.business_glossary import normalize_business_terms

Intent = Literal[
    "database_count",
    "database_list",
    "database_table_count",
    "database_table_list",
    "table_field_list",
    "table_aggregation",
    "bigquery_usage_report_sql",
    "sql_explanation",
    "native_sql_query",
    "metric_trend",
    "metric_value",
    "comparison",
    "detail_lookup",
]


class ParsedIntent(TypedDict):
    intent: Intent
    business_terms: list[str]
    time_grain: str | None
    database_name: str | None
    schema_name: str | None
    table_name: str | None
    field_name: str | None
    date_field_name: str | None
    aggregation_function: str | None
    relative_days: int | None
    raw_question: str


def parse_intent(question: str) -> ParsedIntent:
    trend_words = ("trend", "趋势", "变化", "按天", "daily", "weekly", "monthly")
    comparison_words = ("compare", "对比", "比较", "vs", "同比", "环比")
    database_name = extract_database_name(question)
    schema_name = extract_schema_name(question, database_name)
    table_name = extract_table_name(question)
    field_name = extract_field_name(question)
    date_field_name = extract_date_field_name(question)
    aggregation_function = extract_aggregation_function(question)
    relative_days = extract_relative_days(question)
    time_grain = extract_time_grain(question)
    parsed_intent: Intent
    if is_sql_explanation_request(question):
        parsed_intent = "sql_explanation"
    elif is_bigquery_usage_report_request(question):
        parsed_intent = "bigquery_usage_report_sql"
    elif extract_native_sql(question) or re.match(r"^\s*(ALTER|CREATE|DELETE|DROP|GRANT|INSERT|MERGE|REPLACE|REVOKE|TRUNCATE|UPDATE)\b", question, re.IGNORECASE):
        parsed_intent = "native_sql_query"
    elif table_name and aggregation_function:
        parsed_intent = "table_aggregation"
    elif table_name and any(word in question for word in ("字段", "列", "schema", "结构")):
        parsed_intent = "table_field_list"
    elif database_name and "表" in question and any(word in question for word in ("哪些", "列表", "列出", "所有", "什么")):
        parsed_intent = "database_table_list"
    elif database_name and "表" in question and any(word in question for word in ("多少", "几个", "数量")):
        parsed_intent = "database_table_count"
    elif "数据库" in question and any(word in question for word in ("哪些", "列表", "列出", "所有")):
        parsed_intent = "database_list"
    elif "数据库" in question and any(word in question for word in ("多少", "几个", "数量")):
        parsed_intent = "database_count"
    elif any(word in question.lower() or word in question for word in comparison_words):
        parsed_intent = "comparison"
    elif any(word in question.lower() or word in question for word in trend_words):
        parsed_intent = "metric_trend"
    else:
        parsed_intent = "metric_value"

    return {
        "intent": parsed_intent,
        "business_terms": normalize_business_terms(question),
        "time_grain": (time_grain or "day") if parsed_intent == "metric_trend" else time_grain if parsed_intent == "table_aggregation" else None,
        "database_name": database_name,
        "schema_name": schema_name,
        "table_name": table_name,
        "field_name": field_name,
        "date_field_name": date_field_name,
        "aggregation_function": aggregation_function,
        "relative_days": relative_days,
        "raw_question": question,
    }


def extract_database_name(question: str) -> str | None:
    scoped_match = re.search(r"查询\s*([A-Za-z0-9_\-]+)\s*下\s*([A-Za-z0-9_\-]+)", question)
    if scoped_match:
        return scoped_match.group(1)
    match = re.search(r"([A-Za-z0-9_\-]+)\s*(?:这个)?数据库", question)
    if match:
        return match.group(1)
    return None


def is_bigquery_usage_report_request(question: str) -> bool:
    lowered = question.lower()
    has_bigquery = "bigquery" in lowered or "bigquery" in question
    has_sql_request = any(word in question for word in ("统计语句", "查询汇总语句", "查询语句", "统计")) or "sql" in lowered
    has_month_range = bool(re.search(r"\d{4}-\d{2}|\d{6}|\d{1,2}\s*月", question))
    has_usage_tables = any(table in question for table in ("fs_results", "aigc_imagecontents", "aigc_videocontents", "aigc_sessions", "aigc_audiocontents"))
    has_web_api = "web" in lowered and "api" in lowered
    return has_bigquery and has_sql_request and (has_month_range or has_usage_tables) and has_web_api


# CJK markers are matched as substrings (no word boundaries in Chinese); Latin
# markers are matched with word boundaries so they don't fire on substrings of
# SQL keywords/identifiers (e.g. "run" inside TIMESTAMP_TRUNC).
_EXPLANATION_WORDS_CJK = ("分析", "含义", "解释", "理解", "为啥", "为什么", "什么意思", "对不对", "不太对")
_EXPLANATION_WORDS_LATIN = ("explain", "analyze", "meaning")
_EXECUTION_WORDS_CJK = ("执行", "运行", "查询结果", "返回数据", "最终数据", "获取数据")
_EXECUTION_WORDS_LATIN = ("execute", "run")


def _matches_markers(question: str, cjk: tuple[str, ...], latin: tuple[str, ...]) -> bool:
    if any(word in question for word in cjk):
        return True
    lowered = question.lower()
    return any(re.search(rf"\b{word}\b", lowered) for word in latin)


def is_sql_explanation_request(question: str) -> bool:
    if not extract_native_sql(question):
        return False
    has_explanation_goal = _matches_markers(question, _EXPLANATION_WORDS_CJK, _EXPLANATION_WORDS_LATIN)
    has_execution_goal = _matches_markers(question, _EXECUTION_WORDS_CJK, _EXECUTION_WORDS_LATIN)
    return has_explanation_goal and not has_execution_goal


def wants_sql_explanation(question: str) -> bool:
    if not extract_native_sql(question):
        return False
    return _matches_markers(question, _EXPLANATION_WORDS_CJK, _EXPLANATION_WORDS_LATIN)


def is_safe_rule_intent_override(rule_intent: str | None, llm_intent: str | None) -> bool:
    if not rule_intent or not llm_intent or rule_intent == llm_intent:
        return False
    if rule_intent in {"native_sql_query", "sql_explanation"}:
        return llm_intent in {"native_sql_query", "sql_explanation"}
    if rule_intent == "bigquery_usage_report_sql":
        return llm_intent in {"bigquery_usage_report_sql", "sql_explanation"}
    if rule_intent == "table_aggregation":
        return llm_intent in {"table_aggregation", "table_field_list"}
    return False


def extract_schema_name(question: str, database_name: str | None) -> str | None:
    scoped_match = re.search(r"查询\s*([A-Za-z0-9_\-]+)\s*下\s*([A-Za-z0-9_\-]+)", question)
    if scoped_match:
        return scoped_match.group(2)
    schema_table_match = re.search(r"([A-Za-z0-9_\-]+)\s*下\s*([A-Za-z0-9_\-]+)", question)
    if schema_table_match:
        return schema_table_match.group(1)
    if database_name and "下" in question:
        return database_name
    return None


def extract_table_name(question: str) -> str | None:
    match = re.search(r"([A-Za-z0-9_\-]+)\s*(?:这个)?表", question)
    if match:
        return match.group(1)
    match = re.search(r"([A-Za-z0-9_\-]+)\s+(?:table|collection)\b", question, re.IGNORECASE)
    if match:
        return match.group(1)
    schema_table_match = re.search(
        r"[A-Za-z0-9_\-]+\s*下\s*([A-Za-z0-9_\-\s]+?)(?=\s*(?:最近|近|过去|昨天|上周|每天|按天|每周|按周|每月|按月|count|多少|条数|行数|记录数|求和|总和|合计|平均|最大|最小|字段|列|表|,|，|。|$))",
        question,
    )
    if schema_table_match and extract_aggregation_function(question):
        return _normalize_extracted_table_name(schema_table_match.group(1))
    return None


def _normalize_extracted_table_name(value: str) -> str:
    return re.sub(r"\s+", "_", value.strip())


def extract_aggregation_function(question: str) -> str | None:
    lowered = question.lower()
    if any(word in lowered or word in question for word in ("count", "多少条", "多少行", "记录数", "行数", "条数")):
        return "count"
    if any(word in lowered or word in question for word in ("sum", "求和", "总和", "合计")):
        return "sum"
    if any(word in lowered or word in question for word in ("avg", "average", "平均")):
        return "avg"
    if any(word in lowered or word in question for word in ("max", "最大")):
        return "max"
    if any(word in lowered or word in question for word in ("min", "最小")):
        return "min"
    return None


def extract_field_name(question: str) -> str | None:
    match = re.search(r"(?:字段|列)\s*([A-Za-z0-9_\-]+)", question)
    if match:
        return match.group(1)
    match = re.search(r"([A-Za-z0-9_\-]+)\s*(?:求和|总和|合计|平均|最大|最小|sum|avg|average|max|min)", question, re.IGNORECASE)
    if match:
        field_name = match.group(1)
        if field_name.lower() not in {"count", "sum", "avg", "average", "max", "min"}:
            return field_name
    return None


def extract_date_field_name(question: str) -> str | None:
    match = re.search(r"(?:时间字段|日期字段|按字段)\s*([A-Za-z0-9_\-]+)", question)
    if match:
        return match.group(1)
    return None


def extract_relative_days(question: str) -> int | None:
    match = re.search(r"(?:最近|近|过去)\s*(\d+)\s*天", question)
    if match:
        return int(match.group(1))
    if "昨天" in question:
        return 1
    if "上周" in question:
        return 7
    return None


def extract_time_grain(question: str) -> str | None:
    lowered = question.lower()
    if any(word in question or word in lowered for word in ("每天", "按天", "daily", "day")):
        return "day"
    if any(word in question or word in lowered for word in ("每周", "按周", "weekly", "week")):
        return "week"
    if any(word in question or word in lowered for word in ("每月", "按月", "monthly", "month")):
        return "month"
    return None
