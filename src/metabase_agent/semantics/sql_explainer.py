from __future__ import annotations

import re

from metabase_agent.config.settings import Settings
from metabase_agent.semantics.llm_client import complete

_SYSTEM_PROMPT = (
    "你是资深的数据分析与 SQL 审查专家。只针对用户给出的这条 SQL 进行分析，"
    "绝不编造 SQL 中不存在的表、字段或业务含义。用中文分点输出：\n"
    "1) 这条 SQL 的业务目标；\n"
    "2) 数据来源（涉及的表 / CTE）与口径；\n"
    "3) 关键计算逻辑（聚合、分组、过滤、时间处理等）；\n"
    "4) 容易理解错或存在数据风险的地方（重复计数、时区、NULL、去重口径、汇率/状态码假设等）。"
)


def explain_sql_with_llm(sql: str, settings: Settings) -> str:
    return complete(_SYSTEM_PROMPT, f"请分析下面这条 SQL：\n```sql\n{sql}\n```", settings)


def structural_sql_summary(sql: str) -> str:
    """Deterministic fallback summary computed from the SQL itself (dry-run / no LLM)."""
    tables = _extract_tables(sql)
    ctes = _extract_ctes(sql)
    features = _detect_features(sql)

    lines = ["以下是基于这条 SQL 结构本身的客观摘要（未启用 LLM 深度解读）：", ""]
    if tables:
        lines.append("- 涉及的数据表：" + "、".join(f"`{table}`" for table in tables))
    else:
        lines.append("- 未能从 SQL 中识别出明确的表名（no table detected）")
    if ctes:
        lines.append(f"- 包含 {len(ctes)} 个 CTE 子查询：" + "、".join(ctes))
    for feature in features:
        lines.append("- " + feature)
    lines.append("")
    lines.append(
        "如需逐句的业务含义解读，请在真实模式（配置 OPENAI_API_KEY）下重试，"
        "我会调用大模型针对这条 SQL 本身进行分析。"
    )
    return "\n".join(lines)


def _extract_tables(sql: str) -> list[str]:
    tables = sorted(set(re.findall(r"`([^`]+)`", sql)))
    if tables:
        return tables
    return sorted(
        set(
            re.findall(
                r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)",
                sql,
                re.IGNORECASE,
            )
        )
    )


def _extract_ctes(sql: str) -> list[str]:
    # Match `name AS (` anywhere — including the first CTE on the `WITH name AS (`
    # line. Column aliases (`... AS alias`) are not followed by `(`, so they don't match.
    names = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s+AS\s*\(", sql, re.IGNORECASE)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name.lower() == "with" or name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return ordered


def _detect_features(sql: str) -> list[str]:
    features: list[str] = []
    aggregations = sorted({match.upper() for match in re.findall(r"\b(count|sum|avg|min|max)\s*\(", sql, re.IGNORECASE)})
    if aggregations:
        features.append("聚合函数（aggregations）：" + "、".join(aggregations))
    if re.search(r"\bdistinct\b", sql, re.IGNORECASE):
        features.append("包含去重（DISTINCT）")
    if re.search(r"\bgroup\s+by\b", sql, re.IGNORECASE):
        features.append("按维度分组（GROUP BY）")
    if re.search(r"\bjoin\b", sql, re.IGNORECASE):
        features.append("包含表连接（JOIN）")
    if re.search(r"\bunion\b", sql, re.IGNORECASE):
        features.append("包含 UNION 合并多个结果集")
    if re.search(r"\bwhere\b", sql, re.IGNORECASE):
        features.append("包含过滤条件（WHERE）")
    if re.search(r"\b(?:timestamp_trunc|date_trunc|datetime_trunc)\b", sql, re.IGNORECASE):
        features.append("按时间粒度截断（TIMESTAMP_TRUNC / DATE_TRUNC）")
    if re.search(r"\bover\s*\(", sql, re.IGNORECASE):
        features.append("使用窗口函数（OVER）")
    if re.search(r"\border\s+by\b", sql, re.IGNORECASE):
        features.append("结果排序（ORDER BY）")
    if re.search(r"\blimit\b", sql, re.IGNORECASE):
        features.append("限制返回行数（LIMIT）")
    return features
