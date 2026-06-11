from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

FORBIDDEN_SQL_WORDS = {
    "ALTER",
    "CALL",
    "CREATE",
    "DECLARE",
    "DELETE",
    "DROP",
    "EXECUTE",
    "EXPORT",
    "GRANT",
    "INSERT",
    "LOAD",
    "MERGE",
    "REPLACE",
    "REVOKE",
    "TRUNCATE",
    "UPDATE",
}


@lru_cache(maxsize=1)
def _monthly_usage_report_template() -> str:
    return (Path(__file__).parent / "templates" / "monthly_usage_report.sql").read_text(encoding="utf-8").rstrip("\n")


def build_monthly_usage_report_sql(
    start_date: str = "2025-11-01",
    end_date_exclusive: str = "2026-05-01",
    timezone: str = "US/Pacific",
) -> str:
    return _monthly_usage_report_template().format(start_date=start_date, end_date_exclusive=end_date_exclusive, timezone=timezone)


def extract_native_sql(text: str) -> str | None:
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return normalize_bigquery_sql(_clean_sql(_truncate_sql_candidate(fenced.group(1))))
    match = re.search(r"\b(WITH|SELECT)\b[\s\S]*", text, re.IGNORECASE)
    if match:
        return normalize_bigquery_sql(_clean_sql(_truncate_sql_candidate(match.group(0))))
    return None


def normalize_bigquery_sql(sql: str) -> str:
    normalized = _remove_trailing_non_sql_text(sql)
    normalized = re.sub(r"(?i)\b(FROM|JOIN)\s+business_data\.([A-Za-z_][A-Za-z0-9_]*)", r"\1 `business_data.\2`", normalized)
    normalized = re.sub(
        r"(?<!`)\bbusiness_data\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
        r"`business_data.\1`.\2",
        normalized,
    )
    return normalized


def is_read_only_sql(sql: str) -> bool:
    normalized = _strip_leading_comments(sql).strip()
    if not re.match(r"^(WITH|SELECT)\b", normalized, re.IGNORECASE):
        return False
    if _has_extra_statement(normalized):
        return False
    tokens = {token.upper() for token in re.findall(r"\b[A-Za-z_]+\b", _mask_sql_literals_comments_and_identifiers(normalized))}
    return not bool(tokens & FORBIDDEN_SQL_WORDS)


def _clean_sql(sql: str) -> str:
    cleaned = sql.strip()
    return cleaned[:-1].strip() if cleaned.endswith(";") else cleaned


def _truncate_sql_candidate(sql: str) -> str:
    lines = sql.strip().splitlines()
    cleaned_lines: list[str] = []
    for line in lines:
        cleaned_line = line.rstrip()
        if _contains_cjk(cleaned_line):
            cleaned_line = re.split(r"[\u4e00-\u9fff]", cleaned_line, maxsplit=1)[0].rstrip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
            break
        cleaned_lines.append(cleaned_line)
    return _trim_incomplete_sql_tail("\n".join(cleaned_lines).strip())


def _remove_trailing_non_sql_text(sql: str) -> str:
    return _truncate_sql_candidate(sql)


def _trim_incomplete_sql_tail(sql: str) -> str:
    trimmed = sql.strip()
    while True:
        next_trimmed = re.sub(r"(?i)\s+(?:AS|AND|OR|WHERE|HAVING|LIMIT|OFFSET)\s*$", "", trimmed).rstrip()
        next_trimmed = re.sub(r"(?i)\s+GROUP\s+BY\s*$", "", next_trimmed).rstrip()
        next_trimmed = re.sub(r"(?i)\s+ORDER\s+BY\s*$", "", next_trimmed).rstrip()
        if next_trimmed == trimmed:
            return trimmed
        trimmed = next_trimmed


def _has_extra_statement(sql: str) -> bool:
    for index, char in enumerate(sql):
        if char != ";" or _is_inside_sql_literal_or_comment(sql, index):
            continue
        if sql[index + 1 :].strip():
            return True
    return False


def _mask_sql_literals_comments_and_identifiers(sql: str) -> str:
    chars = list(sql)
    index = 0
    while index < len(chars):
        if sql.startswith("--", index):
            end = sql.find("\n", index)
            end = len(chars) if end == -1 else end
            for mask_index in range(index, end):
                chars[mask_index] = " "
            index = end
            continue
        if sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            end = len(chars) if end == -1 else end + 2
            for mask_index in range(index, end):
                chars[mask_index] = " "
            index = end
            continue
        if chars[index] in {"'", '"', "`"}:
            quote = chars[index]
            end = _quoted_end(sql, index, quote)
            for mask_index in range(index, end):
                chars[mask_index] = " "
            index = end
            continue
        index += 1
    return "".join(chars)


def _is_inside_sql_literal_or_comment(sql: str, target_index: int) -> bool:
    return _mask_sql_literals_comments_and_identifiers(sql[: target_index + 1])[-1] == " "


def _quoted_end(sql: str, start: int, quote: str) -> int:
    index = start + 1
    while index < len(sql):
        if sql[index] == quote:
            if quote == "'" and index + 1 < len(sql) and sql[index + 1] == "'":
                index += 2
                continue
            return index + 1
        index += 1
    return len(sql)



def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _strip_leading_comments(sql: str) -> str:
    cleaned = sql.lstrip()
    while True:
        if cleaned.startswith("--"):
            _, _, cleaned = cleaned.partition("\n")
            cleaned = cleaned.lstrip()
            continue
        if cleaned.startswith("/*"):
            _, end, rest = cleaned.partition("*/")
            if not end:
                return ""
            cleaned = rest.lstrip()
            continue
        return cleaned
