"""Helpers for matching/extracting Metabase databases, tables and fields from API payloads."""
from __future__ import annotations

import re
from typing import Any

import httpx


def _database_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [item for item in payload["data"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _find_database(databases: list[dict[str, Any]], database_name: str) -> dict[str, Any] | None:
    if not database_name:
        return None
    expected = database_name.lower()
    for database in databases:
        name = str(database.get("name", "")).lower()
        if name == expected:
            return database
    for database in databases:
        name = str(database.get("name", "")).lower()
        if expected in name:
            return database
    return None


def _table_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("tables"), list):
        return [item for item in payload["tables"] if isinstance(item, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [item for item in payload["data"] if isinstance(item, dict)]
    return []


def _find_table(tables: list[dict[str, Any]], table_name: str) -> dict[str, Any] | None:
    expected = table_name.lower()
    for table in tables:
        name = str(table.get("name", "")).lower()
        display_name = str(table.get("display_name", "")).lower()
        if name == expected or display_name == expected:
            return table
    for table in tables:
        name = str(table.get("name", "")).lower()
        display_name = str(table.get("display_name", "")).lower()
        if expected in name or expected in display_name:
            return table
    candidates = _rank_table_candidates(tables, table_name)
    if len(candidates) == 1 or (candidates and len(candidates) > 1 and candidates[0][1] > candidates[1][1]):
        return candidates[0][0]
    return None


def _rank_table_candidates(tables: list[dict[str, Any]], table_name: str) -> list[tuple[dict[str, Any], int]]:
    expected_tokens = _table_tokens(table_name)
    if not expected_tokens:
        return []
    ranked: list[tuple[dict[str, Any], int]] = []
    for table in tables:
        name = str(table.get("name") or "")
        display_name = str(table.get("display_name") or "")
        tokens = _table_tokens(f"{name} {display_name}")
        overlap = expected_tokens & tokens
        if not overlap:
            continue
        score = len(overlap) * 10
        if expected_tokens <= tokens:
            score += 5
        if _singularize(table_name.lower()) in {_singularize(name.lower()), _singularize(display_name.lower())}:
            score += 20
        ranked.append((table, score))
    ranked.sort(key=lambda item: (-item[1], str(item[0].get("name") or "")))
    return ranked


def _table_tokens(value: str) -> set[str]:
    raw_tokens = re.findall(r"[a-z0-9]+", value.lower().replace("_", " ").replace("-", " "))
    ignored = {"bll", "tbl", "dim", "fact", "raw", "ods", "dwd", "dws"}
    return {_singularize(token) for token in raw_tokens if token and token not in ignored}


def _singularize(value: str) -> str:
    return value[:-1] if len(value) > 3 and value.endswith("s") else value


def _fields(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    fields = payload.get("fields", [])
    return [field for field in fields if isinstance(field, dict)]


def _field_names(payload: Any) -> list[str]:
    return [str(field.get("name") or field.get("display_name")) for field in _fields(payload)]


def _field_id(field: dict[str, Any]) -> int | None:
    value = field.get("id") or field.get("field_id")
    if value is None:
        return None
    return int(value)


def _agent_field_id(field: dict[str, Any]) -> str | None:
    value = field.get("field_id")
    return str(value) if value is not None else None


def _find_field(fields: list[dict[str, Any]], field_name: str) -> dict[str, Any] | None:
    expected = field_name.lower()
    for field in fields:
        name = str(field.get("name") or "").lower()
        display_name = str(field.get("display_name") or "").lower()
        if name == expected or display_name == expected:
            return field
    for field in fields:
        name = str(field.get("name") or "").lower()
        display_name = str(field.get("display_name") or "").lower()
        if expected in name or expected in display_name:
            return field
    return None


def _match_agent_field_id(agent_fields: list[dict[str, Any]], field: dict[str, Any] | None) -> str | None:
    if not field:
        return None
    direct = _agent_field_id(field)
    if direct:
        return direct
    name = str(field.get("name") or "")
    display_name = str(field.get("display_name") or "")
    for candidate in (name, display_name):
        if not candidate:
            continue
        matched = _find_field(agent_fields, candidate)
        if matched:
            return _agent_field_id(matched)
    return None


def _first_numeric_field(fields: list[dict[str, Any]]) -> dict[str, Any] | None:
    for field in fields:
        field_type = str(field.get("effective_type") or field.get("base_type") or "").lower()
        if "integer" in field_type or "float" in field_type or "decimal" in field_type or "number" in field_type:
            return field
    return None


def _first_datetime_field(fields: list[dict[str, Any]]) -> dict[str, Any] | None:
    preferred_names = ("created_at", "created", "create_time", "event_time", "timestamp", "date", "time", "last_sub_upgrade_time")
    for expected in preferred_names:
        field = _find_field(fields, expected)
        if field and _is_datetime_field(field):
            return field
    for field in fields:
        if _is_datetime_field(field):
            return field
    return None


def _is_datetime_field(field: dict[str, Any]) -> bool:
    field_type = str(field.get("effective_type") or field.get("base_type") or field.get("type") or "").lower()
    semantic_type = str(field.get("semantic_type") or "").lower()
    return "date" in field_type or "time" in field_type or "timestamp" in field_type or "date" in semantic_type or "time" in semantic_type


def _filter_tables_by_schema(tables: list[dict[str, Any]], schema_name: str | None) -> list[dict[str, Any]]:
    if not schema_name:
        return tables
    expected = schema_name.lower()
    return [table for table in tables if str(table.get("schema") or table.get("db_schema") or table.get("database_schema") or "").lower() == expected]


def _table_schema(table: dict[str, Any]) -> str:
    return str(table.get("schema") or table.get("db_schema") or table.get("database_schema") or "")


def _infer_database_name(databases: list[dict[str, Any]], schema_name: str | None) -> str:
    if not schema_name:
        return ""
    for database in databases:
        engine = str(database.get("engine") or database.get("details", {}).get("engine") or "").lower()
        name = str(database.get("name") or "")
        if "bigquery" in engine or "bigquery" in name.lower():
            return name
    for database in databases:
        engine = str(database.get("engine") or database.get("details", {}).get("engine") or "").lower()
        if "mongo" in engine:
            return str(database.get("name") or "")
    for database in databases:
        name = str(database.get("name") or "")
        if "mongo" in name.lower():
            return name
    return ""


def _database_names(databases: list[dict[str, Any]]) -> list[str]:
    return [str(database.get("name")) for database in databases if database.get("name")]


def _is_schema_not_found(exc: httpx.HTTPStatusError) -> bool:
    return exc.response.status_code == 404
