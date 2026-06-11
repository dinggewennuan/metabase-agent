"""Deterministic local sample data for dry-run mode (no Metabase/OpenAI needed)."""
from __future__ import annotations

from typing import Any


def _dry_search() -> dict[str, Any]:
    return {
        "data": [
            {"type": "metric", "id": 10, "name": "Total Revenue", "verified": True, "description": "Revenue metric"}
        ],
        "total_count": 1,
    }


def _dry_metric() -> dict[str, Any]:
    return {
        "type": "metric",
        "id": 10,
        "name": "Total Revenue",
        "verified": True,
        "default_time_dimension_field_id": 305,
        "queryable_dimensions": [{"field_id": 305, "display_name": "Created At"}],
    }


def _dry_result() -> dict[str, Any]:
    return {
        "status": "completed",
        "data": {"cols": [{"display_name": "Created At"}, {"display_name": "Total Revenue"}], "rows": [["2026-05-07", 100], ["2026-05-08", 120]]},
        "row_count": 2,
    }


def _dry_databases() -> list[dict[str, Any]]:
    return [{"id": 19, "name": "BigQuery-GA"}, {"id": 1, "name": "business_data"}, {"id": 2, "name": "product_data"}]


def _dry_tables(database_name: str | None = None) -> list[dict[str, Any]]:
    if database_name == "BigQuery-GA":
        return [
            {"id": 11, "name": "orders", "schema": "business_data"},
            {"id": 12, "name": "users", "schema": "business_data"},
            {"id": 13, "name": "payments", "schema": "business_data"},
            {"id": 14, "name": "events", "schema": "analytics"},
        ]
    return [{"id": 11, "name": "orders"}, {"id": 12, "name": "users"}, {"id": 13, "name": "payments"}]


def _dry_table_fields() -> dict[str, Any]:
    return {"table_name": "orders", "fields": ["id", "created_at", "total", "user_id"]}


def _dry_table_query_metadata() -> dict[str, Any]:
    return {
        "fields": [
            {"id": 2, "name": "created_at", "display_name": "created_at", "base_type": "type/DateTime"},
            {"id": 3, "name": "total", "display_name": "total", "base_type": "type/Float"},
        ]
    }
