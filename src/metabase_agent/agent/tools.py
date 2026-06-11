from __future__ import annotations

from typing import Any

import httpx

from metabase_agent.agent.dry_run import (
    _dry_databases,
    _dry_table_fields,
    _dry_table_query_metadata,
    _dry_tables,
)
from metabase_agent.agent.metadata import (
    _database_items,
    _field_id,
    _field_names,
    _fields,
    _filter_tables_by_schema,
    _find_database,
    _find_field,
    _find_table,
    _first_datetime_field,
    _first_numeric_field,
    _is_schema_not_found,
    _table_items,
    _table_schema,
)
from metabase_agent.config.settings import Settings
from metabase_agent.policy.query_policy import check_program
from metabase_agent.query.bigquery_report_sql import (
    extract_native_sql,
    is_read_only_sql,
)
from metabase_agent.query.query_program_builder import (
    _table_aggregation_v1_payload,
    build_table_aggregation_program,
)
from metabase_agent.tools.metabase.client import MetabaseClient

_AGGREGATIONS = ["count", "sum", "avg", "min", "max"]


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "name": "list_databases",
            "description": "列出当前可访问的所有 Metabase 数据库名称。",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
        {
            "name": "list_tables",
            "description": "列出某个数据库（可指定 schema/dataset）下的表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "database_name": {"type": "string"},
                    "schema_name": {"type": "string", "description": "BigQuery dataset / schema，可选。"},
                },
                "required": ["database_name"],
            },
        },
        {
            "name": "list_fields",
            "description": "列出某张表的字段名。",
            "parameters": {
                "type": "object",
                "properties": {
                    "database_name": {"type": "string"},
                    "table_name": {"type": "string"},
                    "schema_name": {"type": "string"},
                },
                "required": ["database_name", "table_name"],
            },
        },
        {
            "name": "run_aggregation",
            "description": "在某张表上执行只读聚合（count/sum/avg/min/max），支持最近 N 天过滤和按天/周/月分组。",
            "parameters": {
                "type": "object",
                "properties": {
                    "database_name": {"type": "string"},
                    "table_name": {"type": "string"},
                    "schema_name": {"type": "string"},
                    "aggregation": {"type": "string", "enum": _AGGREGATIONS},
                    "field": {"type": "string", "description": "聚合的数值字段，count 时可省略。"},
                    "date_field": {"type": "string"},
                    "relative_days": {"type": "integer"},
                    "time_grain": {"type": "string", "enum": ["day", "week", "month"]},
                },
                "required": ["database_name", "table_name", "aggregation"],
            },
        },
        {
            "name": "run_sql",
            "description": "执行一条只读 SELECT/WITH SQL。执行前需要用户授权。",
            "parameters": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
                "required": ["sql"],
            },
        },
    ]


class AgentTools:
    def __init__(self, settings: Settings, *, dry_run: bool) -> None:
        self.settings = settings
        self.dry_run = dry_run
        self._client: MetabaseClient | None = None
        self.bigquery_database_id = settings.metabase_bigquery_database_id

    def client(self) -> MetabaseClient:
        if self._client is None:
            self._client = MetabaseClient(self.settings.metabase_base_url, self.settings.metabase_api_key)
        return self._client

    def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = {
            "list_databases": self._list_databases,
            "list_tables": self._list_tables,
            "list_fields": self._list_fields,
            "run_aggregation": self._run_aggregation,
            "run_sql": self._run_sql,
        }.get(name)
        if handler is None:
            return {"status": "error", "error": f"unknown tool: {name}"}
        try:
            return handler(arguments)
        except httpx.HTTPStatusError as exc:
            return {"status": "failed", "error": f"{exc.response.status_code} {exc.response.reason_phrase}"}

    def _databases(self) -> list[dict[str, Any]]:
        if self.dry_run:
            return _dry_databases()
        return _database_items(self.client().list_databases())

    def _list_databases(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        databases = self._databases()
        return {"status": "completed", "databases": [str(item.get("name")) for item in databases if item.get("name")]}

    def _resolve_tables(self, database_name: str, schema_name: str | None) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        databases = self._databases()
        database = _find_database(databases, database_name)
        if self.dry_run:
            db_name = str(database.get("name")) if database else database_name
            tables = _dry_tables(db_name)
        else:
            if database is None:
                return None, []
            database_id = int(database["id"])
            tables = self._load_tables(database_id, schema_name)
        if schema_name and any(_table_schema(table) for table in tables):
            tables = _filter_tables_by_schema(tables, schema_name)
        return database, tables

    def _load_tables(self, database_id: int, schema_name: str | None) -> list[dict[str, Any]]:
        if schema_name:
            try:
                return _table_items(self.client().get_database_schema(database_id, schema_name))
            except httpx.HTTPStatusError as exc:
                if not _is_schema_not_found(exc):
                    raise
        metadata = self.client().get_database_metadata(database_id)
        return metadata.get("tables", []) if isinstance(metadata, dict) else []

    def _list_tables(self, arguments: dict[str, Any]) -> dict[str, Any]:
        database, tables = self._resolve_tables(str(arguments.get("database_name") or ""), arguments.get("schema_name"))
        if database is None and not self.dry_run:
            return {"status": "not_found", "error": f"database not found: {arguments.get('database_name')}"}
        names = [str(table.get("name")) for table in tables if isinstance(table, dict) and table.get("name")]
        return {"status": "completed", "tables": names}

    def _table_fields(self, table: dict[str, Any]) -> list[dict[str, Any]]:
        if self.dry_run:
            return _fields(_dry_table_query_metadata())
        return _fields(self.client().get_table_query_metadata(int(table["id"])))

    def _list_fields(self, arguments: dict[str, Any]) -> dict[str, Any]:
        _database, tables = self._resolve_tables(str(arguments.get("database_name") or ""), arguments.get("schema_name"))
        table = _find_table(tables, str(arguments.get("table_name") or ""))
        if table is None:
            if self.dry_run:
                return {"status": "completed", "fields": list(_dry_table_fields()["fields"])}
            return {"status": "not_found", "error": f"table not found: {arguments.get('table_name')}"}
        if self.dry_run:
            return {"status": "completed", "fields": list(_dry_table_fields()["fields"])}
        return {"status": "completed", "fields": _field_names(self.client().get_table_query_metadata(int(table["id"])))}

    def _run_aggregation(self, arguments: dict[str, Any]) -> dict[str, Any]:
        aggregation = str(arguments.get("aggregation") or "count")
        if aggregation not in _AGGREGATIONS:
            return {"status": "error", "error": f"aggregation must be one of {_AGGREGATIONS}"}
        _database, tables = self._resolve_tables(str(arguments.get("database_name") or ""), arguments.get("schema_name"))
        table = _find_table(tables, str(arguments.get("table_name") or ""))
        if table is None:
            return {"status": "not_found", "error": f"table not found: {arguments.get('table_name')}"}
        fields = self._table_fields(table)
        field = None
        if aggregation != "count":
            field = _find_field(fields, str(arguments.get("field") or "")) if arguments.get("field") else _first_numeric_field(fields)
            if field is None:
                return {"status": "not_found", "error": "no numeric field available for aggregation"}
        relative_days = arguments.get("relative_days")
        time_grain = arguments.get("time_grain")
        date_field = None
        if relative_days is not None or time_grain:
            date_field = _find_field(fields, str(arguments.get("date_field") or "")) if arguments.get("date_field") else _first_datetime_field(fields)
            if date_field is None:
                return {"status": "not_found", "error": "no datetime field available for time filter/grouping"}
        field_id = _field_id(field) if field else None
        date_field_id = _field_id(date_field) if date_field else None
        program = build_table_aggregation_program(int(table["id"]), aggregation, field_id, date_field_id=date_field_id, relative_days=relative_days, time_grain=time_grain)
        policy = check_program(program)
        if not policy["allowed"]:
            return {"status": "blocked", "error": policy["reason"]}
        if self.dry_run:
            return {"status": "completed", "row_count": 1, "data": {"cols": [{"display_name": aggregation}], "rows": [[3]]}}
        return self._execute_aggregation(program)

    def _execute_aggregation(self, program: dict[str, Any]) -> dict[str, Any]:
        query_program = {"source": program["source"], "operations": program["operations"]}
        try:
            return self.client().query(query_program)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 404:
                raise
            constructed = self.client().construct_query_v1(_table_aggregation_v1_payload(program))
            return self.client().execute_query_v1(constructed)

    def _run_sql(self, arguments: dict[str, Any]) -> dict[str, Any]:
        sql = extract_native_sql(str(arguments.get("sql") or "")) or str(arguments.get("sql") or "")
        if not sql:
            return {"status": "not_found", "error": "missing SQL"}
        if not is_read_only_sql(sql):
            return {"status": "blocked", "error": "only read-only SELECT/WITH SQL is allowed", "sql": sql}
        if self.dry_run:
            return {"status": "completed", "row_count": 0, "data": {"cols": [], "rows": []}, "dry_run": True, "sql": sql}
        return {**self.client().execute_native_query(self.bigquery_database_id, sql), "sql": sql}
