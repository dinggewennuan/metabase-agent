from __future__ import annotations

import time
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

TRANSIENT_STATUS_CODES = {429, 502, 503, 504}
RETRYABLE_METHODS = {"GET", "HEAD"}


class MetabaseClient:
    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0, max_retries: int = 2) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self._client: httpx.Client | None = None

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"X-API-Key": self.api_key}

    def _http(self) -> httpx.Client:
        # One persistent client per MetabaseClient, so HTTP keep-alive is reused
        # across the several sequential calls a single question makes.
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def request(self, method: str, path: str, *, json: Any | None = None, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        retryable = method.upper() in RETRYABLE_METHODS
        client = self._http()
        response = client.request(method, url, headers=self._headers(), json=json, params=params)
        for attempt in range(self.max_retries):
            if not retryable or response.status_code not in TRANSIENT_STATUS_CODES:
                break
            time.sleep(_retry_delay(response, attempt))
            response = client.request(method, url, headers=self._headers(), json=json, params=params)
        response.raise_for_status()
        return response.json()

    def response_text(self, exc: httpx.HTTPStatusError) -> str:
        return exc.response.text

    def ping(self) -> Any:
        return self.request("GET", "/api/agent/v1/ping")

    def list_databases(self) -> Any:
        return self.request("GET", "/api/database")

    def get_database_metadata(self, database_id: int) -> Any:
        return self.request("GET", f"/api/database/{database_id}/metadata")

    def get_database_schema(self, database_id: int, schema_name: str) -> Any:
        return self.request(
            "GET",
            f"/api/database/{database_id}/schema/{schema_name}",
            params={"include_hidden": "true"},
        )

    def search(self, queries: list[str]) -> Any:
        return self.request(
            "POST",
            "/api/agent/v1/search",
            json={"term_queries": queries, "semantic_queries": queries},
        )

    def get_table(self, table_id: int, *, with_field_values: bool = False) -> Any:
        return self.request(
            "GET",
            f"/api/agent/v1/table/{table_id}",
            params={"with-field-values": str(with_field_values).lower(), "with-metrics": "true"},
        )

    def get_table_query_metadata(self, table_id: int) -> Any:
        return self.request("GET", f"/api/table/{table_id}/query_metadata")

    def get_metric(self, metric_id: int, *, with_field_values: bool = False) -> Any:
        return self.request(
            "GET",
            f"/api/agent/v1/metric/{metric_id}",
            params={"with-field-values": str(with_field_values).lower(), "with-queryable-dimensions": "true"},
        )

    def construct_query(self, program: dict[str, Any]) -> Any:
        return self.request("POST", "/api/agent/v2/construct-query", json=program)

    def construct_query_v1(self, payload: dict[str, Any]) -> Any:
        return self.request("POST", "/api/agent/v1/construct-query", json=payload)

    def execute_query_v1(self, payload: dict[str, Any]) -> Any:
        return self.request("POST", "/api/agent/v1/execute", json=payload)

    def query(self, program_or_token: dict[str, Any]) -> Any:
        return self.request("POST", "/api/agent/v2/query", json=program_or_token)

    def execute_native_query(self, database_id: int, sql: str) -> Any:
        return self.request(
            "POST",
            "/api/dataset",
            json={"database": database_id, "type": "native", "native": {"query": sql}, "parameters": []},
        )

    def execute_mbql_query(self, payload: dict[str, Any]) -> Any:
        return self.request("POST", "/api/dataset", json=payload)


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return min(float(retry_after), 5.0)
        except ValueError:
            retry_at = parsedate_to_datetime(retry_after)
            return min(max(retry_at.timestamp() - time.time(), 0.0), 5.0)
    return min(0.25 * (2**attempt), 2.0)
