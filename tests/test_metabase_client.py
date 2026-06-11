import httpx

from metabase_agent.tools.metabase.client import MetabaseClient


class FakeHttpxClient:
    responses: list[httpx.Response]
    requests: list[httpx.Request]

    def __init__(self, timeout: float) -> None:
        self.timeout = timeout

    def __enter__(self) -> "FakeHttpxClient":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        request = httpx.Request(method, url)
        self.requests.append(request)
        response = self.responses.pop(0)
        response.request = request
        return response


def test_metabase_client_retries_transient_503(monkeypatch) -> None:
    FakeHttpxClient.responses = [
        httpx.Response(503, text="unavailable"),
        httpx.Response(200, json={"data": []}),
    ]
    FakeHttpxClient.requests = []
    monkeypatch.setattr("metabase_agent.tools.metabase.client.httpx.Client", FakeHttpxClient)
    monkeypatch.setattr("metabase_agent.tools.metabase.client.time.sleep", lambda delay: None)

    result = MetabaseClient("https://example.test", "key").list_databases()

    assert result == {"data": []}
    assert len(FakeHttpxClient.requests) == 2


def test_metabase_client_raises_after_retries(monkeypatch) -> None:
    FakeHttpxClient.responses = [
        httpx.Response(503, text="first"),
        httpx.Response(503, text="second"),
        httpx.Response(503, text="third"),
    ]
    FakeHttpxClient.requests = []
    monkeypatch.setattr("metabase_agent.tools.metabase.client.httpx.Client", FakeHttpxClient)
    monkeypatch.setattr("metabase_agent.tools.metabase.client.time.sleep", lambda delay: None)

    try:
        MetabaseClient("https://example.test", "key").list_databases()
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 503
        assert len(FakeHttpxClient.requests) == 3
    else:
        raise AssertionError("Expected HTTPStatusError")


def test_metabase_client_does_not_retry_post(monkeypatch) -> None:
    FakeHttpxClient.responses = [httpx.Response(503, text="unavailable")]
    FakeHttpxClient.requests = []
    monkeypatch.setattr("metabase_agent.tools.metabase.client.httpx.Client", FakeHttpxClient)
    monkeypatch.setattr("metabase_agent.tools.metabase.client.time.sleep", lambda delay: None)

    try:
        MetabaseClient("https://example.test", "key").query({"type": "query"})
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 503
        assert len(FakeHttpxClient.requests) == 1
    else:
        raise AssertionError("Expected HTTPStatusError")


def test_metabase_client_reuses_one_connection_across_calls(monkeypatch) -> None:
    created = {"n": 0}

    class CountingClient(FakeHttpxClient):
        def __init__(self, timeout: float) -> None:
            created["n"] += 1
            super().__init__(timeout)

    FakeHttpxClient.responses = [
        httpx.Response(200, json={"data": []}),
        httpx.Response(200, json={"tables": []}),
    ]
    FakeHttpxClient.requests = []
    monkeypatch.setattr("metabase_agent.tools.metabase.client.httpx.Client", CountingClient)

    client = MetabaseClient("https://example.test", "key")
    client.list_databases()
    client.get_database_metadata(1)

    # One persistent httpx.Client is reused across both method calls.
    assert created["n"] == 1
    assert len(FakeHttpxClient.requests) == 2


def test_metabase_client_executes_native_query_payload(monkeypatch) -> None:
    FakeHttpxClient.responses = [httpx.Response(200, json={"status": "completed"})]
    FakeHttpxClient.requests = []
    bodies: list[object] = []

    def request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        bodies.append(kwargs.get("json"))
        request = httpx.Request(method, url)
        self.requests.append(request)
        response = self.responses.pop(0)
        response.request = request
        return response

    monkeypatch.setattr(FakeHttpxClient, "request", request)
    monkeypatch.setattr("metabase_agent.tools.metabase.client.httpx.Client", FakeHttpxClient)

    result = MetabaseClient("https://example.test", "key").execute_native_query(19, "SELECT 1")

    assert result == {"status": "completed"}
    assert FakeHttpxClient.requests[0].url == "https://example.test/api/dataset"
    assert bodies == [{"database": 19, "type": "native", "native": {"query": "SELECT 1"}, "parameters": []}]
