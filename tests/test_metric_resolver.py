from metabase_agent.metrics.metric_resolver import choose_metric


def test_choose_verified_metric() -> None:
    result = {
        "data": [
            {"type": "table", "id": 1, "name": "orders"},
            {"type": "metric", "id": 2, "name": "Total Revenue", "verified": True},
        ]
    }

    metric = choose_metric(result, ["revenue"])

    assert metric is not None
    assert metric["id"] == 2
