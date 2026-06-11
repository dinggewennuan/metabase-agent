from metabase_agent.agent.graph import _find_table, _infer_database_name, build_graph
from metabase_agent.config.settings import Settings


def test_graph_dry_run_completes() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "上周收入趋势怎么样？", "dry_run": True})


    assert "Total Revenue" in result["answer"]
    assert result["query_result"]["status"] == "completed"
    assert result["program"]["source"] == {"type": "metric", "id": 10}


def test_metric_query_requires_approval_in_real_mode(monkeypatch) -> None:
    executed: list[dict] = []
    monkeypatch.setattr("metabase_agent.agent.graph.parse_intent_with_llm", lambda question, settings: None)
    monkeypatch.setattr(
        "metabase_agent.agent.graph.MetabaseClient.search",
        lambda self, queries: {"data": [{"type": "metric", "id": 10, "name": "Total Revenue", "verified": True}], "total_count": 1},
    )
    monkeypatch.setattr(
        "metabase_agent.agent.graph.MetabaseClient.get_metric",
        lambda self, metric_id, **kwargs: {"type": "metric", "id": 10, "name": "Total Revenue", "default_time_dimension_field_id": 305},
    )
    monkeypatch.setattr(
        "metabase_agent.agent.graph.MetabaseClient.query",
        lambda self, program: (executed.append(program), {"status": "completed", "row_count": 1, "data": {"cols": [], "rows": [[1]]}})[1],
    )
    graph = build_graph(Settings(AGENT_DRY_RUN=False, METABASE_API_KEY="test-key"))

    pending = graph.invoke({"question": "上周收入趋势怎么样？", "dry_run": False, "sql_approved": False})
    assert pending["query_result"]["status"] == "requires_approval"
    assert executed == []  # must NOT hit Metabase before approval

    done = graph.invoke({"question": "上周收入趋势怎么样？", "dry_run": False, "sql_approved": True})
    assert done["query_result"]["status"] == "completed"
    assert len(executed) == 1


def test_graph_dry_run_database_table_count() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "business_data 这个数据库有多少个表？", "dry_run": True})

    assert result["answer"] == "`business_data` 数据库共有 3 个表。"
    assert result["query_plan"] == {"intent": "database_table_count", "database_name": "business_data", "schema_name": None}
    assert result["query_result"]["status"] == "completed"
    assert result["trace"][-1]["step"] == "metadata.dry_run"


def test_graph_dry_run_database_list() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "有哪些数据库？", "dry_run": True})

    assert result["answer"] == "当前可访问的数据库：BigQuery-GA、business_data、product_data"
    assert result["query_result"]["database_count"] == 3


def test_graph_dry_run_database_table_list() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "business_data 这个数据库有哪些表？", "dry_run": True})

    assert result["answer"] == "`business_data` 的表：orders、users、payments"
    assert result["query_result"]["table_count"] == 3


def test_graph_dry_run_bigquery_schema_table_count() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "查询BigQuery-GA 下business_data 这个数据库有多少个表？", "dry_run": True})

    assert result["answer"] == "`BigQuery-GA` 下 `business_data`共有 3 个表。"
    assert result["query_plan"] == {"intent": "database_table_count", "database_name": "BigQuery-GA", "schema_name": "business_data"}
    assert result["query_result"]["tables"] == ["orders", "users", "payments"]


def test_graph_dry_run_bigquery_schema_table_list() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "查询BigQuery-GA 下business_data都有什么表", "dry_run": True})

    assert result["answer"] == "`BigQuery-GA` 下 `business_data` 的表：orders、users、payments"
    assert result["query_result"]["table_count"] == 3


def test_graph_dry_run_unscoped_schema_table_list_stays_metric_query_without_llm() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "查询business_data都有什么表", "dry_run": True})

    assert result["parsed_intent"]["intent"] == "metric_value"
    assert result["query_result"]["status"] == "completed"


def test_graph_dry_run_table_field_list() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "business_data 下orders 这个表有哪些字段？", "dry_run": True})

    assert result["answer"] == "`orders` 表共有 4 个字段：id、created_at、total、user_id"
    assert result["query_result"]["field_count"] == 4


def test_graph_dry_run_table_count_aggregation() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "business_data 下orders 这个表有多少条数据？", "dry_run": True})

    assert result["program"] == {"source": {"type": "table", "id": 11}, "operations": [["aggregate", ["count"]], ["limit", 200]], "preview_sql": "SELECT\n  COUNT(*) AS count\nFROM `business_data.orders`\nLIMIT 200", "execute": False, "requires_approval": True}
    assert result["query_result"]["status"] == "requires_approval"


def test_graph_table_aggregation_review_marks_sql_as_equivalent_preview() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "business_data 下orders 这个表有多少条数据？", "dry_run": True})

    assert result["query_result"]["status"] == "requires_approval"
    # The displayed SQL is NOT what executes (execution goes through Metabase MBQL).
    assert result["query_result"]["preview_only"] is True
    answer = result["answer"]
    assert "等价预览" in answer
    assert ("MBQL" in answer) or ("结构化查询" in answer)


def test_graph_native_sql_review_is_not_marked_preview_only() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "请执行 SELECT 1 AS ok", "dry_run": True})

    assert result["query_result"]["status"] == "requires_approval"
    # Native SQL review runs the exact reviewed SQL — not a preview.
    assert result["query_result"].get("preview_only") is False
    assert "等价预览" not in result["answer"]


def test_graph_dry_run_table_sum_aggregation() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "business_data 下orders 表 total 求和", "dry_run": True})

    assert result["program"]["source"] == {"type": "table", "id": 11}
    assert result["program"]["operations"] == [["aggregate", ["sum", ["field", 3]]], ["limit", 200]]
    assert result["query_result"]["status"] == "requires_approval"
    assert result["query_plan"]["field_name"] == "total"


def test_graph_executes_table_count_aggregation_after_approval() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "business_data 下orders 这个表有多少条数据？", "dry_run": True, "sql_approved": True})

    assert result["program"] == {"source": {"type": "table", "id": 11}, "operations": [["aggregate", ["count"]], ["limit", 200]]}
    assert result["query_result"]["status"] == "completed"


def test_graph_dry_run_table_query_requires_database_confirmation() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "orders 这个表有多少条数据？", "dry_run": True})

    assert result["query_result"]["status"] == "requires_clarification"
    assert result["query_result"]["clarification_type"] == "database"
    assert result["query_result"]["available_databases"] == ["BigQuery-GA", "business_data", "product_data"]
    assert result["query_result"]["suggestions"][0] == "查询BigQuery-GA 下orders 这个表的数据count"


def test_graph_dry_run_missing_table_lists_available_tables() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "business_data 下missing_table 最近7天的每天的数据count", "dry_run": True})

    assert result["query_result"]["status"] == "not_found"
    assert result["query_result"]["clarification_type"] == "table"
    assert result["query_result"]["available_tables"] == ["orders", "users", "payments"]
    assert result["query_result"]["suggestions"][0] == "business_data 下orders 最近7天的每天的数据count"


def test_find_table_matches_singular_business_alias() -> None:
    tables = [{"name": "bll_billing_order"}, {"name": "usr_users"}]

    assert _find_table(tables, "orders") == {"name": "bll_billing_order"}


def test_graph_dry_run_recent_daily_table_count() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "business_data 下orders 最近7天的每天的数据count", "dry_run": True})

    assert result["query_result"]["status"] == "requires_approval"
    assert "SELECT" in result["query_result"]["sql"]
    assert result["program"]["source"] == {"type": "table", "id": 11}
    assert result["program"]["operations"] == [
        ["filter", ["time-interval", ["field", 2], -7, "day"]],
        ["aggregate", ["count"]],
        ["breakout", ["with-temporal-bucket", ["field", 2], "day"]],
        ["order-by", ["with-temporal-bucket", ["field", 2], "day"], "asc"],
        ["limit", 200],
    ]
    assert result["query_plan"]["date_field_name"] == "created_at"
    assert result["query_plan"]["relative_days"] == 7
    assert result["query_plan"]["time_grain"] == "day"


def test_graph_executes_recent_daily_table_count_after_approval() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "business_data 下orders 最近7天的每天的数据count", "dry_run": True, "sql_approved": True})

    assert result["program"] == {
        "source": {"type": "table", "id": 11},
        "operations": [
            ["filter", ["time-interval", ["field", 2], -7, "day"]],
            ["aggregate", ["count"]],
            ["breakout", ["with-temporal-bucket", ["field", 2], "day"]],
            ["order-by", ["with-temporal-bucket", ["field", 2], "day"], "asc"],
            ["limit", 200],
        ],
    }
    assert result["query_plan"]["date_field_name"] == "created_at"
    assert result["query_plan"]["relative_days"] == 7
    assert result["query_plan"]["time_grain"] == "day"
    assert result["query_result"]["row_count"] == 2


def test_infer_database_name_prefers_bigquery_for_schema_queries() -> None:
    databases = [
        {"id": 2, "name": "akool-mongodb-prod"},
        {"id": 19, "name": "BigQuery-GA"},
    ]

    assert _infer_database_name(databases, "business_data") == "BigQuery-GA"


def test_rule_table_aggregation_is_authoritative_when_llm_changes_intent(monkeypatch) -> None:
    monkeypatch.setattr(
        "metabase_agent.agent.graph.parse_intent_with_llm",
        lambda question, settings: {"intent": "metric_trend", "schema_name": "business_data", "table_name": "orders"},
    )
    graph = build_graph(Settings(AGENT_DRY_RUN=False, METABASE_API_KEY="test-key"))

    result = graph.invoke({"question": "business_data 下orders 最近7天的每天的数据count", "dry_run": True})

    assert result["parsed_intent"]["intent"] == "table_aggregation"
    assert result["query_plan"]["table_name"] == "orders"


def test_graph_generates_bigquery_usage_report_sql() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "需要每个月的明细, 202511月到26年4月底, 分web和api 我要在bigquery 中进行数据的统计，给出bigquery 查询汇总语句 fs_results aigc_imagecontents aigc_videocontents", "dry_run": True})

    sql = result["query_result"]["sql"]
    assert result["query_plan"]["intent"] == "bigquery_usage_report_sql"
    assert result["program"]["type"] == "bigquery_sql"
    assert result["program"]["execute"] is False
    assert "`business_data.fs_results`" in sql
    assert "`business_data.open_results`" in sql
    assert "`business_data.aigc_imagecontents`" in sql
    assert "`business_data.open_imagecontents`" in sql
    assert "TIMESTAMP(DATETIME '2025-11-01 00:00:00', 'US/Pacific')" in sql
    assert "TIMESTAMP(DATETIME '2026-05-01 00:00:00', 'US/Pacific')" in sql
    assert "unit" in sql


def test_graph_generates_bigquery_usage_report_sql_with_embedded_sql_examples(monkeypatch) -> None:
    monkeypatch.setattr(
        "metabase_agent.agent.graph.parse_intent_with_llm",
        lambda question, settings: {"intent": "native_sql_query", "database_name": "bigquery", "schema_name": "business_data"},
    )
    graph = build_graph(Settings(AGENT_DRY_RUN=False, METABASE_API_KEY="test-key"))

    result = graph.invoke({"question": """需要每个月的明细, 202511月到26年4月底, 分web和api 我要在bigquery 中进行数据的统计，你要给我对应bigquery 统计语句
web 端详细信息如下：faceswap的结果 其中faceswap_status=3 代表成功的
SELECT
  TIMESTAMP_TRUNC(business_data.fs_results.createdAt, day, 'US/Pacific') AS createdAt,
  COUNT(*) AS count,
  SUM(business_data.fs_results.video_duration) AS sum
FROM business_data.fs_results
WHERE business_data.fs_results.faceswap_status = 3
GROUP BY createdAt
然后看一下 image generate 只有数量了 信息如下
SELECT
  TIMESTAMP_TRUNC(business_data.aigc_imagecontents.create_time, day, 'US/Pacific') AS create_time,
  COUNT(*) AS count
FROM business_data.aigc_imagecontents
根据相关表格和我补充的信息，给出bigquery 查询汇总语句，请先实现bigquery语句，在去查询""", "dry_run": True})

    assert result["parsed_intent"]["intent"] == "bigquery_usage_report_sql"
    assert result["program"]["type"] == "bigquery_sql"
    assert result["program"]["execute"] is False
    assert "metabase.request" not in [event["step"] for event in result["trace"]]


def test_bigquery_report_uses_configured_range_from_settings() -> None:
    graph = build_graph(Settings(
        AGENT_DRY_RUN=True,
        AGENT_REPORT_RANGE_START="2024-01-01",
        AGENT_REPORT_RANGE_END_EXCLUSIVE="2024-07-01",
        AGENT_REPORT_TIMEZONE="UTC",
    ))

    result = graph.invoke({"question": "需要每个月的明细 分web和api fs_results aigc_imagecontents 给出bigquery 查询汇总语句", "dry_run": True})

    assert result["query_plan"]["range_start"] == "2024-01-01"
    assert result["query_plan"]["range_end_exclusive"] == "2024-07-01"
    assert result["query_plan"]["timezone"] == "UTC"
    assert "2024-01-01" in result["query_result"]["sql"]
    assert "US/Pacific" not in result["query_result"]["sql"]


def test_graph_generated_bigquery_usage_report_sql_requires_review_before_execution() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "需要每个月的明细, 202511月到26年4月底, 分web和api 我要在bigquery 中进行数据的统计，给出bigquery 查询汇总语句 fs_results aigc_imagecontents，要执行你优化后的sql,给出最终数据展示的，执行sql", "dry_run": True})

    assert result["query_plan"]["intent"] == "bigquery_usage_report_sql"
    assert result["query_plan"]["requires_approval"] is True
    assert result["program"]["type"] == "bigquery_sql"
    assert result["program"]["execute"] is False
    assert result["query_result"]["status"] == "requires_approval"
    assert "sql.review" in [event["step"] for event in result["trace"]]


def test_graph_executes_generated_bigquery_usage_report_sql_after_approval() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "需要每个月的明细, 202511月到26年4月底, 分web和api 我要在bigquery 中进行数据的统计，给出bigquery 查询汇总语句 fs_results aigc_imagecontents，要执行你优化后的sql,给出最终数据展示的，执行sql", "dry_run": True, "sql_approved": True})

    assert result["query_plan"]["execute"] is True
    assert result["program"]["execute"] is True
    assert result["query_result"]["dry_run"] is True
    assert "bigquery.sql.policy" in [event["step"] for event in result["trace"]]


def test_graph_executes_generated_bigquery_usage_report_sql_when_goal_is_data() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "需要每个月的明细, 202511月到26年4月底, 分web和api 我要在bigquery 中统计 fs_results aigc_imagecontents，并返回数据结果", "dry_run": True})

    assert result["query_plan"]["intent"] == "bigquery_usage_report_sql"
    assert result["query_result"]["status"] == "requires_approval"


def test_graph_executes_generated_bigquery_usage_report_sql_via_metabase(monkeypatch) -> None:
    calls: list[tuple[int, str]] = []

    def execute_native_query(self, database_id: int, sql: str) -> dict[str, object]:
        calls.append((database_id, sql))
        return {"status": "completed", "row_count": 1, "data": {"rows": [["2025-11", "web", "faceswap", None, 10, 100.0, "seconds"]]}}

    monkeypatch.setattr("metabase_agent.agent.graph.parse_intent_with_llm", lambda question, settings: {"intent": "bigquery_usage_report_sql"})
    monkeypatch.setattr("metabase_agent.agent.graph.MetabaseClient.execute_native_query", execute_native_query)
    graph = build_graph(Settings(AGENT_DRY_RUN=False, METABASE_API_KEY="test-key"))

    result = graph.invoke({"question": "需要每个月的明细, 202511月到26年4月底, 分web和api 我要在bigquery 中进行数据的统计，给出bigquery 查询汇总语句 fs_results aigc_imagecontents，自己校验如果语句没问题就自己主动执行语句就行，要执行你最终给的sql,我需要获取最终sql 查询后的数据啊", "dry_run": False, "sql_approved": True})

    assert result["answer"].startswith("已生成并执行 BigQuery")
    assert result["query_result"]["row_count"] == 1
    assert calls and calls[0][0] == 19
    assert calls[0][1] == result["program"]["sql"]
    assert "metabase.request" in [event["step"] for event in result["trace"]]


def test_native_sql_uses_configured_bigquery_database_id() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True, METABASE_BIGQUERY_DATABASE_ID=42))

    result = graph.invoke({"question": "请执行 SELECT 1 AS ok", "dry_run": True})

    assert result["program"]["database_id"] == 42
    assert result["query_plan"]["database_id"] == 42


def test_bigquery_report_executes_against_configured_database_id(monkeypatch) -> None:
    calls: list[tuple[int, str]] = []

    def execute_native_query(self, database_id: int, sql: str) -> dict[str, object]:
        calls.append((database_id, sql))
        return {"status": "completed", "row_count": 0, "data": {"rows": []}}

    monkeypatch.setattr("metabase_agent.agent.graph.parse_intent_with_llm", lambda question, settings: {"intent": "bigquery_usage_report_sql"})
    monkeypatch.setattr("metabase_agent.agent.graph.MetabaseClient.execute_native_query", execute_native_query)
    graph = build_graph(Settings(AGENT_DRY_RUN=False, METABASE_API_KEY="test-key", METABASE_BIGQUERY_DATABASE_ID=77))

    result = graph.invoke({"question": "需要每个月的明细 分web和api fs_results aigc_imagecontents，执行sql 获取最终数据", "dry_run": False, "sql_approved": True})

    assert calls and calls[0][0] == 77
    assert result["program"]["database_id"] == 77


def test_native_sql_executes_against_configured_database_id(monkeypatch) -> None:
    calls: list[tuple[int, str]] = []

    def execute_native_query(self, database_id: int, sql: str) -> dict[str, object]:
        calls.append((database_id, sql))
        return {"status": "completed", "row_count": 0, "data": {"rows": []}}

    monkeypatch.setattr("metabase_agent.agent.graph.parse_intent_with_llm", lambda question, settings: {"intent": "native_sql_query"})
    monkeypatch.setattr("metabase_agent.agent.graph.MetabaseClient.execute_native_query", execute_native_query)
    graph = build_graph(Settings(AGENT_DRY_RUN=False, METABASE_API_KEY="test-key", METABASE_BIGQUERY_DATABASE_ID=88))

    result = graph.invoke({"question": "请执行 SELECT 1 AS ok", "dry_run": False, "sql_approved": True})

    assert calls and calls[0][0] == 88
    assert result["program"]["database_id"] == 88


def test_graph_dry_run_native_sql_query() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "请执行 SELECT 1 AS ok", "dry_run": True})

    assert result["query_plan"] == {"intent": "native_sql_query", "database_name": "BigQuery-GA", "database_id": 19, "requires_approval": True}
    assert result["program"] == {"type": "native_sql", "database_id": 19, "sql": "SELECT 1 AS ok", "execute": False, "requires_approval": True}
    assert result["query_result"]["status"] == "requires_approval"


def test_graph_analyzes_then_requires_review_for_mixed_sql_request() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "SELECT 1 AS ok 分析一下这个sql的含义 并执行sql", "dry_run": True})

    assert result["query_plan"]["intent"] == "native_sql_query"
    assert result["query_result"]["status"] == "requires_approval"
    # Mixed request: structural analysis (dry-run) is shown, then the review prompt.
    assert "客观摘要" in result["answer"]
    assert "授权确认执行" in result["answer"]
    assert "汇率" not in result["answer"]


def test_graph_explains_sql_without_execution_review() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": """WITH invoice_rows AS (
  SELECT invoice, uid, status, amount
  FROM `business_data.fs_chargerecords`
)
SELECT status, COUNT(*) AS invoice_count
FROM invoice_rows
GROUP BY status
分析一下这个sql的含义""", "dry_run": True})

    assert result["query_plan"]["intent"] == "sql_explanation"
    assert result["program"]["execute"] is False
    assert result["query_result"]["status"] == "completed"
    # Structural summary derived from the actual SQL — names the real table and CTE...
    assert "fs_chargerecords" in result["answer"]
    assert "invoice_rows" in result["answer"]
    # ...and never replays the old hardcoded invoice/exchange-rate prose.
    assert "汇率" not in result["answer"]
    assert "JPY" not in result["answer"]


def test_graph_executes_native_sql_query_after_approval() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "请执行 SELECT 1 AS ok", "dry_run": True, "sql_approved": True})

    assert result["program"] == {"type": "native_sql", "database_id": 19, "sql": "SELECT 1 AS ok"}
    assert result["query_result"]["dry_run"] is True


def test_rule_native_sql_is_authoritative_when_llm_changes_intent(monkeypatch) -> None:
    monkeypatch.setattr(
        "metabase_agent.agent.graph.parse_intent_with_llm",
        lambda question, settings: {"intent": "detail_lookup"},
    )
    graph = build_graph(Settings(AGENT_DRY_RUN=False, METABASE_API_KEY="test-key"))

    result = graph.invoke({"question": "SELECT 1 AS ok", "dry_run": True})

    assert result["parsed_intent"]["intent"] == "native_sql_query"
    assert result["program"]["sql"] == "SELECT 1 AS ok"


def test_llm_can_override_native_sql_to_sql_explanation(monkeypatch) -> None:
    monkeypatch.setattr(
        "metabase_agent.agent.graph.parse_intent_with_llm",
        lambda question, settings: {"intent": "sql_explanation"},
    )
    monkeypatch.setattr(
        "metabase_agent.agent.graph.explain_sql_with_llm",
        lambda sql, settings: "（mock LLM 解释）",
    )
    graph = build_graph(Settings(AGENT_DRY_RUN=False, OPENAI_API_KEY="test-key", METABASE_API_KEY="test-key"))

    result = graph.invoke({"question": "SELECT 1 AS ok 这张图理解不太对吧", "dry_run": False})

    assert result["parsed_intent"]["intent"] == "sql_explanation"
    assert result["query_result"]["status"] == "completed"
    assert result["program"]["execute"] is False
    assert "mock LLM" in result["answer"]


def test_graph_dry_run_sql_explanation_describes_pasted_sql_not_hardcoded() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": """SELECT
  TIMESTAMP_TRUNC(create_time, HOUR, 'US/Pacific') AS hour,
  COUNT(DISTINCT user_id) AS active_users
FROM `business_data.aigc_sessions`
WHERE status = 3
GROUP BY hour
ORDER BY hour
分析一下这个sql的含义""", "dry_run": True})

    assert result["query_plan"]["intent"] == "sql_explanation"
    answer = result["answer"]
    assert "business_data.aigc_sessions" in answer
    # The pasted SQL is about sessions/active users, NOT invoices — must not leak the old text.
    assert "发票" not in answer
    assert "汇率" not in answer
    assert "JPY" not in answer


def test_graph_real_mode_sql_explanation_uses_llm(monkeypatch) -> None:
    monkeypatch.setattr(
        "metabase_agent.agent.graph.parse_intent_with_llm",
        lambda question, settings: {"intent": "sql_explanation"},
    )
    monkeypatch.setattr(
        "metabase_agent.agent.graph.explain_sql_with_llm",
        lambda sql, settings: f"LLM解释：包含 {sql.lower().count('select')} 个 SELECT",
    )
    graph = build_graph(Settings(AGENT_DRY_RUN=False, OPENAI_API_KEY="test-key", METABASE_API_KEY="test-key"))

    result = graph.invoke({"question": "SELECT * FROM `business_data.aigc_sessions` 分析含义", "dry_run": False})

    assert result["parsed_intent"]["intent"] == "sql_explanation"
    assert result["answer"].startswith("LLM解释：")


def test_llm_can_override_sql_explanation_to_native_sql_when_user_wants_execution(monkeypatch) -> None:
    monkeypatch.setattr(
        "metabase_agent.agent.graph.parse_intent_with_llm",
        lambda question, settings: {"intent": "native_sql_query"},
    )
    graph = build_graph(Settings(AGENT_DRY_RUN=False, OPENAI_API_KEY="test-key", METABASE_API_KEY="test-key"))

    result = graph.invoke({"question": "SELECT 1 AS ok 分析后如果没问题就执行", "dry_run": True})

    assert result["parsed_intent"]["intent"] == "native_sql_query"
    assert result["query_result"]["status"] == "requires_approval"


def test_graph_native_sql_normalizes_metabase_style_query() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "SELECT business_data.fs_results.createdAt FROM business_data.fs_results ORDER BY business_data.fs_results.createdAt ASC  执行获取数据", "dry_run": True})

    assert result["program"]["sql"] == "SELECT `business_data.fs_results`.createdAt FROM `business_data.fs_results` ORDER BY `business_data.fs_results`.createdAt ASC"


def test_graph_native_sql_trims_noisy_user_suffix() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "Past执行如下sql\nSELECT business_data.fs_results.createdAt FROM business_data.fs_results ORDER BY business_data.fs_results.createdAt AS道不会先用llm去识别这个sql吗", "dry_run": True})

    assert result["program"]["sql"] == "SELECT `business_data.fs_results`.createdAt FROM `business_data.fs_results` ORDER BY `business_data.fs_results`.createdAt"


def test_graph_blocks_extra_native_sql_statement() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "SELECT 1; SELECT 2", "dry_run": True})

    assert result["query_result"]["status"] == "blocked"


def test_graph_blocks_non_read_only_native_sql() -> None:
    graph = build_graph(Settings(AGENT_DRY_RUN=True))

    result = graph.invoke({"question": "DROP TABLE users", "dry_run": True})

    assert result["query_result"]["status"] != "completed"
