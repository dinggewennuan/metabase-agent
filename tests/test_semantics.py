from metabase_agent.semantics.intent_parser import (
    is_safe_rule_intent_override,
    parse_intent,
    wants_sql_explanation,
)


def test_parse_chinese_revenue_trend() -> None:
    parsed = parse_intent("上周收入趋势怎么样？")

    assert parsed["intent"] == "metric_trend"
    assert "revenue" in parsed["business_terms"]
    assert parsed["time_grain"] == "day"


def test_parse_database_table_count() -> None:
    parsed = parse_intent("business_data 这个数据库有多少个表？")

    assert parsed["intent"] == "database_table_count"
    assert parsed["database_name"] == "business_data"


def test_parse_bigquery_schema_table_count() -> None:
    parsed = parse_intent("查询BigQuery-GA 下business_data 这个数据库有多少个表？")

    assert parsed["intent"] == "database_table_count"
    assert parsed["database_name"] == "BigQuery-GA"
    assert parsed["schema_name"] == "business_data"


def test_parse_bigquery_schema_table_list() -> None:
    parsed = parse_intent("查询BigQuery-GA 下business_data都有什么表")

    assert parsed["intent"] == "database_table_list"
    assert parsed["database_name"] == "BigQuery-GA"
    assert parsed["schema_name"] == "business_data"


def test_parse_database_list() -> None:
    parsed = parse_intent("有哪些数据库？")

    assert parsed["intent"] == "database_list"


def test_parse_table_field_list() -> None:
    parsed = parse_intent("orders 这个表有哪些字段？")

    assert parsed["intent"] == "table_field_list"
    assert parsed["table_name"] == "orders"


def test_parse_table_count_aggregation() -> None:
    parsed = parse_intent("orders 这个表有多少条数据？")

    assert parsed["intent"] == "table_aggregation"
    assert parsed["table_name"] == "orders"
    assert parsed["aggregation_function"] == "count"


def test_parse_table_sum_aggregation() -> None:
    parsed = parse_intent("orders 表 total 求和")

    assert parsed["intent"] == "table_aggregation"
    assert parsed["table_name"] == "orders"
    assert parsed["field_name"] == "total"
    assert parsed["aggregation_function"] == "sum"


def test_parse_schema_table_recent_daily_count() -> None:
    parsed = parse_intent("business_data 下fs_times 最近7天的每天的数据count")

    assert parsed["intent"] == "table_aggregation"
    assert parsed["schema_name"] == "business_data"
    assert parsed["table_name"] == "fs_times"
    assert parsed["aggregation_function"] == "count"
    assert parsed["relative_days"] == 7
    assert parsed["time_grain"] == "day"


def test_parse_schema_table_recent_daily_count_with_spaced_identifier() -> None:
    parsed = parse_intent("business_data 下 fs results最近7天的每天的数据count ,并分析一下最近2天数量是否增长，以及哪部分有了增长啊")

    assert parsed["intent"] == "table_aggregation"
    assert parsed["schema_name"] == "business_data"
    assert parsed["table_name"] == "fs_results"
    assert parsed["aggregation_function"] == "count"
    assert parsed["relative_days"] == 7
    assert parsed["time_grain"] == "day"


def test_parse_table_name_with_table_suffix_word_inside_identifier() -> None:
    parsed = parse_intent("business_data 下missing_table 最近7天的每天的数据count")

    assert parsed["table_name"] == "missing_table"


def test_parse_bigquery_usage_report_sql_request() -> None:
    parsed = parse_intent("需要每个月的明细, 202511月到26年4月底, 分web和api 我要在bigquery 中进行数据的统计，给出bigquery 查询汇总语句 fs_results aigc_imagecontents")

    assert parsed["intent"] == "bigquery_usage_report_sql"


def test_parse_bigquery_usage_report_sql_request_with_embedded_examples() -> None:
    parsed = parse_intent("""需要每个月的明细, 202511月到26年4月底, 分web和api 我要在bigquery 中进行数据的统计，给出bigquery 查询汇总语句
web 端详细信息如下：faceswap的结果 其中faceswap_status=3 代表成功的
SELECT business_data.fs_results.createdAt, COUNT(*) AS count
FROM business_data.fs_results
WHERE business_data.fs_results.faceswap_status = 3
GROUP BY business_data.fs_results.createdAt
然后看一下 image generate 只有数量了 信息如下
SELECT business_data.aigc_imagecontents.create_time, COUNT(*) AS count
FROM business_data.aigc_imagecontents
根据相关表格和我补充的信息，给出bigquery 查询汇总语句，请先实现bigquery语句，在去查询""")

    assert parsed["intent"] == "bigquery_usage_report_sql"


def test_parse_native_sql_query() -> None:
    parsed = parse_intent("请执行 SELECT 1 AS ok")

    assert parsed["intent"] == "native_sql_query"


def test_parse_sql_explanation_request() -> None:
    parsed = parse_intent("WITH x AS (SELECT 1 AS ok) SELECT * FROM x 分析一下这个sql的含义")

    assert parsed["intent"] == "sql_explanation"


def test_parse_mixed_sql_explanation_and_execution_as_native_sql() -> None:
    parsed = parse_intent("SELECT 1 AS ok 分析一下这个sql的含义 并执行sql")

    assert parsed["intent"] == "native_sql_query"
    assert wants_sql_explanation("SELECT 1 AS ok 分析一下这个sql的含义 并执行sql") is True


def test_sql_with_timestamp_trunc_and_analysis_is_explanation_not_execution() -> None:
    # "run" is a substring of TIMESTAMP_TRUNC; it must not be read as an execution request.
    parsed = parse_intent("SELECT TIMESTAMP_TRUNC(create_time, HOUR) AS h, COUNT(*) FROM `t` GROUP BY h 分析一下含义")

    assert parsed["intent"] == "sql_explanation"


def test_safe_rule_intent_override_allows_sql_explain_execute_split() -> None:
    assert is_safe_rule_intent_override("native_sql_query", "sql_explanation") is True
    assert is_safe_rule_intent_override("sql_explanation", "native_sql_query") is True
    assert is_safe_rule_intent_override("native_sql_query", "database_table_list") is False


def test_latin_keywords_do_not_match_inside_identifiers() -> None:
    # "accounts"/"discount" contain "count", "summary" contains "sum" —
    # substring matching misrouted field-list questions into aggregations.
    parsed = parse_intent("accounts 这个表有哪些字段")
    assert parsed["intent"] == "table_field_list"
    assert parsed["aggregation_function"] is None

    parsed = parse_intent("discount 这个表有哪些字段")
    assert parsed["intent"] == "table_field_list"
    assert parsed["aggregation_function"] is None


def test_avg_keyword_not_shadowed_by_summary_identifier() -> None:
    from metabase_agent.semantics.intent_parser import extract_aggregation_function

    assert extract_aggregation_function("summary 表 求平均") == "avg"


def test_count_still_matches_after_cjk_text() -> None:
    from metabase_agent.semantics.intent_parser import extract_aggregation_function

    assert extract_aggregation_function("最近7天的每天的数据count") == "count"
