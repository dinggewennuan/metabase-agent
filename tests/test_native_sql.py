from metabase_agent.query.bigquery_report_sql import (
    extract_native_sql,
    is_read_only_sql,
    normalize_bigquery_sql,
)


def test_extract_native_sql_from_plain_text() -> None:
    assert extract_native_sql("请执行 SELECT 1 AS ok;") == "SELECT 1 AS ok"


def test_extract_native_sql_from_fence() -> None:
    assert extract_native_sql("```sql\nSELECT 1 AS ok;\n```") == "SELECT 1 AS ok"


def test_read_only_sql_allows_select_and_with() -> None:
    assert is_read_only_sql("SELECT 1") is True
    assert is_read_only_sql("WITH x AS (SELECT 1) SELECT * FROM x") is True


def test_read_only_sql_blocks_mutations() -> None:
    assert is_read_only_sql("DELETE FROM table") is False
    assert is_read_only_sql("SELECT 1; DROP TABLE users") is False


def test_read_only_sql_ignores_forbidden_words_in_literals_and_identifiers() -> None:
    assert is_read_only_sql("SELECT 'DROP TABLE users' AS message") is True
    assert is_read_only_sql("SELECT `update` FROM `business_data.events`") is True


def test_extract_native_sql_removes_trailing_chinese_instruction() -> None:
    assert extract_native_sql("SELECT 1 AS ok  执行获取数据") == "SELECT 1 AS ok"


def test_extract_native_sql_removes_noisy_corrupted_user_suffix() -> None:
    question = """Past执行如下sql
SELECT
  TIMESTAMP_TRUNC(
    business_data.fs_results.createdAt,
    day,
    'US/Pacific'
  ) AS createdAt,
  COUNT(*) AS count,
  SUM(business_data.fs_results.video_duration) AS sum
FROM
  business_data.fs_results
WHERE
  (
    business_data.fs_results.createdAt >= TIMESTAMP_TRUNC(
      TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL -7 day),
      day,
      'US/Pacific'
    )
  )
GROUP BY
  createdAt
ORDER BY
  createdAt AS道不会先用llm去识别这个sql吗"""

    assert extract_native_sql(question) == """SELECT
  TIMESTAMP_TRUNC(
    `business_data.fs_results`.createdAt,
    day,
    'US/Pacific'
  ) AS createdAt,
  COUNT(*) AS count,
  SUM(`business_data.fs_results`.video_duration) AS sum
FROM `business_data.fs_results`
WHERE
  (
    `business_data.fs_results`.createdAt >= TIMESTAMP_TRUNC(
      TIMESTAMP_ADD(CURRENT_TIMESTAMP(), INTERVAL -7 day),
      day,
      'US/Pacific'
    )
  )
GROUP BY
  createdAt
ORDER BY
  createdAt"""


def test_extract_native_sql_prefers_fenced_block_over_surrounding_text() -> None:
    assert extract_native_sql("先跑这个\n```sql\nSELECT 1 AS ok;\n```\n然后给我结果") == "SELECT 1 AS ok"


def test_read_only_sql_blocks_extra_statement_after_select() -> None:
    assert is_read_only_sql("SELECT 1; SELECT 2") is False
    assert is_read_only_sql("SELECT ';' AS semicolon; DROP TABLE users") is False
    assert is_read_only_sql("SELECT ';' AS semicolon") is True


def test_normalize_bigquery_sql_quotes_metabase_style_table_paths() -> None:
    sql = normalize_bigquery_sql("SELECT business_data.fs_results.createdAt FROM business_data.fs_results WHERE business_data.fs_results.faceswap_status = 3")

    assert sql == "SELECT `business_data.fs_results`.createdAt FROM `business_data.fs_results` WHERE `business_data.fs_results`.faceswap_status = 3"


def test_read_only_sql_blocks_bigquery_scripting_and_export() -> None:
    assert is_read_only_sql("CALL my_dataset.my_procedure()") is False
    assert is_read_only_sql("EXPORT DATA OPTIONS(uri='gs://x/*') AS SELECT 1") is False
    assert is_read_only_sql("LOAD DATA INTO t FROM FILES(uris=['gs://x'])") is False
    assert is_read_only_sql("SELECT 1; EXECUTE IMMEDIATE 'DROP TABLE users'") is False
    assert is_read_only_sql("WITH x AS (SELECT 1) SELECT * FROM x; CALL p()") is False


def test_extract_native_sql_keeps_chinese_comment_and_following_lines() -> None:
    # A Chinese comment must not truncate the SQL: the mutilated statement
    # (WHERE/LIMIT dropped) would still pass the read-only check and run a
    # different query than the one the user approved.
    sql = "SELECT count(*) FROM `business_data.orders`\n-- 只看最近7天\nWHERE created_at > '2026-01-01'\nLIMIT 10"

    extracted = extract_native_sql(sql)

    assert extracted is not None
    assert "只看最近7天" in extracted
    assert "WHERE created_at > '2026-01-01'" in extracted
    assert "LIMIT 10" in extracted


def test_extract_native_sql_keeps_chinese_string_literal() -> None:
    assert extract_native_sql("SELECT * FROM t WHERE city = '北京' LIMIT 5") == "SELECT * FROM t WHERE city = '北京' LIMIT 5"


def test_extract_native_sql_still_strips_prose_after_chinese_literal() -> None:
    extracted = extract_native_sql("SELECT * FROM t WHERE city = '北京' LIMIT 5 帮我看看这个结果")

    assert extracted == "SELECT * FROM t WHERE city = '北京' LIMIT 5"
