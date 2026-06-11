from metabase_agent.query.bigquery_report_sql import build_monthly_usage_report_sql


def test_monthly_usage_report_sql_contains_all_required_branches() -> None:
    sql = build_monthly_usage_report_sql()

    assert "web_faceswap" in sql
    assert "api_faceswap" in sql
    assert "web_image_generate" in sql
    assert "api_image_generate" in sql
    assert "web_video" in sql
    assert "api_video" in sql
    assert "web_streaming_avatar" in sql
    assert "api_streaming_avatar" in sql
    assert "web_voice_lab" in sql
    assert "api_voice_lab" in sql
    assert "image_status IN (3, 4)" in sql
    assert "video_status = 3" in sql
    assert "faceswap_status = 3" in sql
    assert "status = 3" in sql
    assert "WHEN 4 THEN 'Talking Avatar'" in sql
    assert "WHEN 98 THEN 'Headswap'" in sql
    assert "'count' AS unit" in sql
    assert "'seconds' AS unit" in sql


def test_monthly_usage_report_sql_defaults_unchanged() -> None:
    sql = build_monthly_usage_report_sql()

    assert "TIMESTAMP(DATETIME '2025-11-01 00:00:00', 'US/Pacific')" in sql
    assert "TIMESTAMP(DATETIME '2026-05-01 00:00:00', 'US/Pacific')" in sql


def test_monthly_usage_report_sql_accepts_custom_range_and_timezone() -> None:
    sql = build_monthly_usage_report_sql(start_date="2024-01-01", end_date_exclusive="2024-07-01", timezone="UTC")

    assert "TIMESTAMP(DATETIME '2024-01-01 00:00:00', 'UTC')" in sql
    assert "TIMESTAMP(DATETIME '2024-07-01 00:00:00', 'UTC')" in sql
    assert "US/Pacific" not in sql
    assert "2025-11-01" not in sql
