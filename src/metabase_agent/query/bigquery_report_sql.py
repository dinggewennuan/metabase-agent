from __future__ import annotations

import re

FORBIDDEN_SQL_WORDS = {
    "ALTER",
    "CALL",
    "CREATE",
    "DECLARE",
    "DELETE",
    "DROP",
    "EXECUTE",
    "EXPORT",
    "GRANT",
    "INSERT",
    "LOAD",
    "MERGE",
    "REPLACE",
    "REVOKE",
    "TRUNCATE",
    "UPDATE",
}


VIDEO_TYPE_CASE = """CASE type
    WHEN 4 THEN 'Talking Avatar'
    WHEN 5 THEN 'TalkingPhoto'
    WHEN 8 THEN 'VideoTranslate'
    WHEN 15 THEN 'ImageToVideo'
    WHEN 21 THEN 'CharacterFaceswap'
    WHEN 98 THEN 'Headswap'
    ELSE CONCAT('type_', CAST(type AS STRING))
  END"""


def build_monthly_usage_report_sql(
    start_date: str = "2025-11-01",
    end_date_exclusive: str = "2026-05-01",
    timezone: str = "US/Pacific",
) -> str:
    return f"""-- BigQuery Standard SQL
-- Monthly web/api usage details from {start_date} (inclusive) to {end_date_exclusive} (exclusive), {timezone}.
-- unit = 'seconds' means duration_seconds is available; unit = 'count' means only generated_count is meaningful.
WITH params AS (
  SELECT
    TIMESTAMP(DATETIME '{start_date} 00:00:00', '{timezone}') AS start_ts,
    TIMESTAMP(DATETIME '{end_date_exclusive} 00:00:00', '{timezone}') AS end_ts
),
web_faceswap AS (
  SELECT
    TIMESTAMP_TRUNC(r.createdAt, MONTH, '{timezone}') AS month,
    'web' AS channel,
    'faceswap' AS product,
    CAST(NULL AS STRING) AS subtype,
    COUNT(*) AS generated_count,
    SUM(CAST(r.video_duration AS FLOAT64)) AS duration_seconds,
    'seconds' AS unit
  FROM `business_data.fs_results` AS r
  CROSS JOIN params AS p
  WHERE r.createdAt >= p.start_ts
    AND r.createdAt < p.end_ts
    AND r.faceswap_status = 3
  GROUP BY month, channel, product, subtype, unit
),
api_faceswap AS (
  SELECT
    TIMESTAMP_TRUNC(r.createdAt, MONTH, '{timezone}') AS month,
    'api' AS channel,
    'faceswap' AS product,
    CAST(NULL AS STRING) AS subtype,
    COUNT(*) AS generated_count,
    SUM(CAST(r.video_duration AS FLOAT64)) AS duration_seconds,
    'seconds' AS unit
  FROM `business_data.open_results` AS r
  CROSS JOIN params AS p
  WHERE r.createdAt >= p.start_ts
    AND r.createdAt < p.end_ts
    AND r.faceswap_status = 3
  GROUP BY month, channel, product, subtype, unit
),
web_image_generate AS (
  SELECT
    TIMESTAMP_TRUNC(i.create_time, MONTH, '{timezone}') AS month,
    'web' AS channel,
    'image_generate' AS product,
    CAST(NULL AS STRING) AS subtype,
    COUNT(*) AS generated_count,
    CAST(NULL AS FLOAT64) AS duration_seconds,
    'count' AS unit
  FROM `business_data.aigc_imagecontents` AS i
  CROSS JOIN params AS p
  WHERE i.create_time >= p.start_ts
    AND i.create_time < p.end_ts
    AND i.image_status IN (3, 4)
  GROUP BY month, channel, product, subtype, unit
),
api_image_generate AS (
  SELECT
    TIMESTAMP_TRUNC(i.create_time, MONTH, '{timezone}') AS month,
    'api' AS channel,
    'image_generate' AS product,
    CAST(NULL AS STRING) AS subtype,
    COUNT(*) AS generated_count,
    CAST(NULL AS FLOAT64) AS duration_seconds,
    'count' AS unit
  FROM `business_data.open_imagecontents` AS i
  CROSS JOIN params AS p
  WHERE i.create_time >= p.start_ts
    AND i.create_time < p.end_ts
    AND i.image_status IN (3, 4)
  GROUP BY month, channel, product, subtype, unit
),
web_video AS (
  SELECT
    TIMESTAMP_TRUNC(v.create_time, MONTH, '{timezone}') AS month,
    'web' AS channel,
    'video' AS product,
    {VIDEO_TYPE_CASE} AS subtype,
    COUNT(*) AS generated_count,
    SUM(CAST(v.video_duration AS FLOAT64)) AS duration_seconds,
    'seconds' AS unit
  FROM `business_data.aigc_videocontents` AS v
  CROSS JOIN params AS p
  WHERE v.create_time >= p.start_ts
    AND v.create_time < p.end_ts
    AND v.video_status = 3
    AND v.type IN (4, 5, 8, 15, 21, 98)
  GROUP BY month, channel, product, subtype, unit
),
api_video AS (
  SELECT
    TIMESTAMP_TRUNC(v.create_time, MONTH, '{timezone}') AS month,
    'api' AS channel,
    'video' AS product,
    {VIDEO_TYPE_CASE} AS subtype,
    COUNT(*) AS generated_count,
    SUM(CAST(v.video_duration AS FLOAT64)) AS duration_seconds,
    'seconds' AS unit
  FROM `business_data.open_videocontents` AS v
  CROSS JOIN params AS p
  WHERE v.create_time >= p.start_ts
    AND v.create_time < p.end_ts
    AND v.video_status = 3
    AND v.type IN (4, 5, 8, 15, 21, 98)
  GROUP BY month, channel, product, subtype, unit
),
web_streaming_avatar AS (
  SELECT
    TIMESTAMP_TRUNC(s.create_time, MONTH, '{timezone}') AS month,
    'web' AS channel,
    'streaming_avatar' AS product,
    CAST(NULL AS STRING) AS subtype,
    COUNT(*) AS generated_count,
    SUM(CAST(s.duration AS FLOAT64)) AS duration_seconds,
    'seconds' AS unit
  FROM `business_data.aigc_sessions` AS s
  CROSS JOIN params AS p
  WHERE s.create_time >= p.start_ts
    AND s.create_time < p.end_ts
    AND s.status = 3
  GROUP BY month, channel, product, subtype, unit
),
api_streaming_avatar AS (
  SELECT
    TIMESTAMP_TRUNC(s.create_time, MONTH, '{timezone}') AS month,
    'api' AS channel,
    'streaming_avatar' AS product,
    CAST(NULL AS STRING) AS subtype,
    COUNT(*) AS generated_count,
    SUM(CAST(s.duration AS FLOAT64)) AS duration_seconds,
    'seconds' AS unit
  FROM `business_data.open_sessions` AS s
  CROSS JOIN params AS p
  WHERE s.create_time >= p.start_ts
    AND s.create_time < p.end_ts
    AND s.status = 3
  GROUP BY month, channel, product, subtype, unit
),
web_voice_lab AS (
  SELECT
    TIMESTAMP_TRUNC(a.create_time, MONTH, '{timezone}') AS month,
    'web' AS channel,
    'voice_lab' AS product,
    CAST(NULL AS STRING) AS subtype,
    COUNT(*) AS generated_count,
    SUM(CAST(a.duration AS FLOAT64)) AS duration_seconds,
    'seconds' AS unit
  FROM `business_data.aigc_audiocontents` AS a
  CROSS JOIN params AS p
  WHERE a.create_time >= p.start_ts
    AND a.create_time < p.end_ts
    AND a.status = 3
  GROUP BY month, channel, product, subtype, unit
),
api_voice_lab AS (
  SELECT
    TIMESTAMP_TRUNC(a.create_time, MONTH, '{timezone}') AS month,
    'api' AS channel,
    'voice_lab' AS product,
    CAST(NULL AS STRING) AS subtype,
    COUNT(*) AS generated_count,
    SUM(CAST(COALESCE(a.duration, a.audio_duration) AS FLOAT64)) AS duration_seconds,
    'seconds' AS unit
  FROM `business_data.open_audiocontents` AS a
  CROSS JOIN params AS p
  WHERE a.create_time >= p.start_ts
    AND a.create_time < p.end_ts
    AND a.status = 3
  GROUP BY month, channel, product, subtype, unit
)
SELECT
  FORMAT_TIMESTAMP('%Y-%m', month, '{timezone}') AS month,
  channel,
  product,
  subtype,
  generated_count,
  duration_seconds,
  unit
FROM (
  SELECT * FROM web_faceswap
  UNION ALL SELECT * FROM api_faceswap
  UNION ALL SELECT * FROM web_image_generate
  UNION ALL SELECT * FROM api_image_generate
  UNION ALL SELECT * FROM web_video
  UNION ALL SELECT * FROM api_video
  UNION ALL SELECT * FROM web_streaming_avatar
  UNION ALL SELECT * FROM api_streaming_avatar
  UNION ALL SELECT * FROM web_voice_lab
  UNION ALL SELECT * FROM api_voice_lab
)
ORDER BY month, channel, product, subtype;"""


def extract_native_sql(text: str) -> str | None:
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return normalize_bigquery_sql(_clean_sql(_truncate_sql_candidate(fenced.group(1))))
    match = re.search(r"\b(WITH|SELECT)\b[\s\S]*", text, re.IGNORECASE)
    if match:
        return normalize_bigquery_sql(_clean_sql(_truncate_sql_candidate(match.group(0))))
    return None


def normalize_bigquery_sql(sql: str) -> str:
    normalized = _remove_trailing_non_sql_text(sql)
    normalized = re.sub(r"(?i)\b(FROM|JOIN)\s+business_data\.([A-Za-z_][A-Za-z0-9_]*)", r"\1 `business_data.\2`", normalized)
    normalized = re.sub(
        r"(?<!`)\bbusiness_data\.([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b",
        r"`business_data.\1`.\2",
        normalized,
    )
    return normalized


def is_read_only_sql(sql: str) -> bool:
    normalized = _strip_leading_comments(sql).strip()
    if not re.match(r"^(WITH|SELECT)\b", normalized, re.IGNORECASE):
        return False
    if _has_extra_statement(normalized):
        return False
    tokens = {token.upper() for token in re.findall(r"\b[A-Za-z_]+\b", _mask_sql_literals_comments_and_identifiers(normalized))}
    return not bool(tokens & FORBIDDEN_SQL_WORDS)


def _clean_sql(sql: str) -> str:
    cleaned = sql.strip()
    return cleaned[:-1].strip() if cleaned.endswith(";") else cleaned


def _truncate_sql_candidate(sql: str) -> str:
    lines = sql.strip().splitlines()
    cleaned_lines: list[str] = []
    for line in lines:
        cleaned_line = line.rstrip()
        if _contains_cjk(cleaned_line):
            cleaned_line = re.split(r"[\u4e00-\u9fff]", cleaned_line, maxsplit=1)[0].rstrip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
            break
        cleaned_lines.append(cleaned_line)
    return _trim_incomplete_sql_tail("\n".join(cleaned_lines).strip())


def _remove_trailing_non_sql_text(sql: str) -> str:
    return _truncate_sql_candidate(sql)


def _trim_incomplete_sql_tail(sql: str) -> str:
    trimmed = sql.strip()
    while True:
        next_trimmed = re.sub(r"(?i)\s+(?:AS|AND|OR|WHERE|HAVING|LIMIT|OFFSET)\s*$", "", trimmed).rstrip()
        next_trimmed = re.sub(r"(?i)\s+GROUP\s+BY\s*$", "", next_trimmed).rstrip()
        next_trimmed = re.sub(r"(?i)\s+ORDER\s+BY\s*$", "", next_trimmed).rstrip()
        if next_trimmed == trimmed:
            return trimmed
        trimmed = next_trimmed


def _has_extra_statement(sql: str) -> bool:
    for index, char in enumerate(sql):
        if char != ";" or _is_inside_sql_literal_or_comment(sql, index):
            continue
        if sql[index + 1 :].strip():
            return True
    return False


def _mask_sql_literals_comments_and_identifiers(sql: str) -> str:
    chars = list(sql)
    index = 0
    while index < len(chars):
        if sql.startswith("--", index):
            end = sql.find("\n", index)
            end = len(chars) if end == -1 else end
            for mask_index in range(index, end):
                chars[mask_index] = " "
            index = end
            continue
        if sql.startswith("/*", index):
            end = sql.find("*/", index + 2)
            end = len(chars) if end == -1 else end + 2
            for mask_index in range(index, end):
                chars[mask_index] = " "
            index = end
            continue
        if chars[index] in {"'", '"', "`"}:
            quote = chars[index]
            end = _quoted_end(sql, index, quote)
            for mask_index in range(index, end):
                chars[mask_index] = " "
            index = end
            continue
        index += 1
    return "".join(chars)


def _is_inside_sql_literal_or_comment(sql: str, target_index: int) -> bool:
    return _mask_sql_literals_comments_and_identifiers(sql[: target_index + 1])[-1] == " "


def _quoted_end(sql: str, start: int, quote: str) -> int:
    index = start + 1
    while index < len(sql):
        if sql[index] == quote:
            if quote == "'" and index + 1 < len(sql) and sql[index + 1] == "'":
                index += 2
                continue
            return index + 1
        index += 1
    return len(sql)



def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _strip_leading_comments(sql: str) -> str:
    cleaned = sql.lstrip()
    while True:
        if cleaned.startswith("--"):
            _, _, cleaned = cleaned.partition("\n")
            cleaned = cleaned.lstrip()
            continue
        if cleaned.startswith("/*"):
            _, end, rest = cleaned.partition("*/")
            if not end:
                return ""
            cleaned = rest.lstrip()
            continue
        return cleaned
