-- BigQuery Standard SQL
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
    CASE type
    WHEN 4 THEN 'Talking Avatar'
    WHEN 5 THEN 'TalkingPhoto'
    WHEN 8 THEN 'VideoTranslate'
    WHEN 15 THEN 'ImageToVideo'
    WHEN 21 THEN 'CharacterFaceswap'
    WHEN 98 THEN 'Headswap'
    ELSE CONCAT('type_', CAST(type AS STRING))
  END AS subtype,
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
    CASE type
    WHEN 4 THEN 'Talking Avatar'
    WHEN 5 THEN 'TalkingPhoto'
    WHEN 8 THEN 'VideoTranslate'
    WHEN 15 THEN 'ImageToVideo'
    WHEN 21 THEN 'CharacterFaceswap'
    WHEN 98 THEN 'Headswap'
    ELSE CONCAT('type_', CAST(type AS STRING))
  END AS subtype,
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
ORDER BY month, channel, product, subtype;
