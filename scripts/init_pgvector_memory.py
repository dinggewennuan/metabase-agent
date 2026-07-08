from __future__ import annotations

import sys

from metabase_agent.config.settings import get_settings
from metabase_agent.memory.vector import _validate_identifier


def main() -> int:
    settings = get_settings()
    if not settings.agent_pgvector_dsn:
        print("AGENT_PGVECTOR_DSN is required", file=sys.stderr)
        return 2
    if settings.agent_embedding_dimensions < 1:
        print("AGENT_EMBEDDING_DIMENSIONS must be greater than 0", file=sys.stderr)
        return 2

    try:
        import psycopg
    except ImportError:
        print("psycopg is required. Install project dependencies first.", file=sys.stderr)
        return 2

    table = _validate_identifier(settings.agent_pgvector_table)
    vector_index = _validate_identifier(f"{table}_vector_idx")
    filter_index = _validate_identifier(f"{table}_filter_idx")
    dimensions = settings.agent_embedding_dimensions

    statements = [
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"""
        CREATE TABLE IF NOT EXISTS {table} (
          id text PRIMARY KEY,
          tenant_id text NOT NULL,
          user_id text NOT NULL,
          scope text NOT NULL,
          memory_type text NOT NULL,
          memory_id text NOT NULL,
          content text NOT NULL,
          embedding vector({dimensions}) NOT NULL,
          metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
          status text NOT NULL,
          created_at timestamptz NOT NULL,
          updated_at timestamptz NOT NULL
        )
        """,
        f"""
        CREATE INDEX IF NOT EXISTS {vector_index}
        ON {table}
        USING hnsw (embedding vector_cosine_ops)
        """,
        f"""
        CREATE INDEX IF NOT EXISTS {filter_index}
        ON {table} (tenant_id, user_id, memory_type, status)
        """,
    ]
    with psycopg.connect(settings.agent_pgvector_dsn) as conn:
        for statement in statements:
            conn.execute(statement)

    print(f"initialized pgvector memory table {table} with vector({dimensions})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
