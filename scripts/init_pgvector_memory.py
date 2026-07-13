from __future__ import annotations

import sys

from metabase_agent.config.settings import get_settings
from metabase_agent.memory.vector import PgVectorIndex


def main() -> int:
    settings = get_settings()
    if not settings.agent_pgvector_dsn:
        print("AGENT_PGVECTOR_DSN is required", file=sys.stderr)
        return 2
    if settings.agent_embedding_dimensions < 1:
        print("AGENT_EMBEDDING_DIMENSIONS must be greater than 0", file=sys.stderr)
        return 2

    try:
        import psycopg  # noqa: F401
    except ImportError:
        print("psycopg is required. Install project dependencies first.", file=sys.stderr)
        return 2

    # The app auto-creates this on startup too; the script stays as an explicit,
    # idempotent way to provision (or re-provision) the table out of band.
    index = PgVectorIndex(
        settings.agent_pgvector_dsn,
        table=settings.agent_pgvector_table,
        dimensions=settings.agent_embedding_dimensions,
    )
    index.ensure_schema()
    print(f"initialized pgvector memory table {settings.agent_pgvector_table} with vector({settings.agent_embedding_dimensions})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
