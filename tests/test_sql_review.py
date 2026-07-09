from __future__ import annotations

from metabase_agent.agent.metadata import _first_numeric_field
from metabase_agent.agent.sql_review import (
    approved_program_mismatch,
    program_fingerprint,
)


def test_program_fingerprint_ignores_presentation_flags() -> None:
    program = {"type": "native_sql", "database_id": 19, "sql": "SELECT 1"}
    reviewed = {**program, "execute": False, "requires_approval": True, "preview_sql": "SELECT 1"}

    assert program_fingerprint(program) == program_fingerprint(reviewed)


def test_program_fingerprint_changes_with_content() -> None:
    base = {"type": "native_sql", "database_id": 19, "sql": "SELECT 1"}

    assert program_fingerprint(base) != program_fingerprint({**base, "sql": "SELECT 2"})
    assert program_fingerprint(base) != program_fingerprint({**base, "database_id": 7})
    assert program_fingerprint(None) is None
    assert program_fingerprint({}) is None


def test_approved_program_mismatch_detects_drift() -> None:
    program = {"source": {"type": "table", "id": 1}, "operations": [["aggregate", "count"]]}
    state = {"sql_approved": True, "approved_program_hash": program_fingerprint(program)}

    # An approval only authorizes the exact query that was reviewed.
    assert approved_program_mismatch(state, program) is False
    assert approved_program_mismatch(state, {**program, "operations": [["aggregate", "sum"]]}) is True
    # No stored hash (legacy pending approval) — do not block.
    assert approved_program_mismatch({"sql_approved": True}, program) is False


def test_first_numeric_field_skips_primary_and_foreign_keys() -> None:
    fields = [
        {"name": "id", "base_type": "type/BigInteger", "semantic_type": "type/PK"},
        {"name": "user_id", "base_type": "type/Integer", "semantic_type": "type/FK"},
        {"name": "amount", "base_type": "type/Float"},
    ]

    picked = _first_numeric_field(fields)

    # sum(id)/avg(id) is a meaningless answer — keys must never be auto-picked.
    assert picked is not None
    assert picked["name"] == "amount"


def test_first_numeric_field_returns_none_when_only_keys_exist() -> None:
    fields = [{"name": "id", "base_type": "type/BigInteger", "semantic_type": "type/PK"}]

    assert _first_numeric_field(fields) is None
