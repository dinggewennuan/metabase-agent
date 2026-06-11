from __future__ import annotations

from typing import Any, TypedDict


class PolicyResult(TypedDict):
    allowed: bool
    reason: str | None


def check_program(program: dict[str, Any]) -> PolicyResult:
    operations = program.get("operations", [])
    for operation in operations:
        if operation and operation[0] == "limit" and int(operation[1]) > 200:
            return {"allowed": False, "reason": "limit must be <= 200"}
        if operation and operation[0] == "aggregate" and operation[1][0] not in {"count", "sum", "avg", "min", "max"}:
            return {"allowed": False, "reason": "aggregation function is not allowed"}
    if not any(operation and operation[0] == "limit" for operation in operations):
        return {"allowed": False, "reason": "query program must include a limit"}
    if program.get("source", {}).get("type") not in {"metric", "table"}:
        return {"allowed": False, "reason": "source type must be metric or table"}
    return {"allowed": True, "reason": None}
