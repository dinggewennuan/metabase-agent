from __future__ import annotations

from metabase_agent.memory.models import MemoryType


def user_namespace(tenant_id: str, user_id: str, memory_type: MemoryType) -> tuple[str, ...]:
    return ("tenant", tenant_id, "user", user_id, memory_type.value)


def org_namespace(tenant_id: str, memory_type: MemoryType) -> tuple[str, ...]:
    return ("tenant", tenant_id, "org", memory_type.value)


def app_namespace(app_id: str, memory_type: MemoryType) -> tuple[str, ...]:
    return ("app", app_id, memory_type.value)
