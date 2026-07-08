from __future__ import annotations

from collections.abc import Iterable

from metabase_agent.memory.models import MemoryContext, MemoryRecord


def render_memory_context(profile: Iterable[MemoryRecord], active_rules: Iterable[MemoryRecord], related: Iterable[MemoryRecord]) -> str:
    profile_items = list(profile)
    rule_items = list(active_rules)
    related_items = list(related)
    if not profile_items and not rule_items and not related_items:
        return ""

    sections: list[str] = ["长期记忆上下文："]
    if profile_items:
        sections.append("\n用户/业务画像：")
        sections.extend(f"- {item.content}" for item in profile_items[:8])
    if rule_items:
        sections.append("\n有效规则：")
        sections.extend(f"- {item.content}" for item in rule_items[:6])
    if related_items:
        sections.append("\n相关历史与知识：")
        sections.extend(f"- {item.content}" for item in related_items[:8])
    return "\n".join(sections).strip()


def build_context(profile: list[MemoryRecord], active_rules: list[MemoryRecord], related: list[MemoryRecord]) -> MemoryContext:
    return MemoryContext(profile=profile, active_rules=active_rules, related=related, rendered=render_memory_context(profile, active_rules, related))
