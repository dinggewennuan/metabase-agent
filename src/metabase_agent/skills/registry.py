from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from metabase_agent.config.settings import Settings

_REFERENCE_RE = re.compile(r"`([^`]+/[^`]+\.md)`")


@dataclass(slots=True)
class Skill:
    name: str
    description: str
    path: Path
    body: str


class SkillRegistry:
    def __init__(self, skills: list[Skill], *, max_chars: int = 6000) -> None:
        self.skills = skills
        self.max_chars = max_chars

    @classmethod
    def from_path(cls, root: str | Path, *, max_chars: int = 6000) -> SkillRegistry:
        root_path = Path(root)
        if not root_path.exists():
            return cls([], max_chars=max_chars)
        skills: list[Skill] = []
        for skill_md in sorted(root_path.glob("*/SKILL.md")):
            parsed = _parse_skill(skill_md)
            if parsed is not None:
                skills.append(parsed)
        return cls(skills, max_chars=max_chars)

    def match(self, question: str, *, limit: int = 2) -> list[Skill]:
        if not question.strip() or not self.skills:
            return []
        scored: list[tuple[int, Skill]] = []
        query_terms = _terms(question)
        for skill in self.skills:
            haystack = f"{skill.name} {skill.description}".lower()
            score = sum(1 for term in query_terms if term and term in haystack)
            if score:
                scored.append((score, skill))
        scored.sort(key=lambda item: (item[0], item[1].name), reverse=True)
        return [skill for _score, skill in scored[:limit]]

    def render_context(self, question: str) -> str:
        matched = self.match(question)
        if not matched:
            return ""
        sections = ["可用任务技能："]
        remaining = self.max_chars
        for skill in matched:
            content = _skill_content_with_references(skill)
            if len(content) > remaining:
                content = content[:remaining].rstrip() + "\n..."
            sections.append(f"\n## {skill.name}\n{content}")
            remaining -= len(content)
            if remaining <= 0:
                break
        return "\n".join(sections).strip()


def build_skill_registry(settings: Settings) -> SkillRegistry:
    if not settings.agent_skills_enabled:
        return SkillRegistry([])
    return SkillRegistry.from_path(settings.agent_skills_path, max_chars=settings.agent_skills_max_chars)


def _parse_skill(path: Path) -> Skill | None:
    text = path.read_text(encoding="utf-8")
    frontmatter, body = _split_frontmatter(text)
    name = str(frontmatter.get("name") or path.parent.name).strip()
    description = str(frontmatter.get("description") or "").strip()
    if not name or not description:
        return None
    return Skill(name=name, description=description, path=path, body=body.strip())


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    raw = parts[1]
    data: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data, parts[2]


def _skill_content_with_references(skill: Skill) -> str:
    parts = [skill.body]
    for ref in _REFERENCE_RE.findall(skill.body):
        ref_path = (skill.path.parent / ref).resolve()
        try:
            ref_path.relative_to(skill.path.parent.resolve())
        except ValueError:
            continue
        if ref_path.exists() and ref_path.is_file():
            parts.append(f"\n### Reference: {ref}\n{ref_path.read_text(encoding='utf-8').strip()}")
    return "\n".join(part for part in parts if part).strip()


def _terms(text: str) -> set[str]:
    lowered = text.lower()
    words = {word for word in re.split(r"[^a-z0-9_@.\-/]+", lowered) if len(word) >= 2}
    chinese_keywords = {
        keyword
        for keyword in (
            "数据库",
            "表",
            "字段",
            "指标",
            "口径",
            "查询",
            "聚合",
            "sql",
            "审批",
            "安全",
            "记忆",
            "长期",
            "向量",
            "mongodb",
            "pgvector",
            "langgraph",
            "metabase",
        )
        if keyword in lowered
    }
    return words | chinese_keywords
