from pathlib import Path

from metabase_agent.skills.registry import SkillRegistry


def test_skill_registry_parses_and_matches_skill(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "sql-safety"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: sql-safety
description: Use this skill for SQL approval and read-only query safety.
---

# sql-safety

Use approval rules.
""",
        encoding="utf-8",
    )

    registry = SkillRegistry.from_path(tmp_path / "skills")

    rendered = registry.render_context("这个 SQL 可以审批执行吗？")
    assert "sql-safety" in rendered
    assert "Use approval rules" in rendered


def test_missing_skills_path_is_empty(tmp_path: Path) -> None:
    registry = SkillRegistry.from_path(tmp_path / "missing")

    assert registry.render_context("SQL 审批") == ""
