"""LLM-based candidate-memory extraction.

Pipeline (see docs/agent-memory-langgraph-mongodb.md §7/§12): the LLM only
PROPOSES candidates; code rules validate, filter and demote them before
anything is written. Episodic events are recorded deterministically by
MemoryManager, so the LLM is only asked for semantic facts and procedural
rules.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from metabase_agent.config.settings import Settings
from metabase_agent.memory.models import CandidateMemory, MemoryStatus, MemoryType
from metabase_agent.semantics.llm_client import complete

_LOGGER = logging.getLogger("metabase_agent")

_MAX_CANDIDATES = 6
_MIN_CONFIDENCE = 0.6
_MAX_CONTENT_CHARS = 300
_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]{1,120}$")
# Never persist anything that smells like a credential.
_SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|passwd|password|secret|bearer\s+\S|sk-[A-Za-z0-9]{8,})")

_EXTRACTOR_SYSTEM_PROMPT = (
    "你是数据分析 Agent 的记忆提取器。从这一轮问答中提取值得跨会话长期保存的记忆。\n"
    "只返回一个 JSON 对象，格式为 {\"memories\": [ ... ]}，其中数组每个元素是一条记忆；"
    "没有可保存内容时返回 {\"memories\": []}。\n"
    "每条记忆的字段：memory_type、content、key（可选）、value（可选）、confidence（0~1，必填，越确定越接近 1）。\n"
    "memory_type 只能是：\n"
    "- semantic：用户/业务的稳定事实与偏好（回答语言、默认数据库/表、业务口径、表含义）。key 用 profile.* 或 note.*。\n"
    "- procedural：以后遇到类似任务应该怎么做的规则（例如\"列表时忽略某类表\"、\"分析前先确认时间字段\"）。key 用 rule.*。\n"
    "提取规则：\n"
    "1. 只提取长期、可复用的信息；寒暄、一次性指令、本轮临时条件一律不提取。\n"
    "2. 用户交代的业务口径/表含义/默认关注范围都属于 semantic，应逐条提取。"
    "例如\"fs_results 是 faceswap 记录结果表\"就是一条 semantic 记忆。\n"
    "3. 严格保留否定语义：\"users_20xx 这种表之后基本不怎么关注了\"应提取为 procedural 规则"
    "\"列出或分析表时默认忽略 users_20xxxxxx、pseudonymous_users_* 这类日期分表，除非用户明确要求\"，"
    "而不是\"用户关注 users_20xx\"。\n"
    "4. content 用简洁中文陈述句，一条一个事实/规则，confidence 必须给出。\n"
    "5. 没有值得保存的内容就返回 {\"memories\": []}。不要编造。"
)


def extract_candidates_with_llm(
    question: str,
    answer: str,
    query_plan: dict[str, Any] | None,
    settings: Settings,
) -> list[CandidateMemory]:
    """Ask the LLM for candidate memories; invalid/unsafe items are dropped."""
    plan_hint = ""
    if isinstance(query_plan, dict):
        hints = {key: query_plan.get(key) for key in ("intent", "database_name", "schema_name", "table_name") if query_plan.get(key)}
        if hints:
            plan_hint = f"\n本轮查询计划：{json.dumps(hints, ensure_ascii=False)}"
    user_content = f"用户问题：{question}\nAgent 回答（截断）：{answer[:500]}{plan_hint}"
    content = complete(_EXTRACTOR_SYSTEM_PROMPT, user_content, settings, json_mode=True)
    return _parse_candidates(content)


def _parse_candidates(content: str | None) -> list[CandidateMemory]:
    items = _parse_json_items(content)
    candidates: list[CandidateMemory] = []
    for item in items[:_MAX_CANDIDATES]:
        candidate = _validate_candidate(item)
        if candidate is not None:
            candidates.append(candidate)
    # Make "extractor produced nothing" observable, and dump the raw model
    # output when it yields nothing so a prompt/format mismatch is debuggable.
    _LOGGER.info("memory.extractor: LLM proposed %s item(s), %s passed validation", len(items), len(candidates))
    if not items and content:
        _LOGGER.info("memory.extractor: raw output (no items parsed): %s", content.strip()[:400])
    return candidates


def _parse_json_items(content: str | None) -> list[Any]:
    if not content:
        return []
    cleaned = content.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        _LOGGER.warning("memory.extractor: non-JSON LLM output dropped")
        return []
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        # json_object mode forces an object wrapper; accept the known keys, and
        # fall back to the first list-valued field for any other wrapper name.
        for key in ("memories", "candidates", "items", "data", "result", "results"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
        for value in parsed.values():
            if isinstance(value, list):
                return value
    return []


def _validate_candidate(item: Any) -> CandidateMemory | None:
    """Code-rule gate: the LLM proposes, these rules decide."""
    if not isinstance(item, dict):
        return None
    try:
        memory_type = MemoryType(str(item.get("memory_type") or ""))
    except ValueError:
        return None
    if memory_type == MemoryType.EPISODIC:
        # Episodic events are written deterministically per interaction.
        return None
    content = str(item.get("content") or "").strip()
    if len(content) < 6:
        return None
    if _SECRET_PATTERN.search(content):
        _LOGGER.warning("memory.extractor: candidate dropped (credential-like content)")
        return None
    content = content[:_MAX_CONTENT_CHARS]
    try:
        confidence = float(item.get("confidence", 0.0))
    except (TypeError, ValueError):
        return None
    confidence = min(max(confidence, 0.0), 1.0)
    if confidence < _MIN_CONFIDENCE:
        return None
    key = str(item.get("key") or "").strip() or None
    if key is not None and not _KEY_PATTERN.fullmatch(key):
        key = None
    # Procedural memory changes future agent behaviour — it NEVER goes live
    # without review, no matter what the model claims.
    status = MemoryStatus.PENDING_REVIEW if memory_type == MemoryType.PROCEDURAL else MemoryStatus.ACTIVE
    return CandidateMemory(
        memory_type=memory_type,
        content=content,
        key=key,
        value=item.get("value"),
        confidence=confidence,
        status=status,
        source="llm_extractor",
    )
