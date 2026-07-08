from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from metabase_agent.agent.graph import build_graph
from metabase_agent.agent.tool_loop import LoopOutcome, iter_tool_loop
from metabase_agent.agent.tools import AgentTools
from metabase_agent.api.store import SqliteStore
from metabase_agent.config.settings import Settings, get_settings
from metabase_agent.memory import (
    MemoryManager,
    MemoryStatus,
    MemoryType,
    build_memory_manager,
)
from metabase_agent.query.bigquery_report_sql import extract_native_sql
from metabase_agent.semantics.llm_client import build_tool_transport
from metabase_agent.skills import SkillRegistry, build_skill_registry

MAX_MEMORY_MESSAGES = 20
_LOGGER = logging.getLogger("metabase_agent")
CONVERSATION_MEMORY: dict[str, list[dict[str, str]]] = {}
PENDING_SQL_APPROVALS: dict[str, dict[str, Any]] = {}
PENDING_TABLE_CONTEXT: dict[str, dict[str, Any]] = {}
MEMORY_LOADED = False
STATE_LOADED = False
_STATE_LOCK = threading.RLock()  # guards the in-memory dicts and on-disk writes
_GRAPH_CACHE: dict[str, Any] = {}  # compiled graph reused across requests
_SQLITE_STORE: dict[str, SqliteStore] = {}
_MEMORY_MANAGER_CACHE: dict[tuple[Any, ...], MemoryManager] = {}
_SKILL_REGISTRY_CACHE: dict[tuple[Any, ...], SkillRegistry] = {}
_CHECKPOINTER_CACHE: dict[tuple[Any, ...], Any] = {}


def _active_store() -> SqliteStore | None:
    settings = get_settings()
    if settings.agent_store != "sqlite":
        return None
    store = _SQLITE_STORE.get(settings.agent_state_path)
    if store is None:
        store = SqliteStore(settings.agent_state_path)
        _SQLITE_STORE[settings.agent_state_path] = store
    store.purge_expired(settings.agent_session_ttl_seconds)
    return store


def _get_approval(session_id: str) -> dict[str, Any] | None:
    store = _active_store()
    if store is not None:
        return store.get_approval(session_id)
    _load_state()
    return PENDING_SQL_APPROVALS.get(session_id)


def _set_approval(session_id: str, data: dict[str, Any]) -> None:
    store = _active_store()
    if store is not None:
        store.set_approval(session_id, data)
        return
    PENDING_SQL_APPROVALS[session_id] = data
    _save_state()


def _pop_approval(session_id: str) -> None:
    store = _active_store()
    if store is not None:
        store.pop_approval(session_id)
        return
    PENDING_SQL_APPROVALS.pop(session_id, None)
    _save_state()


def _get_table_context(session_id: str) -> dict[str, Any] | None:
    store = _active_store()
    if store is not None:
        return store.get_table_context(session_id)
    _load_state()
    return PENDING_TABLE_CONTEXT.get(session_id)


def _set_table_context(session_id: str, data: dict[str, Any]) -> None:
    store = _active_store()
    if store is not None:
        store.set_table_context(session_id, data)
        return
    PENDING_TABLE_CONTEXT[session_id] = data
    _save_state()


def _pop_table_context(session_id: str) -> None:
    store = _active_store()
    if store is not None:
        store.pop_table_context(session_id)
        return
    PENDING_TABLE_CONTEXT.pop(session_id, None)
    _save_state()


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    dry_run: bool | None = None
    session_id: str = Field(default="default", min_length=1)
    tenant_id: str | None = None
    user_id: str | None = None
    decision: str | None = None  # "approve" / "reject" — explicit signal for a pending SQL review


class ConfigResponse(BaseModel):
    default_dry_run: bool


class AskResponse(BaseModel):
    answer: str
    query_plan: dict[str, Any] | None
    program: dict[str, Any] | None
    query_result: dict[str, Any] | None
    trace: list[dict[str, Any]] | None
    session_id: str | None = None
    memory: list[dict[str, str]] | None = None


class SessionResponse(BaseModel):
    session_id: str
    memory: list[dict[str, str]]
    pending_approval: dict[str, Any] | None = None


class MemoryWriteRequest(BaseModel):
    tenant_id: str | None = None
    user_id: str | None = None
    memory_type: MemoryType = MemoryType.SEMANTIC
    key: str | None = None
    content: str = Field(min_length=1)
    value: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: MemoryStatus = MemoryStatus.ACTIVE


class MemoryStatusRequest(BaseModel):
    tenant_id: str | None = None
    user_id: str | None = None
    status: MemoryStatus


class MemoryItemResponse(BaseModel):
    memory: dict[str, Any]


class MemoryListResponse(BaseModel):
    memories: list[dict[str, Any]]


def _memory_path() -> Path:
    return Path(get_settings().agent_memory_path)


def _state_path() -> Path:
    return Path(get_settings().agent_state_path)


def _atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def _load_state() -> None:
    """Reload pending SQL approvals and table context from disk so they survive a restart."""
    global STATE_LOADED
    with _STATE_LOCK:
        if STATE_LOADED:
            return
        STATE_LOADED = True
        path = _state_path()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        PENDING_SQL_APPROVALS.clear()
        PENDING_TABLE_CONTEXT.clear()
        approvals = payload.get("approvals")
        table_context = payload.get("table_context")
        if isinstance(approvals, dict):
            PENDING_SQL_APPROVALS.update({k: v for k, v in approvals.items() if isinstance(k, str) and isinstance(v, dict)})
        if isinstance(table_context, dict):
            PENDING_TABLE_CONTEXT.update({k: v for k, v in table_context.items() if isinstance(k, str) and isinstance(v, dict)})


def _save_state() -> None:
    with _STATE_LOCK:
        payload = json.dumps({"approvals": PENDING_SQL_APPROVALS, "table_context": PENDING_TABLE_CONTEXT}, ensure_ascii=False, indent=2)
    _atomic_write(_state_path(), payload)


def _load_memory() -> None:
    global MEMORY_LOADED
    with _STATE_LOCK:
        if MEMORY_LOADED:
            return
        MEMORY_LOADED = True
        path = _memory_path()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        CONVERSATION_MEMORY.clear()
        for session_id, messages in payload.items():
            if isinstance(session_id, str) and isinstance(messages, list):
                CONVERSATION_MEMORY[session_id] = [message for message in messages if _is_memory_message(message)][-MAX_MEMORY_MESSAGES:]


def _save_memory() -> None:
    # Atomic write: a concurrent reader (or another process) always sees the
    # complete previous or complete new file, never a half-written one.
    with _STATE_LOCK:
        payload = json.dumps(CONVERSATION_MEMORY, ensure_ascii=False, indent=2)
    _atomic_write(_memory_path(), payload)


def _is_memory_message(value: object) -> bool:
    return isinstance(value, dict) and isinstance(value.get("role"), str) and isinstance(value.get("content"), str)


def _remember(session_id: str, role: str, content: str) -> list[dict[str, str]]:
    store = _active_store()
    if store is not None:
        return store.append_message(session_id, role, content, MAX_MEMORY_MESSAGES)
    _load_memory()
    with _STATE_LOCK:
        history = CONVERSATION_MEMORY.setdefault(session_id, [])
        history.append({"role": role, "content": content})
        del history[:-MAX_MEMORY_MESSAGES]
        snapshot = list(history)
    _save_memory()
    return snapshot


def _history(session_id: str) -> list[dict[str, str]]:
    store = _active_store()
    if store is not None:
        return store.history(session_id)
    _load_memory()
    return list(CONVERSATION_MEMORY.get(session_id, []))


def _last_user_question(history: list[dict[str, str]]) -> str | None:
    for message in reversed(history):
        if message.get("role") == "user":
            return message.get("content")
    return None


def _answer_from_memory(question: str, history: list[dict[str, str]]) -> str | None:
    if not any(word in question for word in ("上次", "刚才", "之前", "上一轮")):
        return None
    if not any(word in question for word in ("问", "内容", "问题")):
        return None
    last_question = _last_user_question(history)
    if not last_question:
        return "我还没有记住上一轮问题。"
    return f"你上次问的是：{last_question}"


_APPROVAL_PHRASES = ("确认执行", "同意执行", "可以执行", "执行吧", "批准执行", "approve", "run it", "execute")
_REJECTION_PHRASES = ("拒绝", "取消", "不要执行", "不执行", "reject", "cancel")


def _is_sql_approval(question: str, decision: str | None = None) -> bool:
    if decision == "approve":
        return True
    if decision is not None:
        return False
    # A message that itself carries SQL is a new request, never an approval of a
    # previously pending SQL — otherwise pasting a 2nd query re-runs the 1st.
    if extract_native_sql(question):
        return False
    lowered = question.lower().strip()
    return any(phrase in question or phrase in lowered for phrase in _APPROVAL_PHRASES)


def _is_sql_rejection(question: str, decision: str | None = None) -> bool:
    if decision == "reject":
        return True
    if decision is not None:
        return False
    if extract_native_sql(question):
        return False
    lowered = question.lower().strip()
    return any(phrase in question or phrase in lowered for phrase in _REJECTION_PHRASES)


def _fill_table_follow_up(question: str, context: dict[str, Any] | None) -> str:
    if not context:
        return question
    lowered = question.lower()
    if any(marker in question or marker in lowered for marker in (" 下", "表", "table", "collection", "select", "with")):
        return question
    if not any(marker in question or marker in lowered for marker in ("count", "多少", "条数", "行数", "最近", "每天", "求和", "平均", "最大", "最小", "增长", "下降", "变化", "对比", "环比", "哪部分")):
        return question
    table_name = str(context.get("table_name") or "")
    if not table_name:
        return question
    schema_name = str(context.get("schema_name") or "")
    prefix = f"{schema_name} 下{table_name}" if schema_name else table_name
    metric_phrase = _follow_up_metric_phrase(question, context)
    return f"{prefix} {metric_phrase}{question}"


def _follow_up_metric_phrase(question: str, context: dict[str, Any]) -> str:
    lowered = question.lower()
    if any(marker in question or marker in lowered for marker in ("count", "多少", "条数", "行数", "求和", "平均", "最大", "最小")):
        return ""
    aggregation = str(context.get("aggregation_function") or "count")
    relative_days = context.get("relative_days")
    time_grain = str(context.get("time_grain") or "")
    parts: list[str] = []
    if isinstance(relative_days, int) and relative_days > 0 and not any(marker in question for marker in ("最近", "近", "过去", "昨天", "上周")):
        parts.append(f"最近{relative_days}天")
    if time_grain == "day" and not any(marker in question or marker in lowered for marker in ("每天", "按天", "daily", "day")):
        parts.append("每天的")
    if aggregation == "count":
        parts.append("数据count")
    elif aggregation:
        parts.append(f"数据{aggregation}")
    return "".join(parts) + " "


@dataclass
class _PreparedAsk:
    question: str
    dry_run: bool
    sql_approved: bool
    tenant_id: str
    user_id: str
    memory_context: str = ""
    skills_context: str = ""


def _prepare_ask(payload: AskRequest) -> tuple[AskResponse | None, _PreparedAsk | None]:
    settings = get_settings()
    dry_run = settings.agent_dry_run if payload.dry_run is None else payload.dry_run
    history = _history(payload.session_id)
    pending_sql = _get_approval(payload.session_id)
    question = payload.question
    sql_approved = False
    if pending_sql and _is_sql_approval(question, payload.decision):
        question = str(pending_sql["question"])
        sql_approved = True
    elif pending_sql and _is_sql_rejection(question, payload.decision):
        _pop_approval(payload.session_id)
        answer = "已拒绝执行，SQL 未运行。"
        _remember(payload.session_id, "user", "拒绝执行")
        memory = _remember(payload.session_id, "assistant", answer)
        return AskResponse(
            answer=answer,
            query_plan={"intent": "sql_approval", "rejected": True},
            program=None,
            query_result={"status": "rejected", "sql": pending_sql.get("sql")},
            trace=[{"step": "sql.review", "status": "rejected"}],
            session_id=payload.session_id,
            memory=memory,
        ), None
    else:
        question = _fill_table_follow_up(question, _get_table_context(payload.session_id))
    memory_answer = None if _use_tools(settings, dry_run) else _answer_from_memory(payload.question, history)
    if memory_answer is not None:
        _remember(payload.session_id, "user", payload.question)
        memory = _remember(payload.session_id, "assistant", memory_answer)
        return AskResponse(
            answer=memory_answer,
            query_plan={"intent": "memory_lookup"},
            program=None,
            query_result={"status": "completed", "source": "memory"},
            trace=[{"step": "memory.lookup", "status": "completed"}],
            session_id=payload.session_id,
            memory=memory,
        ), None
    tenant_id, user_id, memory_context, skills_context = _load_contexts(settings, payload, question)
    return None, _PreparedAsk(
        question=question,
        dry_run=dry_run,
        sql_approved=sql_approved,
        tenant_id=tenant_id,
        user_id=user_id,
        memory_context=memory_context,
        skills_context=skills_context,
    )


def _failed_ask(payload: AskRequest, exc: Exception) -> AskResponse:
    _remember(payload.session_id, "user", payload.question)
    memory = _remember(payload.session_id, "assistant", f"查询失败：{exc}")
    return AskResponse(
        answer=f"查询失败：{exc}",
        query_plan=None,
        program=None,
        query_result={"status": "failed", "error": str(exc)},
        trace=[{"step": "api.ask", "status": "failed", "error": str(exc)}],
        session_id=payload.session_id,
        memory=memory,
    )


def _finalize_ask(payload: AskRequest, run: _PreparedAsk, result: dict[str, Any]) -> AskResponse:
    answer = str(result.get("answer", ""))
    query_result = result.get("query_result")
    if isinstance(query_result, dict) and query_result.get("status") == "requires_approval":
        _set_approval(payload.session_id, {"question": run.question, "sql": query_result.get("sql")})
    elif run.sql_approved:
        _pop_approval(payload.session_id)
    if isinstance(query_result, dict):
        query_plan = result.get("query_plan")
        suggestions = query_result.get("suggestions")
        candidate_tables = query_result.get("candidate_tables")
        selected_table = None
        selected_schema = query_result.get("schema_name")
        if isinstance(candidate_tables, list) and candidate_tables:
            selected_table = candidate_tables[0]
        elif isinstance(query_result.get("table_name"), str) and query_result.get("status") == "completed":
            selected_table = query_result["table_name"]
        elif isinstance(query_plan, dict) and isinstance(query_plan.get("table_name"), str) and query_result.get("status") == "completed":
            selected_table = query_plan["table_name"]
            selected_schema = query_plan.get("schema_name")
        if selected_table:
            data = {"schema_name": selected_schema, "table_name": selected_table}
            if isinstance(query_plan, dict):
                for key in ("aggregation_function", "relative_days", "time_grain", "date_field_name", "field_name"):
                    if query_plan.get(key) is not None:
                        data[key] = query_plan.get(key)
            _set_table_context(payload.session_id, data)
        elif not suggestions and query_result.get("status") == "completed":
            _pop_table_context(payload.session_id)
    _remember(payload.session_id, "user", payload.question)
    memory = _remember(payload.session_id, "assistant", answer)
    _record_long_term_memory(payload, run, result, answer)
    return AskResponse(
        answer=answer,
        query_plan=result.get("query_plan"),
        program=result.get("program"),
        query_result=query_result,
        trace=result.get("trace"),
        session_id=payload.session_id,
        memory=memory,
    )


def _use_tools(settings: Settings, dry_run: bool) -> bool:
    return settings.agent_mode == "tools" and bool(settings.openai_api_key) and not dry_run


def _request_identity(payload: AskRequest, settings: Settings) -> tuple[str, str]:
    tenant_id = payload.tenant_id or settings.agent_tenant_id or "default"
    user_id = payload.user_id or settings.agent_user_id or payload.session_id
    return tenant_id, user_id


def _memory_admin_identity(settings: Settings, tenant_id: str | None, user_id: str | None) -> tuple[str, str]:
    resolved_tenant = tenant_id or settings.agent_tenant_id or "default"
    resolved_user = user_id or settings.agent_user_id
    if not resolved_user:
        raise HTTPException(status_code=400, detail="user_id is required when AGENT_USER_ID is not configured")
    return resolved_tenant, resolved_user


def _require_long_term_memory(settings: Settings) -> None:
    if not settings.agent_long_term_memory_enabled:
        raise HTTPException(status_code=400, detail="AGENT_LONG_TERM_MEMORY_ENABLED is false")


def _memory_manager(settings: Settings) -> MemoryManager:
    key = (
        settings.agent_long_term_memory_enabled,
        settings.agent_mongodb_uri,
        settings.agent_mongodb_database,
        settings.agent_memory_collection,
        settings.agent_pgvector_dsn,
        settings.agent_pgvector_table,
        settings.agent_embedding_provider,
        settings.agent_embedding_model,
        settings.agent_embedding_dimensions,
        settings.openai_api_key,
        settings.openai_base_url,
        settings.siliconflow_api_key,
        settings.siliconflow_base_url,
    )
    manager = _MEMORY_MANAGER_CACHE.get(key)
    if manager is None:
        manager = build_memory_manager(settings)
        _MEMORY_MANAGER_CACHE[key] = manager
    return manager


def _skill_registry(settings: Settings) -> SkillRegistry:
    key = (settings.agent_skills_enabled, settings.agent_skills_path, settings.agent_skills_max_chars)
    registry = _SKILL_REGISTRY_CACHE.get(key)
    if registry is None:
        registry = build_skill_registry(settings)
        _SKILL_REGISTRY_CACHE[key] = registry
    return registry


def _checkpointing_enabled(settings: Settings) -> bool:
    return settings.agent_checkpoint_backend.lower() == "mongodb"


def _graph_config(settings: Settings, session_id: str) -> dict[str, Any] | None:
    if not _checkpointing_enabled(settings):
        return None
    return {"configurable": {"thread_id": session_id}}


def _checkpointer(settings: Settings) -> Any | None:
    if not _checkpointing_enabled(settings):
        return None
    uri = settings.agent_checkpoint_mongodb_uri or settings.agent_mongodb_uri
    if not uri:
        raise RuntimeError("AGENT_CHECKPOINT_MONGODB_URI or AGENT_MONGODB_URI is required for MongoDB checkpointing")
    key = (
        settings.agent_checkpoint_backend,
        uri,
        settings.agent_checkpoint_mongodb_database,
        settings.agent_checkpoint_ttl_seconds,
    )
    saver = _CHECKPOINTER_CACHE.get(key)
    if saver is None:
        from langgraph.checkpoint.mongodb import MongoDBSaver
        from pymongo import MongoClient

        ttl = settings.agent_checkpoint_ttl_seconds if settings.agent_checkpoint_ttl_seconds > 0 else None
        saver = MongoDBSaver(MongoClient(uri), db_name=settings.agent_checkpoint_mongodb_database, ttl=ttl)
        _CHECKPOINTER_CACHE[key] = saver
    return saver


def _load_contexts(settings: Settings, payload: AskRequest, question: str) -> tuple[str, str, str, str]:
    tenant_id, user_id = _request_identity(payload, settings)
    memory_context = ""
    skills_context = ""
    try:
        memory_context = _memory_manager(settings).load_context(tenant_id=tenant_id, user_id=user_id, query=question).rendered
    except Exception as exc:
        _LOGGER.warning("memory.context.load failed: %s", exc)
    try:
        skills_context = _skill_registry(settings).render_context(question)
    except Exception as exc:
        _LOGGER.warning("skills.context.load failed: %s", exc)
    return tenant_id, user_id, memory_context, skills_context


def _record_long_term_memory(payload: AskRequest, run: _PreparedAsk, result: dict[str, Any], answer: str) -> None:
    settings = get_settings()
    if not settings.agent_long_term_memory_enabled:
        return
    query_result = result.get("query_result")
    query_plan = result.get("query_plan")
    try:
        _memory_manager(settings).record_interaction(
            tenant_id=run.tenant_id,
            user_id=run.user_id,
            question=payload.question,
            answer=answer,
            query_result=query_result if isinstance(query_result, dict) else None,
            query_plan=query_plan if isinstance(query_plan, dict) else None,
        )
    except Exception as exc:
        _LOGGER.warning("memory.record_interaction failed: %s", exc)


def _run_ask(payload: AskRequest) -> AskResponse:
    early, run = _prepare_ask(payload)
    if early is not None or run is None:
        return cast(AskResponse, early)
    settings = get_settings()
    if _use_tools(settings, run.dry_run):
        return _run_ask_tools(payload, run, settings)
    graph = _get_graph(settings)
    graph_input = {
        "question": run.question,
        "dry_run": run.dry_run,
        "sql_approved": run.sql_approved,
        "tenant_id": run.tenant_id,
        "user_id": run.user_id,
        "memory_context": run.memory_context,
        "skills_context": run.skills_context,
    }
    graph_config = _graph_config(settings, payload.session_id)
    try:
        result = graph.invoke(graph_input, config=graph_config) if graph_config is not None else graph.invoke(graph_input)
    except Exception as exc:
        return _failed_ask(payload, exc)
    return _finalize_ask(payload, run, result)


_TOOL_STEP_LABELS = {"tool.call": "调用工具", "tool.result": "工具返回", "tool.repeated": "跳过重复调用"}


def _tool_iterator(payload: AskRequest, run: _PreparedAsk, tools: AgentTools, transport: Any) -> Any:
    pending = _get_approval(payload.session_id)
    if run.sql_approved and isinstance(pending, dict) and pending.get("mode") == "tools":
        return iter_tool_loop(
            "",
            cast(list[dict[str, Any]], pending.get("messages") or []),
            transport,
            tools,
            approved_sql=str(pending.get("sql") or ""),
            approved_tool_call_id=str(pending.get("tool_call_id") or ""),
            memory_context=run.memory_context,
            skills_context=run.skills_context,
        )
    return iter_tool_loop(run.question, _history(payload.session_id), transport, tools, memory_context=run.memory_context, skills_context=run.skills_context)


def _finalize_tool_outcome(payload: AskRequest, run: _PreparedAsk, outcome: LoopOutcome) -> AskResponse:
    if outcome.status == "requires_approval":
        _set_approval(payload.session_id, {"question": run.question, "sql": outcome.pending_sql, "messages": outcome.messages, "tool_call_id": outcome.pending_tool_call_id, "mode": "tools"})
        query_result: dict[str, Any] = {"status": "requires_approval", "sql": outcome.pending_sql}
    else:
        _pop_approval(payload.session_id)
        query_result = outcome.last_result or {"status": outcome.status}
    _remember(payload.session_id, "user", payload.question)
    memory = _remember(payload.session_id, "assistant", outcome.answer)
    _record_long_term_memory(payload, run, {"query_plan": {"intent": "tool_loop", "status": outcome.status}, "query_result": query_result}, outcome.answer)
    return AskResponse(
        answer=outcome.answer,
        query_plan={"intent": "tool_loop", "status": outcome.status},
        program=None,
        query_result=query_result,
        trace=outcome.trace,
        session_id=payload.session_id,
        memory=memory,
    )


def _run_ask_tools(payload: AskRequest, run: _PreparedAsk, settings: Settings) -> AskResponse:
    tools = AgentTools(settings, dry_run=run.dry_run)
    transport = build_tool_transport(settings)
    try:
        outcome = LoopOutcome(status="exhausted")
        for kind, item in _tool_iterator(payload, run, tools, transport):
            if kind == "outcome":
                outcome = item
    except Exception as exc:
        return _failed_ask(payload, exc)
    return _finalize_tool_outcome(payload, run, outcome)


def _stream_ask_tools(payload: AskRequest, run: _PreparedAsk, settings: Settings) -> Any:
    tools = AgentTools(settings, dry_run=run.dry_run)
    transport = build_tool_transport(settings)
    outcome = LoopOutcome(status="exhausted")
    try:
        for kind, item in _tool_iterator(payload, run, tools, transport):
            if kind == "trace":
                yield _stream_event("status", {"message": _TOOL_STEP_LABELS.get(item.get("step"), "工具进行中"), "tool": item.get("tool"), "node": "tool_loop"})
            elif kind == "outcome":
                outcome = item
        data = _finalize_tool_outcome(payload, run, outcome).model_dump()
    except Exception as exc:
        data = _failed_ask(payload, exc).model_dump()
    yield _stream_event("final", data)


def _get_graph(settings: Settings) -> Any:
    """Build the LangGraph once per distinct backend config and reuse it.

    Previously every request rebuilt the graph and a fresh MetabaseClient; the
    cache keeps a single compiled graph (and its connection-pooled client) alive.
    """
    key = hashlib.sha256(
        "\x1f".join(
            (
                settings.metabase_base_url,
                settings.metabase_api_key,
                settings.openai_api_key,
                settings.openai_base_url,
                settings.openai_model,
                settings.openai_wire_api,
                settings.agent_checkpoint_backend,
                settings.agent_mongodb_uri,
                settings.agent_checkpoint_mongodb_uri,
                settings.agent_checkpoint_mongodb_database,
                str(settings.agent_checkpoint_ttl_seconds),
            )
        ).encode("utf-8")
    ).hexdigest()
    with _STATE_LOCK:
        graph = _GRAPH_CACHE.get(key)
        if graph is None:
            checkpointer = _checkpointer(settings)
            graph = build_graph(settings, checkpointer=checkpointer) if checkpointer is not None else build_graph(settings)
            _GRAPH_CACHE[key] = graph
    return graph


def _check_token(x_agent_token: str | None) -> None:
    settings = get_settings()
    token = settings.agent_api_token
    if settings.agent_require_token and not token:
        raise HTTPException(status_code=401, detail="server requires AGENT_API_TOKEN but none is configured")
    if token and x_agent_token != token:
        raise HTTPException(status_code=401, detail="invalid or missing X-Agent-Token")


_NODE_LABELS = {
    "parse": "已解析意图",
    "sql_explanation": "已生成 SQL 解读",
    "native_sql": "已处理 SQL 请求",
    "bigquery_sql": "已生成报表 SQL",
    "database_metadata": "已查询元数据",
    "search": "已搜索相关 Metric",
    "inspect": "已读取 Metric 详情",
    "plan": "已生成 Query Plan",
    "build_program": "已构建查询 Program",
    "policy": "已通过策略校验",
    "execute": "已执行查询",
    "answer": "已生成回答",
}


def _stream_event(name: str, payload: dict[str, Any]) -> str:
    return f"event: {name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app() -> FastAPI:
    app = FastAPI(title="Metabase Agent Python")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return HTML

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/api/config", response_model=ConfigResponse)
    def config() -> ConfigResponse:
        settings = get_settings()
        return ConfigResponse(default_dry_run=settings.agent_dry_run)

    @app.get("/api/sessions/{session_id}", response_model=SessionResponse)
    def session(session_id: str, x_agent_token: str | None = Header(default=None)) -> SessionResponse:
        _check_token(x_agent_token)
        pending = _get_approval(session_id)
        pending_approval = {"status": "requires_approval", "sql": pending.get("sql")} if pending else None
        return SessionResponse(session_id=session_id, memory=_history(session_id), pending_approval=pending_approval)

    @app.get("/api/memories", response_model=MemoryListResponse)
    def list_memories(
        tenant_id: str | None = Query(default=None),
        user_id: str | None = Query(default=None),
        memory_type: MemoryType | None = Query(default=None),
        status: MemoryStatus | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
        x_agent_token: str | None = Header(default=None),
    ) -> MemoryListResponse:
        _check_token(x_agent_token)
        settings = get_settings()
        _require_long_term_memory(settings)
        resolved_tenant, resolved_user = _memory_admin_identity(settings, tenant_id, user_id)
        records = _memory_manager(settings).list_memories(
            tenant_id=resolved_tenant,
            user_id=resolved_user,
            memory_type=memory_type,
            status=status,
            limit=limit,
        )
        return MemoryListResponse(memories=[record.to_dict() for record in records])

    @app.post("/api/memories", response_model=MemoryItemResponse)
    def put_memory(payload: MemoryWriteRequest, x_agent_token: str | None = Header(default=None)) -> MemoryItemResponse:
        _check_token(x_agent_token)
        settings = get_settings()
        _require_long_term_memory(settings)
        tenant_id, user_id = _memory_admin_identity(settings, payload.tenant_id, payload.user_id)
        record = _memory_manager(settings).put_memory(
            tenant_id=tenant_id,
            user_id=user_id,
            memory_type=payload.memory_type,
            key=payload.key,
            content=payload.content,
            value=payload.value,
            metadata=payload.metadata,
            confidence=payload.confidence,
            status=payload.status,
            source="manual_api",
        )
        if record is None:
            raise HTTPException(status_code=400, detail="memory was rejected by validation rules")
        return MemoryItemResponse(memory=record.to_dict())

    @app.post("/api/memories/{memory_id}/status", response_model=MemoryItemResponse)
    def update_memory_status(memory_id: str, payload: MemoryStatusRequest, x_agent_token: str | None = Header(default=None)) -> MemoryItemResponse:
        _check_token(x_agent_token)
        settings = get_settings()
        _require_long_term_memory(settings)
        tenant_id, user_id = _memory_admin_identity(settings, payload.tenant_id, payload.user_id)
        record = _memory_manager(settings).update_status(
            tenant_id=tenant_id,
            user_id=user_id,
            record_id=memory_id,
            status=payload.status,
        )
        if record is None:
            raise HTTPException(status_code=404, detail="memory not found")
        return MemoryItemResponse(memory=record.to_dict())

    @app.post("/api/ask", response_model=AskResponse)
    def ask(payload: AskRequest, x_agent_token: str | None = Header(default=None)) -> AskResponse:
        _check_token(x_agent_token)
        return _run_ask(payload)

    @app.post("/api/ask/stream")
    def ask_stream(payload: AskRequest, x_agent_token: str | None = Header(default=None)) -> StreamingResponse:
        _check_token(x_agent_token)

        def events() -> Any:
            yield _stream_event("status", {"message": "正在解析问题..."})
            early, run = _prepare_ask(payload)
            if early is not None or run is None:
                yield _stream_event("final", cast(AskResponse, early).model_dump())
                return
            yield _stream_event("status", {"message": "已授权，正在执行 SQL..." if run.sql_approved else "正在规划查询..."})
            settings = get_settings()
            if _use_tools(settings, run.dry_run):
                yield _stream_event("status", {"message": "Agent 正在调用工具...", "node": "tool_loop"})
                yield from _stream_ask_tools(payload, run, settings)
                return
            graph = _get_graph(settings)
            result: dict[str, Any] = {}
            graph_input = {
                "question": run.question,
                "dry_run": run.dry_run,
                "sql_approved": run.sql_approved,
                "tenant_id": run.tenant_id,
                "user_id": run.user_id,
                "memory_context": run.memory_context,
                "skills_context": run.skills_context,
            }
            graph_config = _graph_config(settings, payload.session_id)
            try:
                stream = graph.stream(graph_input, config=graph_config, stream_mode="updates") if graph_config is not None else graph.stream(graph_input, stream_mode="updates")
                for chunk in stream:
                    for node, delta in chunk.items():
                        if isinstance(delta, dict):
                            result.update(delta)
                        yield _stream_event("status", {"message": _NODE_LABELS.get(node, node), "node": node})
                data = _finalize_ask(payload, run, result).model_dump()
            except Exception as exc:
                data = _failed_ask(payload, exc).model_dump()
            yield _stream_event("final", data)

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


_STATIC_DIR = Path(__file__).parent / "static"
HTML = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


app = create_app()
