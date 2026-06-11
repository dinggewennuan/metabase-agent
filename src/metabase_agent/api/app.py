from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from metabase_agent.agent.graph import build_graph
from metabase_agent.config.settings import Settings, get_settings
from metabase_agent.query.bigquery_report_sql import extract_native_sql

MAX_MEMORY_MESSAGES = 20
CONVERSATION_MEMORY: dict[str, list[dict[str, str]]] = {}
PENDING_SQL_APPROVALS: dict[str, dict[str, Any]] = {}
PENDING_TABLE_CONTEXT: dict[str, dict[str, Any]] = {}
MEMORY_LOADED = False
STATE_LOADED = False
_STATE_LOCK = threading.RLock()  # guards the in-memory dicts and on-disk writes
_GRAPH_CACHE: dict[tuple[str, ...], Any] = {}  # compiled graph reused across requests


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    dry_run: bool | None = None
    session_id: str = Field(default="default", min_length=1)
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
    _load_memory()
    with _STATE_LOCK:
        history = CONVERSATION_MEMORY.setdefault(session_id, [])
        history.append({"role": role, "content": content})
        del history[:-MAX_MEMORY_MESSAGES]
        snapshot = list(history)
    _save_memory()
    return snapshot


def _history(session_id: str) -> list[dict[str, str]]:
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
    if not any(marker in question or marker in lowered for marker in ("count", "多少", "条数", "行数", "最近", "每天", "求和", "平均", "最大", "最小")):
        return question
    table_name = str(context.get("table_name") or "")
    if not table_name:
        return question
    schema_name = str(context.get("schema_name") or "")
    prefix = f"{schema_name} 下{table_name}" if schema_name else table_name
    return f"{prefix} {question}"


def _run_ask(payload: AskRequest) -> AskResponse:
    settings = get_settings()
    graph = _get_graph(settings)
    dry_run = settings.agent_dry_run if payload.dry_run is None else payload.dry_run
    _load_memory()
    _load_state()
    history = CONVERSATION_MEMORY.setdefault(payload.session_id, [])
    pending_sql = PENDING_SQL_APPROVALS.get(payload.session_id)
    question = payload.question
    sql_approved = False
    if pending_sql and _is_sql_approval(question, payload.decision):
        question = str(pending_sql["question"])
        sql_approved = True
    elif pending_sql and _is_sql_rejection(question, payload.decision):
        PENDING_SQL_APPROVALS.pop(payload.session_id, None)
        _save_state()
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
        )
    else:
        question = _fill_table_follow_up(question, PENDING_TABLE_CONTEXT.get(payload.session_id))
    memory_answer = _answer_from_memory(payload.question, history)
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
        )
    try:
        result = graph.invoke({"question": question, "dry_run": dry_run, "sql_approved": sql_approved})
    except Exception as exc:
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
    answer = str(result.get("answer", ""))
    query_result = result.get("query_result")
    if isinstance(query_result, dict) and query_result.get("status") == "requires_approval":
        PENDING_SQL_APPROVALS[payload.session_id] = {"question": question, "sql": query_result.get("sql")}
    elif sql_approved:
        PENDING_SQL_APPROVALS.pop(payload.session_id, None)
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
            PENDING_TABLE_CONTEXT[payload.session_id] = {"schema_name": selected_schema, "table_name": selected_table}
        elif not suggestions and query_result.get("status") == "completed":
            PENDING_TABLE_CONTEXT.pop(payload.session_id, None)
    _save_state()
    _remember(payload.session_id, "user", payload.question)
    memory = _remember(payload.session_id, "assistant", answer)
    return AskResponse(
        answer=answer,
        query_plan=result.get("query_plan"),
        program=result.get("program"),
        query_result=query_result,
        trace=result.get("trace"),
        session_id=payload.session_id,
        memory=memory,
    )


def _get_graph(settings: Settings) -> Any:
    """Build the LangGraph once per distinct backend config and reuse it.

    Previously every request rebuilt the graph and a fresh MetabaseClient; the
    cache keeps a single compiled graph (and its connection-pooled client) alive.
    """
    key = (
        settings.metabase_base_url,
        settings.metabase_api_key,
        settings.openai_api_key,
        settings.openai_base_url,
        settings.openai_model,
        settings.openai_wire_api,
    )
    with _STATE_LOCK:
        graph = _GRAPH_CACHE.get(key)
        if graph is None:
            graph = build_graph(settings)
            _GRAPH_CACHE[key] = graph
    return graph


def _check_token(x_agent_token: str | None) -> None:
    """Enforce X-Agent-Token when AGENT_API_TOKEN is configured; no-op otherwise."""
    token = get_settings().agent_api_token
    if token and x_agent_token != token:
        raise HTTPException(status_code=401, detail="invalid or missing X-Agent-Token")


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
        return SessionResponse(session_id=session_id, memory=_history(session_id))

    @app.post("/api/ask", response_model=AskResponse)
    def ask(payload: AskRequest, x_agent_token: str | None = Header(default=None)) -> AskResponse:
        _check_token(x_agent_token)
        return _run_ask(payload)

    @app.post("/api/ask/stream")
    def ask_stream(payload: AskRequest, x_agent_token: str | None = Header(default=None)) -> StreamingResponse:
        _check_token(x_agent_token)

        def events() -> Any:
            yield _stream_event("status", {"message": "正在解析问题..."})
            if PENDING_SQL_APPROVALS.get(payload.session_id) and _is_sql_approval(payload.question, payload.decision):
                yield _stream_event("status", {"message": "已授权，正在执行 SQL..."})
            elif PENDING_SQL_APPROVALS.get(payload.session_id) and _is_sql_rejection(payload.question, payload.decision):
                yield _stream_event("status", {"message": "正在拒绝本次执行..."})
            else:
                yield _stream_event("status", {"message": "正在规划查询..."})
            data = _run_ask(payload).model_dump()
            yield _stream_event("final", data)

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


_STATIC_DIR = Path(__file__).parent / "static"
HTML = (_STATIC_DIR / "index.html").read_text(encoding="utf-8")


app = create_app()
