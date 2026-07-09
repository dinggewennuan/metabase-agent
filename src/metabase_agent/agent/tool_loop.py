from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from metabase_agent.agent.tools import AgentTools, tool_schemas

_SYSTEM_PROMPT = (
    "你是 Metabase 只读数据分析 Agent。通过调用提供的工具来回答用户问题："
    "先用 list_databases / list_tables / list_fields 探查元数据，再用 run_aggregation 做聚合，"
    "或用 run_sql 执行只读 SQL（需用户授权）。拿到工具结果后，用简洁中文给出最终回答，并说明数据口径。"
    "不要编造数据；如果工具返回错误或 not_found，请如实说明并建议下一步。"
)


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LoopOutcome:
    status: str
    answer: str = ""
    trace: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    pending_sql: str | None = None
    pending_database_name: str | None = None
    pending_tool_call_id: str | None = None
    last_result: dict[str, Any] | None = None


class LLMTransport(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str | list[ToolCall]: ...


def _build_messages(question: str, history: list[dict[str, Any]], *, memory_context: str = "", skills_context: str = "") -> list[dict[str, Any]]:
    system_prompt = _SYSTEM_PROMPT
    if memory_context.strip():
        system_prompt = f"{system_prompt}\n\n{memory_context.strip()}"
    if skills_context.strip():
        system_prompt = f"{system_prompt}\n\n{skills_context.strip()}"
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    if question:
        messages.append({"role": "user", "content": question})
    return messages


def run_tool_loop(
    question: str,
    history: list[dict[str, Any]],
    transport: LLMTransport,
    tools: AgentTools,
    *,
    max_iterations: int = 6,
    approved_sql: str | None = None,
    approved_database_name: str | None = None,
    approved_tool_call_id: str | None = None,
    memory_context: str = "",
    skills_context: str = "",
) -> LoopOutcome:
    outcome = LoopOutcome(status="exhausted")
    for kind, payload in iter_tool_loop(
        question,
        history,
        transport,
        tools,
        max_iterations=max_iterations,
        approved_sql=approved_sql,
        approved_database_name=approved_database_name,
        approved_tool_call_id=approved_tool_call_id,
        memory_context=memory_context,
        skills_context=skills_context,
    ):
        if kind == "outcome":
            outcome = payload
    return outcome


def iter_tool_loop(
    question: str,
    history: list[dict[str, Any]],
    transport: LLMTransport,
    tools: AgentTools,
    *,
    max_iterations: int = 6,
    approved_sql: str | None = None,
    approved_database_name: str | None = None,
    approved_tool_call_id: str | None = None,
    memory_context: str = "",
    skills_context: str = "",
) -> Iterator[tuple[str, Any]]:
    schemas = tool_schemas()
    last_result: dict[str, Any] | None = None
    seen_calls: set[str] = set()
    if approved_sql is not None:
        messages = list(history)
        run_sql_arguments: dict[str, Any] = {"sql": approved_sql}
        if approved_database_name:
            run_sql_arguments["database_name"] = approved_database_name
        last_result = tools.dispatch("run_sql", run_sql_arguments)
        trace: list[dict[str, Any]] = [{"step": "tool.result", "tool": "run_sql", "status": last_result.get("status"), "approved": True}]
        messages.append(_tool_result_message(approved_tool_call_id or "run_sql", "run_sql", last_result))
    else:
        messages = _build_messages(question, history, memory_context=memory_context, skills_context=skills_context)
        trace = []

    for _ in range(max_iterations):
        reply = transport.complete(messages, schemas)
        if isinstance(reply, str):
            messages.append({"role": "assistant", "content": reply})
            yield "outcome", LoopOutcome(status="completed", answer=reply, trace=trace, messages=messages, last_result=last_result)
            return

        messages.append({"role": "assistant", "content": "", "tool_calls": [_serialize_call(call) for call in reply]})
        for index, call in enumerate(reply):
            call_event = {"step": "tool.call", "tool": call.name, "arguments": call.arguments}
            trace.append(call_event)
            yield "trace", call_event
            if call.name == "run_sql":
                sql = str(call.arguments.get("sql") or "")
                # Every tool_call in this batch except run_sql itself needs a
                # response message, or the resumed conversation is rejected by
                # the API for having unanswered tool calls.
                for deferred in reply[index + 1 :]:
                    messages.append(_tool_result_message(deferred.id, deferred.name, {"status": "deferred", "error": "deferred until pending SQL approval is resolved; call again if still needed"}))
                yield "outcome", LoopOutcome(
                    status="requires_approval",
                    answer="请先 review 这条 SQL，确认后授权执行，或拒绝本次执行。",
                    trace=trace,
                    messages=messages,
                    pending_sql=sql,
                    pending_database_name=str(call.arguments.get("database_name") or "") or None,
                    pending_tool_call_id=call.id,
                )
                return
            signature = _call_signature(call)
            if signature in seen_calls:
                result: dict[str, Any] = {"status": "error", "error": "repeated identical tool call ignored; change arguments or answer with what you have"}
                repeat_event = {"step": "tool.repeated", "tool": call.name}
                trace.append(repeat_event)
                yield "trace", repeat_event
            else:
                seen_calls.add(signature)
                result = tools.dispatch(call.name, call.arguments)
                last_result = result
            result_event = {"step": "tool.result", "tool": call.name, "status": result.get("status")}
            trace.append(result_event)
            yield "trace", result_event
            messages.append(_tool_result_message(call.id, call.name, result))

    yield "outcome", LoopOutcome(status="exhausted", answer="未能在限定步数内完成查询，请缩小问题范围或补充条件。", trace=trace, messages=messages, last_result=last_result)


def _call_signature(call: ToolCall) -> str:
    return f"{call.name}:{json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)}"


def _serialize_call(call: ToolCall) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": call.arguments}


def _tool_result_message(tool_call_id: str, name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": json.dumps(result, ensure_ascii=False)}
