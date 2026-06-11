from __future__ import annotations

import json
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
    pending_tool_call_id: str | None = None
    last_result: dict[str, Any] | None = None


class LLMTransport(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str | list[ToolCall]: ...


def _build_messages(question: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
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
    approved_tool_call_id: str | None = None,
) -> LoopOutcome:
    schemas = tool_schemas()
    if approved_sql is not None:
        messages = list(history)
        result = tools.dispatch("run_sql", {"sql": approved_sql})
        trace: list[dict[str, Any]] = [{"step": "tool.result", "tool": "run_sql", "status": result.get("status"), "approved": True}]
        messages.append(_tool_result_message(approved_tool_call_id or "run_sql", "run_sql", result))
    else:
        messages = _build_messages(question, history)
        trace = []

    for _ in range(max_iterations):
        reply = transport.complete(messages, schemas)
        if isinstance(reply, str):
            messages.append({"role": "assistant", "content": reply})
            return LoopOutcome(status="completed", answer=reply, trace=trace, messages=messages)

        messages.append({"role": "assistant", "content": "", "tool_calls": [_serialize_call(call) for call in reply]})
        for call in reply:
            trace.append({"step": "tool.call", "tool": call.name, "arguments": call.arguments})
            if call.name == "run_sql":
                sql = str(call.arguments.get("sql") or "")
                return LoopOutcome(
                    status="requires_approval",
                    answer="请先 review 这条 SQL，确认后授权执行，或拒绝本次执行。",
                    trace=trace,
                    messages=messages,
                    pending_sql=sql,
                    pending_tool_call_id=call.id,
                )
            result = tools.dispatch(call.name, call.arguments)
            trace.append({"step": "tool.result", "tool": call.name, "status": result.get("status")})
            messages.append(_tool_result_message(call.id, call.name, result))

    return LoopOutcome(status="exhausted", answer="未能在限定步数内完成查询，请缩小问题范围或补充条件。", trace=trace, messages=messages)


def _serialize_call(call: ToolCall) -> dict[str, Any]:
    return {"id": call.id, "name": call.name, "arguments": call.arguments}


def _tool_result_message(tool_call_id: str, name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "name": name, "content": json.dumps(result, ensure_ascii=False)}
