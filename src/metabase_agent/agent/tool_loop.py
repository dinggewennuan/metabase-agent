from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from metabase_agent.agent.tools import AgentTools, tool_schemas

_SYSTEM_PROMPT = (
    "你是 Metabase 只读数据分析 Agent。\n"
    "\n"
    "工作方式：\n"
    "- 先用 list_databases / list_tables / list_fields 探查元数据，确认库、schema/表、字段和口径。\n"
    "- 简单聚合（count/sum/avg/min/max，可按时间分组）用 run_aggregation。\n"
    "- 任何 run_aggregation 表达不了的分析——按任意字段分组/拆分、多条件过滤、多表 join、"
    "货币或单位换算、占比、环比、窗口函数等——直接写只读 SQL 用 run_sql。"
    "run_sql 能表达任意只读分析，是通用手段，不要因为某种分析没有专门工具就说做不了。\n"
    "- run_sql 已内置人工审批：系统会在执行前自动把 SQL 展示给用户 review 并等待授权。"
    "所以准备好 SQL 就直接调用 run_sql，不要再用文字问一遍\"是否确认执行\"（否则用户要确认两次）。"
    "只有还缺信息、无法把请求落成 SQL 时，才用文字向用户提问。\n"
    "\n"
    "口径与诚实（对所有问题一致适用）：\n"
    "- 每次回答都说明数据口径：数据库、schema/表、时间字段与范围、过滤条件、聚合方式、单位。\n"
    "- 不要编造数据库里的数据；工具返回错误或 not_found 时如实说明并建议下一步。\n"
    "- 当分析需要数据库里没有的外部知识（汇率、分类/枚举含义、阈值、单位换算、行业分组等）时，"
    "你可以用自己已知的值来完成分析，但必须：(1) 明确标注这些是\"估算/假设值，非权威、可能过时，请核对\"；"
    "(2) 逐一列出你用到的每个值；(3) 优先使用用户提供的值或数据库中已有的对应表，估算只作为最后手段。"
    "绝不把估算值当作准确值呈现。"
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


def _build_messages(
    question: str,
    history: list[dict[str, Any]],
    *,
    memory_context: str = "",
    skills_context: str = "",
) -> list[dict[str, Any]]:
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
                # Multi-step analyses legitimately need several SQL statements,
                # each with its own approval. Say so explicitly — otherwise the
                # repeated review prompt reads like a broken loop.
                prompt_line = (
                    "上一条已授权的 SQL 已执行完成。为了继续分析，需要再执行一条**新的** SQL——请 review 后授权执行，或拒绝。"
                    if approved_sql is not None
                    else "请先 review 这条 SQL，确认后授权执行，或拒绝本次执行。"
                )
                database_name = str(call.arguments.get("database_name") or "")
                database_line = f"\n目标数据库：`{database_name}`" if database_name else ""
                # The SQL must be IN the message: the reviewer can't approve
                # what they can't see (query_result.sql alone only shows up in
                # the raw JSON panel).
                approval_answer = f"{prompt_line}{database_line}\n\n```sql\n{sql}\n```"
                yield "outcome", LoopOutcome(
                    status="requires_approval",
                    answer=approval_answer,
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
