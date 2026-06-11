"""Unified LLM access for both wire protocols.

The "responses" path intentionally uses raw httpx instead of the OpenAI SDK:
the OpenAI-compatible gateway used in production does not accept SDK-shaped
/responses requests. Do not "simplify" this back to the SDK.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from metabase_agent.config.settings import Settings

if TYPE_CHECKING:
    from metabase_agent.agent.tool_loop import ToolCall

ReasoningEffort = Literal["high", "xhigh"]


def reasoning_effort(model: str) -> ReasoningEffort:
    normalized = model.lower()
    if normalized.startswith("gpt-5.5"):
        return "xhigh"
    return "high"


def responses_output_text(payload: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in payload.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and content.get("type") == "output_text":
                parts.append(str(content.get("text", "")))
    return "\n".join(part for part in parts if part).strip()


def complete(system_prompt: str, user_content: str, settings: Settings, *, json_mode: bool = False) -> str:
    """Run a single completion over the configured wire protocol.

    Raises RuntimeError when no API key is configured or the response is empty.
    """
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    if settings.openai_wire_api == "responses":
        return _complete_responses(system_prompt, user_content, settings)
    return _complete_chat(system_prompt, user_content, settings, json_mode=json_mode)


def _complete_responses(system_prompt: str, user_content: str, settings: Settings) -> str:
    response = httpx.post(
        f"{settings.openai_base_url.rstrip('/')}/responses",
        headers={"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"},
        json={
            "model": settings.openai_model,
            "input": f"{system_prompt}\n\n{user_content}",
            "reasoning": {"effort": reasoning_effort(settings.openai_model)},
        },
        timeout=settings.openai_timeout,
    )
    response.raise_for_status()
    text = responses_output_text(response.json())
    if not text:
        raise RuntimeError("empty LLM response")
    return text


def _complete_chat(system_prompt: str, user_content: str, settings: Settings, *, json_mode: bool) -> str:
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url, timeout=settings.openai_timeout)
    messages = cast(
        list[ChatCompletionMessageParam],
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )
    if json_mode:
        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                reasoning_effort=reasoning_effort(settings.openai_model),
                temperature=0,
                messages=messages,
                response_format={"type": "json_object"},
            )
        except Exception:
            response = client.chat.completions.create(
                model=settings.openai_model,
                reasoning_effort=reasoning_effort(settings.openai_model),
                temperature=0,
                messages=messages,
            )
    else:
        response = client.chat.completions.create(
            model=settings.openai_model,
            reasoning_effort=reasoning_effort(settings.openai_model),
            messages=messages,
        )
    text = response.choices[0].message.content
    if not text or not text.strip():
        raise RuntimeError("empty LLM response")
    return text.strip()


def _to_tool_calls(raw: list[tuple[str, str, str]]) -> list[ToolCall]:
    from metabase_agent.agent.tool_loop import ToolCall

    calls: list[ToolCall] = []
    for call_id, name, arguments in raw:
        try:
            parsed = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError:
            parsed = {}
        calls.append(ToolCall(id=call_id, name=name, arguments=parsed if isinstance(parsed, dict) else {}))
    return calls


def _responses_function_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"type": "function", "name": tool["name"], "description": tool["description"], "parameters": tool["parameters"]} for tool in tools]


def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "tool":
            items.append({"type": "function_call_output", "call_id": message.get("tool_call_id"), "output": str(message.get("content") or "")})
        elif role == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                items.append({"type": "function_call", "call_id": call["id"], "name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)})
        else:
            items.append({"role": role, "content": str(message.get("content") or "")})
    return items


class ResponsesToolTransport:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str | list[ToolCall]:
        response = httpx.post(
            f"{self.settings.openai_base_url.rstrip('/')}/responses",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}", "Content-Type": "application/json"},
            json={
                "model": self.settings.openai_model,
                "input": _responses_input(messages),
                "tools": _responses_function_tools(tools),
                "reasoning": {"effort": reasoning_effort(self.settings.openai_model)},
            },
            timeout=self.settings.openai_timeout,
        )
        response.raise_for_status()
        payload = response.json()
        raw_calls = [
            (str(item.get("call_id") or item.get("id") or ""), str(item.get("name") or ""), str(item.get("arguments") or ""))
            for item in payload.get("output", [])
            if isinstance(item, dict) and item.get("type") == "function_call"
        ]
        if raw_calls:
            return _to_tool_calls(raw_calls)
        return responses_output_text(payload)


class ChatToolTransport:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str | list[ToolCall]:
        client = OpenAI(api_key=self.settings.openai_api_key, base_url=self.settings.openai_base_url, timeout=self.settings.openai_timeout)
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            reasoning_effort=reasoning_effort(self.settings.openai_model),
            messages=cast(Any, _chat_messages(messages)),
            tools=cast(Any, [{"type": "function", "function": tool} for tool in tools]),
        )
        choice = response.choices[0].message
        if choice.tool_calls:
            raw: list[tuple[str, str, str]] = []
            for call in choice.tool_calls:
                function = getattr(call, "function", None)
                if function is not None:
                    raw.append((call.id, function.name, function.arguments or ""))
            if raw:
                return _to_tool_calls(raw)
        return (choice.content or "").strip()


def _chat_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            converted.append({
                "role": "assistant",
                "content": message.get("content") or "",
                "tool_calls": [
                    {"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}
                    for call in message["tool_calls"]
                ],
            })
        elif message.get("role") == "tool":
            converted.append({"role": "tool", "tool_call_id": message.get("tool_call_id"), "content": str(message.get("content") or "")})
        else:
            converted.append({"role": message.get("role"), "content": str(message.get("content") or "")})
    return converted


def build_tool_transport(settings: Settings) -> ResponsesToolTransport | ChatToolTransport:
    if settings.openai_wire_api == "responses":
        return ResponsesToolTransport(settings)
    return ChatToolTransport(settings)
