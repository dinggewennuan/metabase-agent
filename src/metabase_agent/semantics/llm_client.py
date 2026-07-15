"""Unified LLM access for the supported wire protocols.

Wire protocols (OPENAI_WIRE_API):
- "chat_completions"        /chat/completions via the OpenAI SDK (default)
- "chat_completions_httpx"  /chat/completions via raw httpx — for gateways
  whose WAF rejects the SDK's request fingerprint (x-stainless-* headers,
  SDK User-Agent) with 403 "Your request was blocked"
- "responses"               /responses via raw httpx

The raw-httpx paths are intentional: the OpenAI-compatible gateways used in
production reject SDK-shaped requests. Do not "simplify" them back to the SDK.
"""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Literal, cast

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from metabase_agent.config.settings import Settings

if TYPE_CHECKING:
    from metabase_agent.agent.tool_loop import ToolCall

_LOGGER = logging.getLogger("metabase_agent")

ReasoningEffort = Literal["high", "xhigh"]

# Transport-level flakiness (proxy drops the TLS handshake / connection with
# "UNEXPECTED_EOF" or "Server disconnected") is intermittent on the akool
# gateway, so a few spaced retries recover most turns that would otherwise die.
_TRANSPORT_RETRY_BACKOFF = (0.4, 1.0, 2.0)


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
    if settings.openai_wire_api == "chat_completions_httpx":
        return _complete_chat_httpx(system_prompt, user_content, settings, json_mode=json_mode)
    return _complete_chat(system_prompt, user_content, settings, json_mode=json_mode)


# Explicit neutral User-Agent: httpx's default is "python-httpx/x.y" and
# "python-*" agents are a stock gateway-WAF block rule (the same WAFs accept
# Postman/curl). Never let the default leak through.
_HTTPX_HEADERS_UA = "metabase-agent/0.1"


def _llm_headers(settings: Settings) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
        "User-Agent": _HTTPX_HEADERS_UA,
    }


def _complete_responses(system_prompt: str, user_content: str, settings: Settings) -> str:
    response = httpx.post(
        f"{settings.openai_base_url.rstrip('/')}/responses",
        headers=_llm_headers(settings),
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


def _post_chat_completions(settings: Settings, body: dict[str, Any]) -> dict[str, Any]:
    # Plain httpx with only auth/content-type/UA: no SDK fingerprint headers
    # (x-stainless-*) for a gateway WAF to block.
    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
    headers = _llm_headers(settings)
    response = _post_with_transport_retry(url, headers, body, settings.openai_timeout)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _post_with_transport_retry(url: str, headers: dict[str, str], body: dict[str, Any], timeout: float) -> httpx.Response:
    """POST with retries on TransportError only.

    A TransportError means no response was received (TLS handshake / connection
    dropped), so retrying is safe — no risk of duplicating a side effect. HTTP
    status errors are NOT retried here (handled by the 4xx-fallback caller).
    """
    for attempt, backoff in enumerate((*_TRANSPORT_RETRY_BACKOFF, None)):
        try:
            return httpx.post(url, headers=headers, json=body, timeout=timeout)
        except httpx.TransportError as exc:
            if backoff is None:
                raise
            _LOGGER.warning("llm transport error (attempt %s), retrying in %ss: %s", attempt + 1, backoff, exc)
            time.sleep(backoff)
    raise AssertionError("unreachable")  # pragma: no cover


# Body keys beyond the bare {model, messages, tools} minimum. Some gateways
# 4xx on any of them; the verified-working request shape is the minimal one,
# so on a 4xx we retry once without the extras.
_OPTIONAL_CHAT_KEYS = ("reasoning_effort", "response_format", "temperature")


def _post_chat_completions_with_fallback(settings: Settings, body: dict[str, Any]) -> dict[str, Any]:
    try:
        return _post_chat_completions(settings, body)
    except httpx.HTTPStatusError as exc:
        minimal = {key: value for key, value in body.items() if key not in _OPTIONAL_CHAT_KEYS}
        if exc.response.status_code >= 500 or minimal == body:
            raise
        return _post_chat_completions(settings, minimal)


def _chat_payload_message(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return {}
    message = choices[0].get("message")
    return message if isinstance(message, dict) else {}


def _complete_chat_httpx(system_prompt: str, user_content: str, settings: Settings, *, json_mode: bool) -> str:
    body: dict[str, Any] = {
        "model": settings.openai_model,
        "reasoning_effort": reasoning_effort(settings.openai_model),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    if json_mode:
        # response_format/temperature/reasoning_effort are all dropped by the
        # fallback if the gateway rejects them; JSON shape is then enforced by
        # the prompt alone, same as the SDK path's degradation.
        payload = _post_chat_completions_with_fallback(settings, {**body, "temperature": 0, "response_format": {"type": "json_object"}})
    else:
        payload = _post_chat_completions_with_fallback(settings, body)
    text = str(_chat_payload_message(payload).get("content") or "")
    if not text.strip():
        raise RuntimeError("empty LLM response")
    return text.strip()


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
            headers=_llm_headers(self.settings),
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


class ChatHttpxToolTransport:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def complete(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> str | list[ToolCall]:
        payload = _post_chat_completions_with_fallback(
            self.settings,
            {
                "model": self.settings.openai_model,
                "reasoning_effort": reasoning_effort(self.settings.openai_model),
                "messages": _chat_messages(messages),
                "tools": [{"type": "function", "function": tool} for tool in tools],
            },
        )
        message = _chat_payload_message(payload)
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            raw: list[tuple[str, str, str]] = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if isinstance(function, dict):
                    raw.append((str(call.get("id") or ""), str(function.get("name") or ""), str(function.get("arguments") or "")))
            if raw:
                return _to_tool_calls(raw)
        return str(message.get("content") or "").strip()


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


def build_tool_transport(settings: Settings) -> ResponsesToolTransport | ChatToolTransport | ChatHttpxToolTransport:
    if settings.openai_wire_api == "responses":
        return ResponsesToolTransport(settings)
    if settings.openai_wire_api == "chat_completions_httpx":
        return ChatHttpxToolTransport(settings)
    return ChatToolTransport(settings)
