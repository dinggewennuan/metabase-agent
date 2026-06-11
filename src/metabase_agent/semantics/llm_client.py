"""Unified LLM access for both wire protocols.

The "responses" path intentionally uses raw httpx instead of the OpenAI SDK:
the OpenAI-compatible gateway used in production does not accept SDK-shaped
/responses requests. Do not "simplify" this back to the SDK.
"""
from __future__ import annotations

from typing import Any, Literal, cast

import httpx
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from metabase_agent.config.settings import Settings

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
