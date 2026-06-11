from __future__ import annotations

import traceback
from typing import Literal

import httpx
import openai
from openai import OpenAI
from openai.types.shared_params.reasoning import Reasoning

from metabase_agent.config.settings import get_settings


ReasoningEffort = Literal["high", "xhigh"]


def _reasoning_effort(model: str) -> ReasoningEffort:
    if model.lower().startswith("gpt-5.5"):
        return "xhigh"
    return "high"


def _log_request(request: httpx.Request) -> None:
    print("\n=== HTTP Request ===")
    print(f"{request.method} {request.url}")
    for key, value in request.headers.items():
        if key.lower() == "authorization":
            value = "Bearer ***"
        print(f"{key}: {value}")
    print("--- body ---")
    print(request.content.decode("utf-8", errors="replace"))


def _log_response(response: httpx.Response) -> None:
    response.read()
    print("\n=== HTTP Response ===")
    print(f"status: {response.status_code}")
    for key, value in response.headers.items():
        print(f"{key}: {value}")
    print("--- body ---")
    print(response.text)


def main() -> None:
    settings = get_settings()
    print("=== OpenAI Responses SDK Demo ===")
    print(f"openai_version={openai.__version__}")
    print(f"base_url={settings.openai_base_url}")
    print(f"model={settings.openai_model}")
    print(f"api_key_set={bool(settings.openai_api_key)}")

    http_client = httpx.Client(
        event_hooks={
            "request": [_log_request],
            "response": [_log_response],
        },
        timeout=180,
    )
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        http_client=http_client,
    )

    try:
        response = client.responses.create(
            model=settings.openai_model,
            reasoning=Reasoning(effort=_reasoning_effort(settings.openai_model)),
            input="Write a short bedtime story about a unicorn.",
        )
    except Exception as exc:
        print("\n=== Exception ===")
        print(f"type={type(exc).__name__}")
        print(f"message={exc}")
        print("--- traceback ---")
        traceback.print_exc()
        raise

    print("\n=== Parsed Output ===")
    print(response.output_text)


if __name__ == "__main__":
    main()
