"""Connectivity self-check for the configured OpenAI-compatible gateway.

Exercises the exact code path the app uses (metabase_agent.semantics.llm_client),
including the raw-httpx /responses transport, plus a one-round tool-calling probe.
Run: uv run python scripts/openai_responses_demo.py
"""
from __future__ import annotations

import traceback

from metabase_agent.agent.tools import tool_schemas
from metabase_agent.config.settings import get_settings
from metabase_agent.semantics.llm_client import build_tool_transport, complete


def main() -> None:
    settings = get_settings()
    print("=== Gateway Connectivity Check ===")
    print(f"base_url={settings.openai_base_url}")
    print(f"model={settings.openai_model}")
    print(f"wire_api={settings.openai_wire_api}")
    print(f"api_key_set={bool(settings.openai_api_key)}")
    if not settings.openai_api_key:
        print("\nOPENAI_API_KEY is empty; set it in .env to run this check.")
        return

    try:
        text = complete("你是一个测试助手，只回一句话。", "说一句中文问候。", settings)
        print("\n=== Plain completion OK ===")
        print(text)
    except Exception as exc:
        print("\n=== Plain completion FAILED ===")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return

    try:
        transport = build_tool_transport(settings)
        reply = transport.complete(
            [{"role": "user", "content": "列出有哪些数据库。"}],
            tool_schemas(),
        )
        print("\n=== Tool-calling probe OK ===")
        print("tool_calls" if isinstance(reply, list) else "text", "->", reply)
    except Exception as exc:
        print("\n=== Tool-calling probe FAILED ===")
        print(f"{type(exc).__name__}: {exc}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
