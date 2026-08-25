"""Client-side web/X search when Voice emits them as function calls.

Native `{type: web_search}` runs on xAI during a Voice turn. After a custom
tool like run_grok, Voice sometimes calls `web_search` as a *function* instead.
Those land in handle_tool and used to return 'Unknown tool'.
"""

from __future__ import annotations

import os

from grapefruit.env import get_xai_api_key

TEXT_MODEL = os.getenv("GROK_TEXT_MODEL", "grok-4.6")


def _brief(query: str, tool: dict, label: str) -> str:
    from openai import OpenAI

    client = OpenAI(base_url="https://api.x.ai/v1", api_key=get_xai_api_key())
    prompt = (
        f"{label} for: {query}\n"
        "Return a short briefing with the main findings and source names or URLs. "
        "No markdown tables."
    )
    resp = client.responses.create(
        model=TEXT_MODEL,
        input=prompt,
        tools=[tool],
    )
    text = (getattr(resp, "output_text", None) or "").strip()
    if text:
        return text
    return f"{label} returned no text for: {query}"


def web_search(query: str, **_kwargs) -> str:
    query = (query or "").strip()
    if not query:
        return "web_search needs a query."
    return _brief(query, {"type": "web_search"}, "Web search")


def x_search(query: str, **_kwargs) -> str:
    query = (query or "").strip()
    if not query:
        return "x_search needs a query."
    return _brief(query, {"type": "x_search"}, "X search")
