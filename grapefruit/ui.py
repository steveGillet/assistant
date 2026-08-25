"""Minimal terminal chrome. Rich if installed, plain text otherwise.

Does not take over the screen (no TUI). Just prefixes, colors, and
line discipline so status lines never splice into a spoken transcript.
"""

from __future__ import annotations

import json
import sys
from typing import Any

try:
    from rich.console import Console

    _console = Console(highlight=False, soft_wrap=True, emoji=False)
    _RICH = True
except ImportError:
    _console = None
    _RICH = False

_asst_open = False


def _close_asst() -> None:
    global _asst_open
    if not _asst_open:
        return
    if _RICH:
        _console.print()
    else:
        print(flush=True)
    _asst_open = False


def _print(markup: str, **kwargs: Any) -> None:
    _close_asst()
    if _RICH:
        _console.print(markup, **kwargs)
    else:
        plain = (
            markup.replace("[bold cyan]", "")
            .replace("[bold magenta]", "")
            .replace("[yellow]", "")
            .replace("[red]", "")
            .replace("[dim]", "")
            .replace("[/]", "")
        )
        print(plain, **kwargs)


def banner(grok_bin: str | None) -> None:
    _print("[dim]grapefruit[/]  voice + terminal  ·  /help")
    if grok_bin:
        _print(f"[dim]cli[/]        {grok_bin}")


def user(text: str, *, spoken: bool = False) -> None:
    tag = " (speech)" if spoken else ""
    _print(f"  [bold cyan]you[/]{tag}   {text}")


def assistant_delta(piece: str) -> None:
    global _asst_open
    if not piece:
        return
    if not _asst_open:
        if _RICH:
            _console.print("  [bold magenta]grok[/]  ", end="")
        else:
            print("\n  grok  ", end="", flush=True)
        _asst_open = True
    if _RICH:
        _console.print(piece, end="", markup=False, highlight=False)
    else:
        sys.stdout.write(piece)
        sys.stdout.flush()


def assistant_end() -> None:
    _close_asst()


def heard(text: str) -> None:
    user(text, spoken=True)


def tool(name: str, args: dict | None = None) -> None:
    extra = ""
    if args:
        raw = json.dumps(args, ensure_ascii=False)
        if len(raw) > 140:
            raw = raw[:137] + "..."
        extra = f"  [dim]{raw}[/]"
    _print(f"  [yellow]tool[/]  {name}{extra}")


def tool_result(text: str, limit: int = 500) -> None:
    snippet = (text or "").replace("\n", " ").strip()
    if len(snippet) > limit:
        snippet = snippet[: limit - 3] + "..."
    _print(f"  [dim]done[/]  {snippet}")


def cli(line: str) -> None:
    _print(f"  [dim]cli[/]   {line}")


def status(line: str) -> None:
    _print(f"  [dim]···[/]  {line}")


def error(line: str) -> None:
    _print(f"  [red]err[/]   {line}")


def session(line: str) -> None:
    _print(f"  [dim]──[/]   {line}")


def info(line: str) -> None:
    _print(f"  [dim]    [/]  {line}")
