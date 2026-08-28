"""Minimal terminal chrome. Rich if installed, plain text otherwise.

Does not take over the screen (no TUI). Just prefixes, colors, and
line discipline so status lines never splice into a spoken transcript.

Colors are warm orange on a dark terminal so they sit next to a gray
background and orange shell text. Summaries are never ellipsized.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

_MARKUP_TAGS = re.compile(
    r"\[/?("
    r"bold cyan|bold magenta|bold dark_orange|bold orange1|"
    r"yellow|gold1|red|bold orange_red1|dim|grey54|"
    r"gf\.you|gf\.grok|gf\.tool|gf\.dim|gf\.err"
    r")\]"
)

try:
    from rich.console import Console
    from rich.theme import Theme

    _THEME = Theme(
        {
            "gf.you": "bold dark_orange",
            "gf.grok": "bold orange1",
            "gf.tool": "gold1",
            "gf.dim": "grey54",
            "gf.err": "bold orange_red1",
        }
    )
    _console = Console(
        theme=_THEME,
        highlight=False,
        soft_wrap=True,
        emoji=False,
        color_system="256",
    )
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


def _strip_markup(markup: str) -> str:
    return _MARKUP_TAGS.sub("", markup).replace("[/]", "")


def _print(markup: str, **kwargs: Any) -> None:
    _close_asst()
    if _RICH:
        _console.print(markup, overflow="fold", crop=False, **kwargs)
    else:
        print(_strip_markup(markup), **kwargs)


def _print_plain_body(prefix_markup: str, body: str) -> None:
    """Print a styled prefix, then the full body. Never ellipsize."""
    _close_asst()
    text = body or ""
    lines = text.splitlines() or ([text] if text else [""])
    if _RICH:
        _console.print(prefix_markup, end="", overflow="ignore", crop=False)
        sys.stdout.write(lines[0] + "\n")
        for line in lines[1:]:
            sys.stdout.write("         " + line + "\n")
        sys.stdout.flush()
        return
    print(_strip_markup(prefix_markup) + lines[0])
    for line in lines[1:]:
        print("         " + line)


def banner(grok_bin: str | None) -> None:
    _print("[gf.dim]grapefruit[/]  voice + terminal  ·  /help")
    if grok_bin:
        _print(f"[gf.dim]cli[/]        {grok_bin}")


def user(text: str, *, spoken: bool = False) -> None:
    tag = " (speech)" if spoken else ""
    _print_plain_body(f"  [gf.you]you[/]{tag}   ", text)


def assistant_delta(piece: str) -> None:
    global _asst_open
    if not piece:
        return
    if not _asst_open:
        if _RICH:
            _console.print("  [gf.grok]grok[/]  ", end="", overflow="fold", crop=False)
        else:
            print("\n  grok  ", end="", flush=True)
        _asst_open = True
    if _RICH:
        _console.print(piece, end="", markup=False, highlight=False, overflow="fold", crop=False)
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
        extra = json.dumps(args, ensure_ascii=False)
    _print_plain_body(f"  [gf.tool]tool[/]  {name}  ", extra)


def tool_result(text: str, limit: int | None = None) -> None:
    body = (text or "").rstrip()
    _print_plain_body("  [gf.dim]done[/]  ", body)


def cli(line: str) -> None:
    _print_plain_body("  [gf.dim]cli[/]   ", line)


def status(line: str) -> None:
    _print_plain_body("  [gf.dim]···[/]  ", line)


def error(line: str) -> None:
    _print_plain_body("  [gf.err]err[/]   ", line)


def session(line: str) -> None:
    _print_plain_body("  [gf.dim]──[/]   ", line)


def info(line: str) -> None:
    _print_plain_body("  [gf.dim]    [/]  ", line)
