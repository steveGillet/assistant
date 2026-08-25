"""Save, compact, and search Voice/terminal conversations as JSONL on disk."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from grapefruit.paths import CONVERSATIONS, ensure_dirs

CONTEXT_TOKEN_LIMIT = int(os.getenv("GROK_VOICE_CONTEXT_TOKENS", "20000"))
COMPACT_RATIO = float(os.getenv("GROK_VOICE_COMPACT_RATIO", "0.7"))
KEEP_LAST = int(os.getenv("GROK_VOICE_KEEP_LAST", "12"))
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // CHARS_PER_TOKEN) if text else 0


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def slugify(text: str, max_len: int = 40) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return (text[:max_len].strip("-") or "conversation")


@dataclass
class Message:
    ts: str
    role: str
    text: str
    source: str = "unknown"
    name: str | None = None
    kind: str | None = None

    def tokens(self) -> int:
        return estimate_tokens(self.text)


@dataclass
class ConversationMeta:
    id: str
    path: str
    title: str
    started: str
    updated: str
    summary: str = ""
    tokens_est: int = 0


def _index_path(root: Path) -> Path:
    return root / "index.json"


def load_index(root: Path | None = None) -> list[ConversationMeta]:
    root = root or CONVERSATIONS
    path = _index_path(root)
    if not path.is_file():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    out = []
    for item in raw:
        try:
            out.append(ConversationMeta(**item))
        except TypeError:
            continue
    return out


def save_index(items: list[ConversationMeta], root: Path | None = None) -> None:
    root = root or CONVERSATIONS
    root.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in items]
    _index_path(root).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_messages(path: Path) -> list[Message]:
    messages: list[Message] = []
    if not path.is_file():
        return messages
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        messages.append(
            Message(
                ts=str(data.get("ts", "")),
                role=str(data.get("role", "")),
                text=str(data.get("text", "")),
                source=str(data.get("source") or "unknown"),
                name=data.get("name"),
                kind=data.get("kind"),
            )
        )
    return messages


def extractive_summary(messages: list[Message], max_chars: int = 2000) -> str:
    if not messages:
        return ""
    parts: list[str] = []
    for msg in messages:
        if msg.kind == "compact":
            parts.append(msg.text)
            continue
        prefix = msg.role
        if msg.name:
            prefix = f"{msg.role}:{msg.name}"
        snippet = " ".join(msg.text.split())
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        parts.append(f"{prefix}: {snippet}")
    blob = "\n".join(parts)
    if len(blob) > max_chars:
        blob = blob[: max_chars - 3] + "..."
    return blob


def compact_messages(
    messages: list[Message],
    *,
    keep_last: int = KEEP_LAST,
    now: datetime | None = None,
) -> list[Message]:
    if len(messages) <= keep_last + 1:
        return messages
    older, recent = messages[:-keep_last], messages[keep_last * -1 :]
    summary = extractive_summary(older)
    compact = Message(
        ts=(now or utc_now()).isoformat(),
        role="system",
        source="compact",
        kind="compact",
        text="Earlier conversation (compacted):\n" + summary,
    )
    return [compact] + list(recent)


def render_for_restore(messages: list[Message]) -> str:
    lines = []
    for msg in messages:
        if msg.kind == "compact":
            lines.append(msg.text)
            continue
        who = "User" if msg.role == "user" else "Assistant" if msg.role == "assistant" else msg.role
        if msg.source and msg.role == "user":
            who = f"User ({msg.source})"
        if msg.name:
            who = f"{who} [{msg.name}]"
        lines.append(f"{who}: {msg.text}")
    return "\n".join(lines)


def _query_date(query: str) -> date | None:
    q = query.lower()
    today = date.today()
    if "yesterday" in q:
        return today - timedelta(days=1)
    if "today" in q:
        return today
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", q)
    if match:
        try:
            return date.fromisoformat(match.group(1))
        except ValueError:
            return None
    return None


def score_conversation(meta: ConversationMeta, query: str, messages: list[Message] | None = None) -> float:
    if not query.strip():
        return 0.0
    q = query.lower().strip()
    stop = {
        "the", "a", "an", "on", "about", "we", "had", "conversation",
        "yesterday", "today", "restore", "load", "please",
    }
    words = [w for w in re.split(r"\W+", q) if w and w not in stop]
    hay = f"{meta.title} {meta.summary} {meta.id}".lower()
    if messages:
        hay += " " + " ".join(m.text.lower() for m in messages[:8])
    score = 0.0
    if q in hay:
        score += 5
    for word in words:
        stem = word.rstrip("s")
        if word in hay or (stem and stem in hay):
            score += 1
    wanted = _query_date(query)
    if wanted:
        try:
            started = datetime.fromisoformat(meta.started.replace("Z", "+00:00")).date()
        except ValueError:
            started = None
        if started == wanted:
            score += 3
        elif started:
            score -= 1
    return score


class ConversationLog:
    def __init__(self, root: Path | None = None, *, token_limit: int = CONTEXT_TOKEN_LIMIT):
        self.root = root or CONVERSATIONS
        ensure_dirs()
        self.root.mkdir(parents=True, exist_ok=True)
        self.token_limit = token_limit
        self.meta: ConversationMeta | None = None
        self.messages: list[Message] = []
        self._titled_from_user = False

    @property
    def path(self) -> Path | None:
        if not self.meta:
            return None
        return self.root / self.meta.path

    def start(self, title_hint: str = "") -> ConversationMeta:
        now = utc_now()
        stamp = now.strftime("%Y-%m-%dT%H%M%SZ")
        slug = slugify(title_hint) if title_hint else "session"
        conv_id = f"{stamp}_{slug}"
        self.meta = ConversationMeta(
            id=conv_id,
            path=f"{conv_id}.jsonl",
            title=title_hint.strip() or "Untitled session",
            started=now.isoformat(),
            updated=now.isoformat(),
            summary="",
            tokens_est=0,
        )
        self.messages = []
        self._titled_from_user = False
        (self.root / self.meta.path).write_text("", encoding="utf-8")
        self._write_index()
        return self.meta

    def append(
        self,
        role: str,
        text: str,
        *,
        source: str = "unknown",
        name: str | None = None,
        kind: str | None = None,
    ) -> Message | None:
        text = (text or "").strip()
        if not text:
            return None
        if self.meta is None:
            self.start(text if role == "user" else "")
        assert self.meta is not None
        if role == "user" and not self._titled_from_user:
            self.meta.title = text[:80]
            self._titled_from_user = True
            old = self.path
            slug = slugify(text)
            new_id = f"{self.meta.id.split('_', 1)[0]}_{slug}"
            new_path = f"{new_id}.jsonl"
            if old and old.exists() and new_path != self.meta.path:
                dest = self.root / new_path
                if not dest.exists():
                    old.rename(dest)
            self.meta.id = new_id
            self.meta.path = new_path
        msg = Message(
            ts=utc_now().isoformat(),
            role=role,
            text=text,
            source=source,
            name=name,
            kind=kind,
        )
        self.messages.append(msg)
        self._append_line(msg)
        self.maybe_compact()
        self.meta.updated = utc_now().isoformat()
        self.meta.tokens_est = sum(m.tokens() for m in self.messages)
        self.meta.summary = extractive_summary(self.messages, max_chars=400)
        self._write_index()
        return msg

    def maybe_compact(self) -> bool:
        if not self.messages or not self.meta:
            return False
        total = sum(m.tokens() for m in self.messages)
        if total < int(self.token_limit * COMPACT_RATIO):
            return False
        compacted = compact_messages(self.messages)
        if compacted == self.messages:
            return False
        self.messages = compacted
        self._rewrite()
        return True

    def set_title(self, title: str) -> None:
        if not title.strip() or not self.meta:
            return
        self.meta.title = title.strip()[:80]
        self.meta.updated = utc_now().isoformat()
        self._write_index()

    def context_for_model(self) -> str:
        return render_for_restore(self.messages)

    def _append_line(self, msg: Message) -> None:
        path = self.path
        if path is None:
            return
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(msg), ensure_ascii=False) + "\n")

    def _rewrite(self) -> None:
        path = self.path
        if path is None:
            return
        lines = [json.dumps(asdict(m), ensure_ascii=False) for m in self.messages]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def _write_index(self) -> None:
        if not self.meta:
            return
        items = [m for m in load_index(self.root) if m.id != self.meta.id]
        items.append(self.meta)
        items.sort(key=lambda m: m.updated, reverse=True)
        save_index(items, self.root)


def list_conversations(root: Path | None = None, limit: int = 15) -> list[ConversationMeta]:
    items = load_index(root)
    items.sort(key=lambda m: m.updated, reverse=True)
    return items[:limit]


def _short_title(title: str, width: int = 72) -> str:
    title = (title or "").replace("\n", " ").strip()
    if len(title) <= width:
        return title
    return title[: width - 1].rstrip() + "…"


def format_conversation_list(items: list[ConversationMeta]) -> str:
    if not items:
        return "No saved conversations yet."
    lines = [
        "Saved chats. /restore 1 picks a row; /restore raspberry searches.",
        "",
    ]
    for i, meta in enumerate(items, start=1):
        when = meta.started[:10] if meta.started else "?"
        lines.append(f"  {i}  {when}  {_short_title(meta.title)}")
    return "\n".join(lines)


def find_conversation(query: str, root: Path | None = None) -> ConversationMeta | None:
    root = root or CONVERSATIONS
    items = load_index(root)
    if not items:
        return None
    q = query.strip()
    if not q:
        items.sort(key=lambda m: m.updated, reverse=True)
        return items[0]
    if q.isdigit():
        ranked = list_conversations(root)
        n = int(q)
        if 1 <= n <= len(ranked):
            return ranked[n - 1]
        return None
    scored: list[tuple[float, ConversationMeta]] = []
    for meta in items:
        path = root / meta.path
        msgs = read_messages(path) if path.is_file() else []
        scored.append((score_conversation(meta, query, msgs), meta))
    scored.sort(key=lambda pair: (pair[0], pair[1].updated), reverse=True)
    best, meta = scored[0]
    if best <= 0:
        return None
    return meta


def load_restore_text(query: str, root: Path | None = None) -> tuple[str, ConversationMeta | None]:
    root = root or CONVERSATIONS
    if not query.strip():
        listing = format_conversation_list(list_conversations(root))
        return listing, None
    meta = find_conversation(query, root)
    if not meta:
        listing = format_conversation_list(list_conversations(root))
        return f"No match for {query!r}. Recent:\n{listing}", None
    messages = read_messages(root / meta.path)
    body = render_for_restore(messages)
    header = f"Restored conversation {meta.title!r} from {meta.started[:10]} (id {meta.id})."
    return f"{header}\n\n{body}", meta
