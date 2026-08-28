"""Save, compact, and search Voice/terminal conversations as JSONL on disk."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from grapefruit.paths import CONVERSATIONS, ensure_dirs
from grapefruit.protocol import is_mute_command, is_quit_command, is_unmute_command

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


_RESTORE_REQUEST = re.compile(
    r"^\s*(?:please\s+|just\s+|okay\s+|ok\s+|hey\s+)?"
    r"(?:can you\s+|could you\s+|would you\s+)?"
    r"(?:please\s+)?"
    r"(?:restore|resume|load|pull up|bring back|open)\b",
    re.I,
)
_RESTORE_ASSISTANT = re.compile(
    r"^(restored the |i'll (bring|pull|load|open) that |i will (bring|pull|load) |"
    r"i'm listening|i am listening|what can i help|happy to help|"
    r"i'm here and ready|what were you going to)",
    re.I,
)


def is_restore_request(text: str) -> bool:
    """True when the line is asking to load an earlier chat, not doing work."""
    t = " ".join((text or "").split()).strip().lower()
    if not t:
        return False
    t = t.replace("’", "'")
    if t in {"resume", "restore", "load", "/resume", "/restore", "/load"}:
        return True
    if _RESTORE_REQUEST.search(t):
        return True
    if re.search(r"\b(restore|resume|load)\b", t) and re.search(
        r"\b(chat|conversation|session)\b", t
    ):
        return True
    words = re.findall(r"[a-z0-9']+", t)
    if "conversation" in words and len(words) <= 10:
        return True
    return False


def is_restore_noise(msg: Message) -> bool:
    if msg.kind in {"restore", "compact"}:
        return True
    if msg.name == "restore_conversation":
        return True
    text = msg.text or ""
    if msg.role == "tool" and "Prior conversation loaded" in text:
        return True
    if msg.role == "user" and is_restore_request(text):
        return True
    if msg.role == "assistant" and _RESTORE_ASSISTANT.match(text.strip()):
        return True
    if msg.role == "assistant" and re.match(
        r"^Still here\.\s+Picked up ", text.strip()
    ):
        return True
    return False


def _is_low_content_user(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if is_mute_command(t) or is_quit_command(t) or is_unmute_command(t):
        return True
    if is_restore_request(t):
        return True
    words = re.findall(r"[a-z0-9']+", t.lower())
    if not words:
        return True
    if "grapefruit" in words:
        return True
    if len(words) <= 2 and words[0] in {
        "you", "what", "yeah", "ok", "okay", "um", "uh", "huh", "hm", "hmm",
    }:
        return True
    return False


def unique_messages(messages: list[Message]) -> list[Message]:
    return [m for m in messages if not is_restore_noise(m)]


def last_exchange(messages: list[Message]) -> tuple[str, str]:
    """Most recent real user and assistant lines, skipping restore/mute noise."""
    last_user = ""
    last_asst = ""
    for msg in reversed(messages):
        if is_restore_noise(msg):
            continue
        if msg.role == "assistant" and not last_asst:
            text = (msg.text or "").strip()
            if text and not _RESTORE_ASSISTANT.match(text):
                last_asst = text
        elif msg.role == "user" and not last_user:
            text = (msg.text or "").strip()
            if text and not _is_low_content_user(text):
                last_user = text
        if last_user and last_asst:
            break
    return last_user, last_asst


def format_restore_recap(messages: list[Message]) -> str:
    last_user, last_asst = last_exchange(messages)
    lines: list[str] = []
    if last_user:
        lines.append(f"Last you: {last_user}")
    if last_asst:
        lines.append(f"Last grok: {last_asst}")
    if not lines:
        return "No prior messages in that chat."
    return "\n".join(lines)


def recap_from_restore_text(text: str) -> str:
    """Use only the header recap, not nested Last you/Last grok dumps."""
    last_you = ""
    last_grok = ""
    for line in (text or "").splitlines():
        if not last_you and line.startswith("Last you:"):
            last_you = line
        elif not last_grok and line.startswith("Last grok:"):
            last_grok = line
        if last_you and last_grok:
            break
    return "\n".join(part for part in (last_you, last_grok) if part)


def clip_spoken(text: str, width: int = 360) -> str:
    text = " ".join((text or "").split())
    if len(text) <= width:
        return text
    cut = text[:width]
    for punct in (". ", "! ", "? "):
        i = cut.rfind(punct)
        if i >= 60:
            return cut[: i + 1].strip()
    return cut.rsplit(" ", 1)[0].strip()


def spoken_job_recap(text: str, width: int = 400) -> str:
    """Turn a long CLI dump into a short spoken recap of the actual answer."""
    cleaned = re.sub(r"```.*?```", " ", text or "", flags=re.S)
    cleaned = " ".join(cleaned.split())
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    thinking = re.compile(
        r"^(I'll |I will |Let me |I'm going to |I am going to |"
        r"The broad search|so I'll |I'll have )",
        re.I,
    )
    useful = [s for s in sentences if not thinking.match(s)]
    if not useful:
        useful = sentences[-4:] if sentences else []
    blob = " ".join(useful[-5:])
    if not blob:
        return clip_spoken(cleaned, width)
    if len(blob) <= width:
        return blob
    tail = blob[-width:]
    for punct in (". ", "! ", "? "):
        i = tail.find(punct)
        if 0 <= i < 100:
            return tail[i + 2 :].strip()
    return tail.lstrip()


def spoken_restore_confirm(title: str, recap_text: str) -> str:
    spoken = _short_title(title, width=80)
    last_you = ""
    last_grok = ""
    for line in (recap_text or "").splitlines():
        if line.startswith("Last you:"):
            last_you = line[len("Last you:") :].strip()
        elif line.startswith("Last grok:"):
            last_grok = line[len("Last grok:") :].strip()
    bits = [f"Restored the conversation about {spoken}."]
    if last_you:
        bits.append("You last said: " + clip_spoken(last_you, 220))
    if last_grok:
        bits.append("I last said: " + clip_spoken(last_grok, 320))
    return " ".join(bits)


def render_for_restore(messages: list[Message]) -> str:
    """Render history for Voice, collapsing nested restore dumps so it does not re-restore."""
    lines = []
    noted_prior_restore = False
    for msg in messages:
        if msg.kind == "compact":
            lines.append(msg.text)
            continue
        if (
            msg.name == "restore_conversation"
            or msg.kind == "restore"
            or (msg.role == "tool" and "Prior conversation loaded" in (msg.text or ""))
        ):
            if not noted_prior_restore:
                lines.append(
                    "Assistant [restore]: An earlier chat was already loaded into this session."
                )
                noted_prior_restore = True
            continue
        if msg.role == "user" and (
            is_restore_request(msg.text or "") or _is_low_content_user(msg.text or "")
        ):
            continue
        if msg.role == "assistant" and _RESTORE_ASSISTANT.match((msg.text or "").strip()):
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


def _term_in(hay: str, word: str) -> bool:
    """Match query terms; short words need a real word boundary so 'pi' ≠ '.py'."""
    if not word or not hay:
        return False
    if len(word) <= 3:
        return bool(re.search(rf"\b{re.escape(word)}\b", hay))
    stem = word.rstrip("s")
    if word in hay:
        return True
    return bool(stem) and len(stem) >= 4 and stem in hay


def score_conversation(meta: ConversationMeta, query: str, messages: list[Message] | None = None) -> float:
    if not query.strip():
        return 0.0
    q = query.lower().strip()
    stop = {
        "the", "a", "an", "on", "about", "we", "had", "conversation",
        "yesterday", "today", "restore", "load", "please", "can", "you",
        "for", "or", "check",
    }
    words = [w for w in re.split(r"\W+", q) if w and w not in stop]
    msgs = messages or []
    unique = unique_messages(msgs)
    restore_shell = is_restore_request(meta.title) or (
        bool(msgs)
        and msgs[0].role == "user"
        and is_restore_request(msgs[0].text or "")
    )
    user_hay = " ".join(m.text.lower() for m in unique if m.role == "user")
    unique_hay = " ".join(m.text.lower() for m in unique)
    title_hay = f"{meta.title} {meta.id}".lower()
    score = 0.0
    if q and q in user_hay:
        score += 6
    elif q and q in unique_hay:
        score += 3
    elif q and q in title_hay and not restore_shell:
        score += 4
    for word in words:
        if _term_in(user_hay, word):
            score += 2
        elif _term_in(unique_hay, word):
            score += 1
        elif _term_in(title_hay, word):
            score += 0.5 if restore_shell else 1
    unique_turns = sum(1 for m in unique if m.role in {"user", "assistant"})
    score += min(8.0, unique_turns / 12.0)
    title_hit = bool(words) and all(_term_in(title_hay, w) for w in words)
    if restore_shell and title_hit and unique_turns >= 12:
        score += 5
    elif restore_shell and unique_turns <= 3:
        score -= 6
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
        self._prior_ids: set[str] = set()
        self._restored_ids: list[str] = []

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
        self._prior_ids = set()
        self._restored_ids = []
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
        if role == "user" and not self._titled_from_user and not text.startswith("/"):
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
            if new_id != self.meta.id:
                self._prior_ids.add(self.meta.id)
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

    def note_restored(self, conv_id: str) -> None:
        conv_id = (conv_id or "").strip()
        if conv_id and conv_id not in self._restored_ids:
            self._restored_ids.append(conv_id)

    def exclude_ids(self) -> list[str]:
        ids: list[str] = []
        if self.meta:
            ids.append(self.meta.id)
            ids.extend(self._prior_ids)
        ids.extend(self._restored_ids)
        return ids

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
        drop = {self.meta.id} | self._prior_ids
        items = [m for m in load_index(self.root) if m.id not in drop]
        items.append(self.meta)
        items.sort(key=lambda m: m.updated, reverse=True)
        save_index(items, self.root)


def _exclude_set(exclude_ids: list[str] | None) -> set[str]:
    return {i for i in (exclude_ids or []) if i}


def list_conversations(
    root: Path | None = None,
    limit: int = 15,
    exclude_ids: list[str] | None = None,
) -> list[ConversationMeta]:
    skip = _exclude_set(exclude_ids)
    items = [m for m in load_index(root) if m.id not in skip]
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


def find_conversation(
    query: str,
    root: Path | None = None,
    exclude_ids: list[str] | None = None,
) -> ConversationMeta | None:
    root = root or CONVERSATIONS
    skip = _exclude_set(exclude_ids)
    items = [m for m in load_index(root) if m.id not in skip]
    if not items:
        return None
    q = query.strip()
    if not q:
        items.sort(key=lambda m: m.updated, reverse=True)
        return items[0]
    if q.isdigit():
        ranked = list_conversations(root, exclude_ids=exclude_ids)
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


def load_restore_text(
    query: str,
    root: Path | None = None,
    exclude_ids: list[str] | None = None,
) -> tuple[str, ConversationMeta | None]:
    root = root or CONVERSATIONS
    listing = format_conversation_list(list_conversations(root, exclude_ids=exclude_ids))
    if not query.strip():
        return listing, None
    meta = find_conversation(query, root, exclude_ids=exclude_ids)
    if not meta:
        return f"No match for {query!r}. Recent:\n{listing}", None
    messages = read_messages(root / meta.path)
    body = render_for_restore(messages)
    recap = format_restore_recap(messages)
    header = (
        f"Restored conversation {meta.title!r} from {meta.started[:10]} (id {meta.id}).\n"
        f"{recap}"
    )
    return f"{header}\n\n{body}", meta
