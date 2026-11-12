"""Summary-free compaction boundaries and in-window metadata enrichment."""
from __future__ import annotations

from typing import Any

from .rollout_events import _event_ts, _text_message_id


def nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def codex_compaction_source(obj: dict[str, Any]) -> str | None:
    typ = obj.get("type")
    payload = obj.get("payload")
    if typ == "compacted" or (typ == "response_item" and isinstance(payload, dict) and payload.get("type") == "compacted"):
        return "checkpoint"
    if typ == "event_msg" and isinstance(payload, dict) and payload.get("type") == "context_compacted":
        return "notification"
    return None


def compaction_event(obj: dict[str, Any]) -> dict[str, Any] | None:
    typ = obj.get("type")
    backend = None
    metadata: dict[str, Any] = {}
    if typ == "compaction":
        backend = "pi"
        before = nonnegative_int(obj.get("tokensBefore"))
        if before is not None:
            metadata["tokens_before"] = before
        # fromHook records an extension override, not the trigger. Pi's manual
        # /compact is intercepted by the TUI and neither it nor reason persists.
    elif typ == "system" and obj.get("subtype") == "compact_boundary":
        backend = "cc"
        compact = obj.get("compactMetadata")
        if isinstance(compact, dict):
            before = nonnegative_int(compact.get("preTokens"))
            if before is not None:
                metadata["tokens_before"] = before
            if compact.get("trigger") in ("auto", "manual"):
                metadata["trigger"] = compact["trigger"]
    elif codex_compaction_source(obj):
        backend = "codex"
    if backend is None:
        return None
    event: dict[str, Any] = {"role": "system", "kind": "compaction", **metadata}
    ts = _event_ts(obj)
    if ts is not None:
        event["ts"] = ts
    entry_id = obj.get("id", obj.get("uuid", ""))
    if ts is not None or entry_id:
        event["message_id"] = "compaction:" + _text_message_id(message_class="compaction", text=f"{backend}:{entry_id}", ts=ts)
    return event


def enrich_compaction_events(rows: list[tuple[dict[str, Any], dict[str, Any] | None]]) -> list[dict[str, Any]]:
    """Use only records already in the projection window; never read a file.

    Pi's compaction usage is the summarizer's usage, not the new context size.
    The first subsequent assistant usage.totalTokens supplies the latter. A
    partial window cannot prove a kept count, so omit it unless the named entry
    is present (including non-visible/tool entries, whose visible count is zero).
    Codex persists a checkpoint and may also persist its completion notification;
    complementary adjacent sources represent one boundary, not two.
    """
    out: list[dict[str, Any]] = []
    visible = 0
    entry_counts: dict[str, int] = {}
    pending_pi: dict[str, Any] | None = None
    previous_codex: str | None = None
    for obj, event in rows:
        entry_id = obj.get("id")
        if isinstance(entry_id, str):
            entry_counts[entry_id] = visible
        source = codex_compaction_source(obj)
        if source:
            if previous_codex == "checkpoint" and source == "notification":
                previous_codex = None
                continue
            previous_codex = source
        elif event is not None:
            previous_codex = None
        if obj.get("type") == "compaction" and event is not None:
            pending_pi = event
            first = obj.get("firstKeptEntryId")
            if isinstance(first, str) and first in entry_counts:
                event["kept_messages"] = visible - entry_counts[first]
        elif obj.get("type") == "message" and pending_pi is not None:
            message = obj.get("message")
            if isinstance(message, dict) and message.get("role") == "assistant":
                usage = message.get("usage")
                after = nonnegative_int(usage.get("totalTokens")) if isinstance(usage, dict) else None
                if after is not None:
                    pending_pi["tokens_after"] = after
                    pending_pi = None
        if event is not None:
            out.append(event)
            if event.get("role") in ("user", "assistant"):
                visible += 1
    return out
