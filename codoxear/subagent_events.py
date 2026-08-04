from __future__ import annotations

"""Backend-neutral transcript events for subagent lifecycle/progress notices."""

from typing import Any


_SUPPORTED_BACKENDS = frozenset({"pi", "codex", "cc"})


def emit_subagent_event(
    backend: str,
    *,
    event_id: str,
    text: str,
    ts: float | None = None,
) -> dict[str, Any]:
    """Build one normalized assistant narration event for a backend subagent.

    Each backend supplies its native durable event identity.  The normalized
    prefix lets tail, history, search, and live delivery deduplicate the same
    event without a frontend/backend-specific rendering branch.
    """
    if backend not in _SUPPORTED_BACKENDS:
        raise ValueError(f"unsupported subagent backend: {backend}")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("subagent event_id must be a non-empty string")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("subagent text must be non-empty")
    event: dict[str, Any] = {
        "role": "assistant",
        "text": text,
        "message_class": "narration",
        "message_id": f"{backend}-subagent:{event_id}",
    }
    if ts is not None:
        event["ts"] = float(ts)
    return event


def is_subagent_event_id(value: Any) -> bool:
    return isinstance(value, str) and any(value.startswith(f"{backend}-subagent:") for backend in _SUPPORTED_BACKENDS)
