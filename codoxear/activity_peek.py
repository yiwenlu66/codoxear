from __future__ import annotations

import json
from typing import Any, Iterable, TypedDict

from .cc_log import cc_assistant_content_parts, cc_user_tool_result_ids
from .pi_message import pi_assistant_content_parts, pi_log_row_is_transcript_excluded, pi_tool_result_id
from .rollout_events import _codex_event_text


class ActivityPeek(TypedDict):
    # label is None only in the flip-only signal {label: None, running: False}:
    # "the call you are showing finished; keep the text, drop the marker".
    label: str | None
    running: bool


def _label(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= 120 else text[:119].rstrip() + "…"


def _first_line(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return next((line.strip() for line in value.splitlines() if line.strip()), "")


def _scalar(value: Any) -> str:
    if isinstance(value, str):
        return _first_line(value)
    if isinstance(value, (int, float, bool)):
        return str(value)
    return ""


def format_tool_label(name: str, args: Any) -> str:
    """Format a compact hint, never a serialization of a tool's full input."""
    name = name.strip().lower()
    tool = name.rsplit(".", 1)[-1]
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except (ValueError, RecursionError):
            args = None
    if not isinstance(args, dict):
        return _label(name)

    def first(*keys: str) -> str:
        return next((text for key in keys if (text := _scalar(args.get(key)))), "")

    detail = ""
    if tool in {"bash", "shell", "exec", "exec_command", "local_shell_call", "shell_command"}:
        name = "bash"
        command = args.get("command", args.get("cmd"))
        if isinstance(command, list) and all(isinstance(part, str) for part in command):
            command = " ".join(command)
        detail = _first_line(command)
    elif tool in {"read", "write", "edit", "str_replace", "apply_patch", "read_file", "write_file"}:
        detail = first("path", "file_path", "file", "filename")
    elif tool in {"grep", "search", "glob"}:
        detail = first("pattern", "query", "search_pattern")
    elif "web" in tool or tool in {"fetch", "browse"}:
        detail = first("url", "query", "search_query")
    elif "subagent" in tool or tool in {"task", "agent"}:
        name = "subagent"
        detail = first("task", "description", "prompt")
    else:
        detail = next((text for value in args.values() if (text := _scalar(value))), "")
    return _label(f"{name}: {detail}" if detail else name)


def _row_updates(obj: dict[str, Any]) -> Iterable[tuple[str, str | None, str]]:
    """Yield ordered (kind, call id, label) updates using backend-owned helpers."""
    for part in pi_assistant_content_parts(obj) + cc_assistant_content_parts(obj):
        kind = part.get("type")
        if kind in ("toolCall", "tool_use"):
            name = part.get("name")
            if isinstance(name, str) and name.strip():
                call_id = part.get("id")
                yield "call", call_id if isinstance(call_id, str) else None, format_tool_label(
                    name, part.get("arguments") if kind == "toolCall" else part.get("input")
                )
        elif kind == "thinking":
            text = _first_line(part.get("thinking"))
            yield "thinking", None, _label(f"thinking: {text}" if text else "thinking")

    pi_result = pi_tool_result_id(obj)
    for call_id in ([pi_result] if pi_result is not None else []) + cc_user_tool_result_ids(obj):
        yield "result", call_id, ""

    payload = obj.get("payload")
    if not isinstance(payload, dict):
        return
    kind = payload.get("type")
    call_id = payload.get("call_id")
    call_id = call_id if isinstance(call_id, str) else None
    if obj.get("type") == "response_item":
        if kind in ("function_call", "custom_tool_call"):
            name = payload.get("name")
            if isinstance(name, str) and name.strip():
                yield "call", call_id, format_tool_label(name, payload.get("arguments", payload.get("input")))
        elif kind == "local_shell_call":
            yield "call", call_id, format_tool_label("local_shell_call", payload.get("action"))
        elif kind in ("function_call_output", "custom_tool_call_output"):
            yield "result", call_id, ""
        elif kind == "reasoning":
            summary = payload.get("summary")
            text = next(
                (text for part in summary if isinstance(part, dict) and (text := _first_line(part.get("text")))),
                "",
            ) if isinstance(summary, list) else _first_line(summary)
            yield "thinking", None, _label(f"thinking: {text}" if text else "thinking")
    elif obj.get("type") == "event_msg":
        if kind == "agent_reasoning":
            text = _first_line(payload.get("text")) or _first_line(_codex_event_text(payload))
            yield "thinking", None, _label(f"thinking: {text}" if text else "thinking")
        elif kind == "exec_command_begin":
            yield "call", call_id, format_tool_label("bash", payload)
        elif kind == "exec_command_end":
            yield "result", call_id, ""


def scan_window_activity(objs: Iterable[Any]) -> tuple[bool, ActivityPeek | None]:
    """Project ephemeral activity from exactly this window, without carried state.

    A tool result never hides its call: fast tools execute inside one read
    window, so clearing on the result would erase the call before it is ever
    displayed. Instead a matching result flips the call to not-running, and a
    result whose call is outside the window emits a flip-only signal
    (``label=None``) telling the client to drop the running marker from the
    call it is already showing. A result for an older parallel call changes
    nothing.
    """
    relevant = False
    activity: ActivityPeek | None = None
    latest_call: str | None = None
    for obj in objs:
        if not isinstance(obj, dict):
            continue
        # The peek is transcript chrome: harness/intercom traffic that every
        # transcript projection filters out must not surface here either.
        if pi_log_row_is_transcript_excluded(obj):
            continue
        for kind, call_id, label in _row_updates(obj):
            relevant = True
            if kind == "call":
                latest_call = call_id
                activity = {"label": label, "running": True}
            elif kind == "thinking":
                activity = {"label": label, "running": False}
            elif activity is not None and activity.get("running") and (call_id is None or call_id == latest_call):
                activity = {"label": activity["label"], "running": False}
            elif activity is None:
                activity = {"label": None, "running": False}
    return relevant, activity
