from __future__ import annotations

import json
from typing import Any


def _jsonl_line(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True) + "\n"


def pi_rpc_request_payloads() -> dict[str, dict[str, Any]]:
    return {
        "prompt": {
            "id": "cmd-prompt-1",
            "type": "prompt",
            "message": "Summarize the current repository state.",
        },
        "abort": {
            "id": "cmd-abort-1",
            "type": "abort",
        },
        "get_state": {
            "id": "cmd-state-1",
            "type": "get_state",
        },
    }


def pi_rpc_response_lines() -> dict[str, str]:
    return {
        "prompt": _jsonl_line(
            {
                "type": "response",
                "id": "cmd-prompt-1",
                "command": "prompt",
                "success": True,
                "data": {"queued": False},
            }
        ),
        "abort": _jsonl_line(
            {
                "type": "response",
                "id": "cmd-abort-1",
                "command": "abort",
                "success": True,
                "data": {},
            }
        ),
        "get_state": _jsonl_line(
            {
                "type": "response",
                "id": "cmd-state-1",
                "command": "get_state",
                "success": True,
                "data": {
                    "isStreaming": False,
                    "pendingMessageCount": 0,
                    "sessionId": "pi-session-001",
                    "sessionFile": "/tmp/pi-session-001.jsonl",
                    "messageCount": 2,
                },
            }
        ),
    }


def pi_stream_events() -> list[dict[str, Any]]:
    return [
        {
            "type": "turn.started",
            "turn_id": "turn-001",
            "role": "user",
            "text": "Summarize the current repository state.",
        },
        {
            "type": "message.delta",
            "turn_id": "turn-001",
            "role": "assistant",
            "delta": "Codoxear serves a browser UI for Codex-style sessions.",
        },
        {
            "type": "tool.started",
            "turn_id": "turn-001",
            "tool_name": "read",
            "call_id": "tool-001",
        },
        {
            "type": "turn.completed",
            "turn_id": "turn-001",
            "role": "assistant",
            "text": "Codoxear serves a browser UI for Codex-style sessions.",
        },
    ]


def pi_ui_request_event() -> dict[str, Any]:
    return {
        "type": "extension_ui_request",
        "id": "ui-req-1",
        "method": "select",
        "title": "Pick a location",
        "options": ["Details", "Sidebar"],
        "timeout": 10000,
    }


def pi_ui_response_payload() -> dict[str, Any]:
    return {
        "type": "extension_ui_response",
        "id": "ui-req-1",
        "value": "Details",
    }


def pi_persisted_session_entries() -> list[dict[str, Any]]:
    return [
        {
            "type": "message",
            "id": "msg-user-001",
            "payload": {
                "type": "message",
                "id": "msg-user-001",
                "turn_id": "turn-001",
                "role": "user",
                "source_event": "turn.started",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Summarize the current repository state.",
                    }
                ],
            },
        },
        {
            "type": "message",
            "id": "msg-assistant-001",
            "payload": {
                "type": "message",
                "id": "msg-assistant-001",
                "turn_id": "turn-001",
                "role": "assistant",
                "source_event": "turn.completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Codoxear serves a browser UI for Codex-style sessions.",
                    }
                ],
            },
        },
    ]


def pi_persisted_session_file() -> list[dict[str, Any]]:
    return [
        {
            "type": "session",
            "session_id": "pi-session-001",
            "cwd": "/workspace/codoxear",
            "version": 1,
        },
        *pi_persisted_session_entries(),
    ]


def pi_runtime_session_file() -> list[dict[str, Any]]:
    return [
        {
            "type": "session",
            "version": 3,
            "id": "pi-session-001",
            "timestamp": "2026-03-28T14:38:08.286Z",
            "cwd": "/workspace/codoxear",
        },
        {
            "type": "message",
            "id": "msg-user-001",
            "timestamp": "2026-03-28T14:38:27.330Z",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "Summarize the current repository state."}],
                "timestamp": 1774708707280,
            },
        },
        {
            "type": "message",
            "id": "msg-assistant-001",
            "timestamp": "2026-03-28T14:38:36.099Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Codoxear serves a browser UI for Codex-style sessions."}],
                "timestamp": 1774708716099,
            },
        },
    ]


def build_session_history(entries: list[dict[str, Any]]) -> list[dict[str, str]]:
    history: list[dict[str, str]] = []
    for entry in entries:
        payload = entry.get("payload") if entry.get("type") == "message" else entry
        if not isinstance(payload, dict):
            continue

        role = payload.get("role")
        content = payload.get("content")
        text = None
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "output_text" and isinstance(item.get("text"), str) and item["text"]:
                    text = item["text"]
                    break

        if role in {"user", "assistant"} and isinstance(text, str) and text:
            history.append({"role": role, "text": text})
    return history
