from __future__ import annotations

from typing import Any

from .cc_log import cc_assistant_thinking_count
from .cc_log import cc_assistant_tool_use_count
from .cc_log import cc_is_turn_end
from .cc_log import cc_message_role
from .cc_log import cc_system_api_error_is_terminal
from .cc_log import cc_user_text
from .pi_log import pi_assistant_thinking_count
from .pi_log import pi_assistant_tool_use_count
from .pi_log import pi_assistant_is_aborted_turn
from .pi_log import pi_user_text
from .rollout_chat_events import _single_chat_event
from .rollout_events import _codex_error_affects_turn_status


def _extract_chat_events(
    objs: list[dict[str, Any]],
    *,
    initial_cc_pending_tool_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, bool], dict[str, Any]]:
    events: list[dict[str, Any]] = []
    total_thinking = 0
    total_tools = 0
    total_system = 0
    turn_start = False
    turn_end = False
    turn_aborted = False
    tool_names: set[str] = set()
    last_tool: str | None = None
    cc_pending_tool_ids: set[str] = set(initial_cc_pending_tool_ids or set())
    seen_pi_subagent_ids: set[str] = set()
    for obj in objs:
        typ = obj.get("type")
        event = _single_chat_event(obj, cc_pending_tool_ids=cc_pending_tool_ids)
        if event is not None:
            message_id = event.get("message_id")
            if isinstance(message_id, str) and message_id.startswith("pi-subagent:"):
                if message_id in seen_pi_subagent_ids:
                    event = None
                else:
                    seen_pi_subagent_ids.add(message_id)
            if event is not None:
                events.append(event)

        if typ == "user":
            user_text = cc_user_text(obj)
            if isinstance(user_text, str) and user_text:
                turn_start = True
                continue
            if cc_message_role(obj) == "toolResult":
                total_tools += 1
                continue

        if typ == "assistant":
            tool_count = cc_assistant_tool_use_count(obj)
            thinking_count = cc_assistant_thinking_count(obj)
            if thinking_count > 0:
                total_thinking += thinking_count
            if tool_count > 0:
                # `_single_chat_event` already updates `cc_pending_tool_ids`; do
                # not call `cc_assistant_pending_tool_use_ids()` again here
                # because id-less Claude Code tool uses receive generated
                # placeholders and a second call would create phantom pending ids.
                total_tools += tool_count
                message = obj.get("message")
                content = message.get("content") if isinstance(message, dict) else None
                for part in content if isinstance(content, list) else []:
                    if isinstance(part, dict) and part.get("type") == "tool_use":
                        name = part.get("name")
                        if isinstance(name, str) and name:
                            tool_names.add(name)
                            last_tool = name
            if event is not None and event.get("message_class") == "final_response":
                turn_end = True
            if event is not None and event.get("message_class") == "error":
                turn_end = True
            continue

        if typ == "system":
            if event is not None and event.get("message_class") == "error":
                turn_end = True
                cc_pending_tool_ids.clear()
                continue
            if cc_is_turn_end(obj):
                if not cc_pending_tool_ids:
                    turn_end = True
                continue
            if cc_system_api_error_is_terminal(obj):
                turn_end = True
                cc_pending_tool_ids.clear()
                continue

        if typ == "message":
            user_text = pi_user_text(obj)
            if isinstance(user_text, str) and user_text:
                turn_start = True
                continue

            tool_count = pi_assistant_tool_use_count(obj)
            thinking_count = pi_assistant_thinking_count(obj)
            if thinking_count > 0:
                total_thinking += thinking_count
            if tool_count > 0:
                total_tools += tool_count
                tool_names.add("pi_tool")
                last_tool = "pi_tool"
            if pi_assistant_is_aborted_turn(obj):
                turn_aborted = True
            elif event is not None and event.get("message_class") == "final_response":
                turn_end = True
            continue

        if typ == "event_msg":
            p = obj.get("payload")
            if not isinstance(p, dict):
                raise ValueError("invalid event_msg payload")
            pt = p.get("type")
            if pt == "user_message":
                msg = p.get("message")
                if isinstance(msg, str):
                    turn_start = True
                continue
            if pt == "agent_reasoning":
                total_thinking += 1
                continue
            if pt == "turn_aborted":
                turn_aborted = True
                continue
            if pt in ("error", "stream_error", "warning"):
                if pt == "error" and _codex_error_affects_turn_status(p):
                    turn_end = True
                continue
            if pt in ("task_complete", "turn_complete"):
                turn_end = True
                continue
            if pt == "token_count":
                continue

        if typ == "response_item":
            p = obj.get("payload")
            if not isinstance(p, dict):
                raise ValueError("invalid response_item payload")
            pt = p.get("type")
            if pt == "message":
                role = p.get("role")
                if role in ("developer", "system"):
                    total_system += 1
                continue

            if pt == "reasoning":
                total_thinking += 1
                continue
            if pt == "function_call":
                nm = p.get("name")
                if isinstance(nm, str) and nm:
                    tool_names.add(nm)
                    last_tool = nm
                total_tools += 1
                continue
            if pt in (
                "function_call_output",
                "custom_tool_call",
                "custom_tool_call_output",
                "web_search_call",
                "local_shell_call",
            ):
                total_tools += 1
                continue

    return (
        events,
        {"thinking": total_thinking, "tool": total_tools, "system": total_system},
        {"turn_start": turn_start, "turn_end": turn_end, "turn_aborted": turn_aborted},
        {"tool_names": sorted(tool_names), "last_tool": last_tool},
    )
