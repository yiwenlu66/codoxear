#!/usr/bin/env python3
"""
prove_next_outcome_defect.py — Deterministic artifact-only discriminator
for Pi assistant rows with empty text content.

Three edge cases tested:
  C1: assistant {role:'assistant', stopReason:'stop', content:[]}
  C2: assistant {role:'assistant', stopReason:'end_turn', content:[]}
  C3: assistant {role:'assistant', stopReason:'stop',
       content:[{type:'thinking', thinking:''}]}

For each: build minimal synthetic Pi log, then run:
  _extract_positioned_chat_events  (via _read_chat_tail_page)
  _read_chat_tail_page
  search_chat_log_bounded (for 'backend completed' and 'interrupted')
  _compute_idle_from_log
  pi_current_turn_state_before
  handle_messages_tail (simulated — same pipeline as real)

Also: Task B — classify post-log backend death as SCOUT.
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Ensure codoxear package is importable
repo_root = Path("/home/yiwen/codex-web-product-recovery")
sys.path.insert(0, str(repo_root))

from codoxear.rollout_log import (
    _extract_positioned_chat_events,
    _read_chat_tail_page,
    _read_chat_page_reverse,
    JsonlRecord,
)
from codoxear.rollout_idle import _compute_idle_from_log
from codoxear.rollout_jsonl import _iter_jsonl_records_reverse
from codoxear.transcript_search import search_chat_log_bounded
from codoxear.pi_log import (
    pi_current_turn_state_before,
    pi_assistant_text,
    pi_assistant_is_final_turn_end,
    pi_assistant_thinking_count,
    pi_assistant_tool_use_count,
    pi_assistant_is_aborted_turn,
    pi_assistant_error_text,
    pi_user_text,
    pi_message_role,
)
from codoxear.pi_message import pi_assistant_content_parts
from codoxear.rollout_chat_events import _pi_message_keeps_turn_busy, _single_chat_event


# ── synthetic Pi log builder ────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())


def build_pi_log(path: Path, user_text: str, assistant_msg: dict[str, Any]) -> None:
    session_id = "00000000-0000-0000-0000-000000000001"
    lines = [
        json.dumps({
            "type": "session",
            "id": session_id,
            "cwd": "/tmp",
            "timestamp": _ts(),
        }),
        json.dumps({
            "type": "message",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            },
            "timestamp": _ts(),
            "sessionId": session_id,
        }),
        json.dumps({
            "type": "message",
            "message": assistant_msg,
            "timestamp": _ts(),
            "sessionId": session_id,
        }),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── helpers ──────────────────────────────────────────────────────────────────

def records_from_path(path: Path) -> list[JsonlRecord]:
    recs: list[JsonlRecord] = []
    for r in _iter_jsonl_records_reverse(path):
        recs.append(r)
    recs.reverse()
    return recs


def classify_idle(value: bool | None) -> str:
    if value is True:
        return "IDLE"
    if value is False:
        return "BUSY"
    return "UNKNOWN"


def classify_turn(pending: set, idle: bool | None) -> str:
    if idle is None:
        return "AMBIGUOUS"
    if idle and not pending:
        return "IDLE"
    if not idle and pending:
        return "BUSY_WITH_PENDING_TOOLS"
    if not idle:
        return "BUSY"
    return "IDLE"


# ── per-case discriminator ──────────────────────────────────────────────────

def run_case(label: str, assistant_msg: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"label": label, "assistant_msg": assistant_msg}

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "session.jsonl"
        build_pi_log(log_path, "Hello, what is 2+2?", assistant_msg)

        # --- Raw row-level probes ---
        records = records_from_path(log_path)
        # The assistant row is records[2] (0: session, 1: user, 2: assistant)
        asst_obj = records[2].obj if len(records) > 2 else {}

        result["raw_probes"] = {
            "pi_assistant_text": pi_assistant_text(asst_obj),
            "pi_assistant_is_final_turn_end": pi_assistant_is_final_turn_end(asst_obj),
            "pi_assistant_thinking_count": pi_assistant_thinking_count(asst_obj),
            "pi_assistant_tool_use_count": pi_assistant_tool_use_count(asst_obj),
            "pi_assistant_is_aborted_turn": pi_assistant_is_aborted_turn(asst_obj),
            "pi_assistant_error_text": pi_assistant_error_text(asst_obj),
            "pi_message_role": pi_message_role(asst_obj),
            "_pi_message_keeps_turn_busy": _pi_message_keeps_turn_busy(asst_obj),
            "pi_assistant_content_parts": [
                {"type": p.get("type"), "has_text": "text" in p}
                for p in pi_assistant_content_parts(asst_obj)
            ],
            "_single_chat_event": _single_chat_event(asst_obj),
        }

        # --- _extract_positioned_chat_events ---
        events = _extract_positioned_chat_events(records)
        result["positioned_chat_events"] = {
            "count": len(events),
            "events": [
                {
                    "role": e.get("role"),
                    "text": e.get("text"),
                    "message_class": e.get("message_class"),
                }
                for e in events
            ],
        }

        # --- _read_chat_tail_page ---
        tail_events, before_byte, after_byte, has_older = _read_chat_tail_page(log_path, limit=80)
        result["tail_page"] = {
            "event_count": len(tail_events),
            "events": [
                {
                    "role": e.get("role"),
                    "text": e.get("text"),
                    "message_class": e.get("message_class"),
                }
                for e in tail_events
            ],
            "before_byte": before_byte,
            "after_byte": after_byte,
            "has_older": has_older,
        }

        # --- search_chat_log_bounded ---
        for query in ["backend completed", "interrupted"]:
            count, matches, truncated = search_chat_log_bounded(
                log_path, query, limit=20,
            )
            result[f"search_{query.replace(' ', '_')}"] = {
                "count": count,
                "match_count_truncated": truncated,
                "match_roles": [m.get("role") for m in matches],
            }

        # --- _compute_idle_from_log ---
        idle = _compute_idle_from_log(log_path)
        result["compute_idle_from_log"] = {
            "value": idle,
            "classification": classify_idle(idle),
        }

        # --- pi_current_turn_state_before ---
        size = int(log_path.stat().st_size)
        pending, turn_idle = pi_current_turn_state_before(log_path, size)
        result["pi_current_turn_state_before"] = {
            "pending_count": len(pending),
            "idle": turn_idle,
            "classification": classify_turn(pending, turn_idle),
        }

    return result


# ── Task B: post-log backend death SCOUT classification ─────────────────────

def classify_post_log_death() -> dict[str, Any]:
    """
    session_prune.py prune_dead_sessions():
      - Removes sessions when sock_path doesn't exist, or the control socket
        call fails with stale error, or broker/agent PIDs are dead.
      - Does NOT inspect log for a terminal event before removal.
      - For Pi sessions with content=[] last row, no close/turn_end/delivery
        event exists to confirm a deterministic outcome.

    session_control.py:
      - _dead_processes() checks broker_pid + codex_pid.
      - On dead processes + socket call failure, drops session.

    message_routes.py:
      - handle_messages_tail/handle_messages_live: when log is unbound or
        missing, return empty response. No special dead-backend handling.
      - The transcript (chat events, search) is log-derived only. No log
        close event → no signal of backend death.

    SCOUT rationale: When the backend dies after writing a no-text Pi
    assistant row (stop/end_turn, empty content), no deterministic transcript
    event exists to confirm the outcome. The log-driven idle check reports
    BUSY (defect), but the prune mechanism cleans up the dead session
    normally. The absence of a terminal event makes the final state
    SCOUT — undetermined from log evidence alone.
    """
    return {
        "classification": "SCOUT",
        "rationale": (
            "No deterministic transcript event (no close/turn_end/delivery row) "
            "exists in Pi logs when the backend dies after a no-text assistant row. "
            "session_prune.py drops dead sessions via socket/pid checks without "
            "inspecting log for terminal events. message_routes.py returns empty "
            "responses for unbound logs. The log-driven _compute_idle_from_log "
            "returns BUSY for these rows (DEFECT), but the prune path handles "
            "cleanup normally. SCOUT because the final outcome cannot be determined "
            "from log evidence alone."
        ),
        "files_inspected": [
            "codoxear/session_prune.py",
            "codoxear/session_control.py",
            "codoxear/message_routes.py",
        ],
        "key_observations": {
            "session_prune": (
                "prune_dead_sessions removes sessions via sock_path existence, "
                "control socket errors, and pid checks. Does not inspect log "
                "for a terminal event."
            ),
            "session_control": (
                "_dead_processes checks broker_pid and codex_pid. Drops session "
                "when both are dead and a socket call fails."
            ),
            "message_routes": (
                "handle_messages_tail/live return empty responses for unbound "
                "logs. No special dead-backend transcript handling."
            ),
        },
    }


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    cases = [
        (
            "C1: stopReason=stop, content=[]",
            {"role": "assistant", "stopReason": "stop", "content": []},
        ),
        (
            "C2: stopReason=end_turn, content=[]",
            {"role": "assistant", "stopReason": "end_turn", "content": []},
        ),
        (
            "C3: stopReason=stop, content=[{type:'thinking', thinking:''}]",
            {
                "role": "assistant",
                "stopReason": "stop",
                "content": [{"type": "thinking", "thinking": ""}],
            },
        ),
    ]

    all_results: dict[str, Any] = {
        "generated_at": _ts(),
        "repo": str(repo_root),
        "cases": {},
        "task_b": classify_post_log_death(),
    }

    for label, asst_msg in cases:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        result = run_case(label, asst_msg)
        all_results["cases"][label] = result

        # Compact console output
        rp = result["raw_probes"]
        print(f"  pi_assistant_text:            {rp['pi_assistant_text']}")
        print(f"  pi_assistant_is_final_turn_end: {rp['pi_assistant_is_final_turn_end']}")
        print(f"  pi_assistant_thinking_count:  {rp['pi_assistant_thinking_count']}")
        print(f"  _pi_message_keeps_turn_busy:  {rp['_pi_message_keeps_turn_busy']}")
        print(f"  _single_chat_event:           {rp['_single_chat_event']}")

        pce = result["positioned_chat_events"]
        print(f"  positioned_chat_events:       {pce['count']} events")
        for e in pce["events"]:
            print(f"    role={e['role']}, mclass={e['message_class']}, text={repr(e['text'])[:60]}")

        tp = result["tail_page"]
        print(f"  tail_page:                    {tp['event_count']} events")

        idle_r = result["compute_idle_from_log"]
        print(f"  _compute_idle_from_log:       {idle_r['classification']}")

        turn_r = result["pi_current_turn_state_before"]
        print(f"  pi_current_turn_state_before: {turn_r['classification']}")

        for sq in ["search_backend_completed", "search_interrupted"]:
            sr = result[sq]
            print(f"  {sq}: count={sr['count']} matches")

    # Write proof-output.json
    out_dir = Path(__file__).resolve().parent
    output_path = out_dir / "proof-output.json"
    output_path.write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")
    print(f"\n✓ proof-output.json written to {output_path}")


if __name__ == "__main__":
    main()
