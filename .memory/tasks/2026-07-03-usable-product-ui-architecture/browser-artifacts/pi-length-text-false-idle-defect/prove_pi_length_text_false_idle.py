#!/usr/bin/env python3
"""Deterministic artifact-only proof for Pi stopReason=length text false-idle.

This script imports the current checkout code at HEAD, builds synthetic Pi JSONL
rows in a temporary directory, and records how the existing projection, idle,
turn-state, broker reducer, and readiness mechanisms classify those rows.
It intentionally does not modify source or tests and does not touch runtime dirs.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from codoxear.agent_backend import PI_BACKEND
from codoxear.broker_log_watcher import _apply_log_objects_to_state
from codoxear.broker_turn_state import State, _should_clear_busy_state
from codoxear.pi_log import pi_current_turn_state_before
from codoxear.pi_message import pi_assistant_is_final_turn_end
from codoxear.rollout_idle import _compute_idle_from_log
from codoxear.session_runtime import BrokerRuntimeState, resolve_runtime_status, session_runtime_readiness

ARTIFACT_DIR = Path(__file__).resolve().parent
OUTPUT_JSON = ARTIFACT_DIR / "proof-output.json"
SUMMARY_TXT = ARTIFACT_DIR / "proof-summary.txt"
REPORT_MD = ARTIFACT_DIR / "VERIFICATION-REPORT.md"


def _session_row() -> dict[str, Any]:
    return {"type": "session", "id": "synthetic-pi-length-proof", "timestamp": "2026-07-06T00:00:00Z"}


def _user_row() -> dict[str, Any]:
    return {
        "type": "message",
        "timestamp": "2026-07-06T00:00:01Z",
        "message": {"role": "user", "content": [{"type": "text", "text": "please continue"}]},
    }


def _assistant_length_text_row() -> dict[str, Any]:
    return {
        "type": "message",
        "timestamp": "2026-07-06T00:00:02Z",
        "message": {
            "role": "assistant",
            "stopReason": "length",
            "content": [{"type": "text", "text": "partial before compaction"}],
        },
    }


def _compaction_rows() -> list[dict[str, Any]]:
    return [
        {"type": "custom_message", "timestamp": "2026-07-06T00:00:03Z", "message": "Compacting conversation"},
        {"type": "custom_message", "timestamp": "2026-07-06T00:00:04Z", "message": "Continuing after compaction"},
    ]


def _assistant_tooluse_continuation_row() -> dict[str, Any]:
    return {
        "type": "message",
        "timestamp": "2026-07-06T00:00:05Z",
        "message": {
            "role": "assistant",
            "stopReason": "toolUse",
            "content": [
                {"type": "text", "text": "resuming after compaction and calling a tool"},
                {"type": "toolCall", "id": "toolu_1", "name": "read", "input": {"path": "README.md"}},
            ],
        },
    }


def _assistant_stop_text_row() -> dict[str, Any]:
    return {
        "type": "message",
        "timestamp": "2026-07-06T00:00:02Z",
        "message": {
            "role": "assistant",
            "stopReason": "stop",
            "content": [{"type": "text", "text": "complete final answer"}],
        },
    }


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _projection(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        event = PI_BACKEND.chat_event_from_log_row(row)
        out.append(
            {
                "row_index": index,
                "row_type": row.get("type"),
                "result_type": type(event).__name__ if event is not None else None,
                "event": event,
                "role": event.get("role") if isinstance(event, dict) else None,
                "text": event.get("text") if isinstance(event, dict) else None,
                "message_class": event.get("message_class") if isinstance(event, dict) else None,
            }
        )
    return out


def _broker_state_after(rows: list[dict[str, Any]]) -> dict[str, Any]:
    st = State(
        codex_pid=0,
        pty_master_fd=-1,
        cwd=str(Path.cwd()),
        start_ts=0.0,
        codex_home=ARTIFACT_DIR,
        sessions_dir=ARTIFACT_DIR,
    )
    clock = {"t": 0.0}

    def now() -> float:
        clock["t"] += 1.0
        return clock["t"]

    _apply_log_objects_to_state(st, rows, now=now)
    clear_now = clock["t"] + 10.0
    return {
        "busy": st.busy,
        "turn_open": st.turn_open,
        "turn_has_completion_candidate": st.turn_has_completion_candidate,
        "pending_calls": _json_safe(st.pending_calls),
        "last_turn_activity_ts": st.last_turn_activity_ts,
        "should_clear_busy_state_at_t_plus_10_quiet0": _should_clear_busy_state(
            st,
            clear_now,
            busy_quiet_seconds=0.0,
            busy_interrupt_grace_seconds=0.0,
        ),
    }


def _runtime_projection(*, log_idle: bool | None, broker_state: dict[str, Any], log_exists: bool = True) -> dict[str, Any]:
    runtime = resolve_runtime_status(
        broker=BrokerRuntimeState(
            busy=bool(broker_state["busy"]),
            queue_len=0,
            interrupted_idle=False,
        ),
        log_exists=log_exists,
        log_idle=log_idle,
        send_boundary_unresolved=False,
    )
    readiness = session_runtime_readiness(runtime)
    return {
        "runtime_status": _json_safe(runtime),
        "readiness": _json_safe(readiness),
        "api_sessions_would_project_busy": runtime.busy,
        "api_sessions_would_project_sendable_remote_ready": readiness.direct_send,
    }


def _analyze_case(name: str, rows: list[dict[str, Any]], log_path: Path) -> dict[str, Any]:
    _write_jsonl(log_path, rows)
    size = log_path.stat().st_size
    pending, turn_idle = pi_current_turn_state_before(log_path, size)
    idle = _compute_idle_from_log(log_path)
    broker = _broker_state_after(rows)
    return {
        "name": name,
        "rows": rows,
        "log_size": size,
        "pi_assistant_is_final_turn_end_by_row": [
            {"row_index": i, "value": pi_assistant_is_final_turn_end(row)} for i, row in enumerate(rows)
        ],
        "projection": _projection(rows),
        "compute_idle_from_log": idle,
        "pi_current_turn_state_before_eof": {"pending": _json_safe(pending), "idle": turn_idle},
        "broker_turn_state_after_rows": broker,
        "runtime_readiness_from_log_and_broker": _runtime_projection(log_idle=idle, broker_state=broker),
    }


def _repo_state() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=Path.cwd(), text=True).strip()

    return {
        "head": run(["git", "rev-parse", "--short", "HEAD"]),
        "status_short_before_script_output_write": run(["git", "status", "--short"]),
    }


def _verdict(results: dict[str, Any]) -> str:
    prefix = results["cases"]["length_text_prefix"]
    cont = results["cases"]["length_text_then_tooluse_continuation"]
    control = results["cases"]["stop_text_control"]
    defect = (
        prefix["pi_assistant_is_final_turn_end_by_row"][2]["value"] is True
        and prefix["projection"][2]["message_class"] == "final_response"
        and prefix["compute_idle_from_log"] is True
        and prefix["pi_current_turn_state_before_eof"]["idle"] is True
        and prefix["runtime_readiness_from_log_and_broker"]["api_sessions_would_project_busy"] is False
        and prefix["runtime_readiness_from_log_and_broker"]["api_sessions_would_project_sendable_remote_ready"] is True
        and cont["compute_idle_from_log"] is False
        and cont["runtime_readiness_from_log_and_broker"]["api_sessions_would_project_busy"] is True
        and control["compute_idle_from_log"] is True
    )
    return "DEFECT" if defect else "PASS"


def _write_summary(results: dict[str, Any]) -> None:
    prefix = results["cases"]["length_text_prefix"]
    cont = results["cases"]["length_text_then_tooluse_continuation"]
    control = results["cases"]["stop_text_control"]
    lines = [
        f"Verdict: {results['verdict']}",
        f"HEAD: {results['repo']['head']}",
        "",
        "Length+visible-text prefix:",
        f"- pi_assistant_is_final_turn_end(length row): {prefix['pi_assistant_is_final_turn_end_by_row'][2]['value']}",
        f"- PiBackend projection class: {prefix['projection'][2]['message_class']}",
        f"- _compute_idle_from_log: {prefix['compute_idle_from_log']}",
        f"- pi_current_turn_state_before EOF idle: {prefix['pi_current_turn_state_before_eof']['idle']}",
        f"- broker reducer busy/turn_open: {prefix['broker_turn_state_after_rows']['busy']} / {prefix['broker_turn_state_after_rows']['turn_open']}",
        f"- runtime busy/sendable: {prefix['runtime_readiness_from_log_and_broker']['api_sessions_would_project_busy']} / {prefix['runtime_readiness_from_log_and_broker']['api_sessions_would_project_sendable_remote_ready']}",
        "",
        "Continuation after compaction:",
        f"- last assistant projection class: {cont['projection'][-1]['message_class']}",
        f"- _compute_idle_from_log: {cont['compute_idle_from_log']}",
        f"- current-turn pending/idle: {cont['pi_current_turn_state_before_eof']['pending']} / {cont['pi_current_turn_state_before_eof']['idle']}",
        f"- broker reducer busy/turn_open/pending: {cont['broker_turn_state_after_rows']['busy']} / {cont['broker_turn_state_after_rows']['turn_open']} / {cont['broker_turn_state_after_rows']['pending_calls']}",
        "",
        "Control stop+visible-text:",
        f"- final turn end: {control['pi_assistant_is_final_turn_end_by_row'][2]['value']}",
        f"- projection class: {control['projection'][2]['message_class']}",
        f"- _compute_idle_from_log: {control['compute_idle_from_log']}",
    ]
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_report(results: dict[str, Any]) -> None:
    prefix = results["cases"]["length_text_prefix"]
    cont = results["cases"]["length_text_then_tooluse_continuation"]
    control = results["cases"]["stop_text_control"]
    text = f"""# Pi `stopReason:\"length\"` visible-text false-idle proof

Verdict: **{results['verdict']}**

## Command

```bash
python3 .memory/tasks/2026-07-03-usable-product-ui-architecture/browser-artifacts/pi-length-text-false-idle-defect/prove_pi_length_text_false_idle.py
```

## Observation

### Synthetic prefix: user row then assistant `stopReason:\"length\"` with visible text and no tool call

- `pi_assistant_is_final_turn_end()` on the length row: `{prefix['pi_assistant_is_final_turn_end_by_row'][2]['value']}`
- `PiBackend.chat_event_from_log_row()` projects the length row as role `{prefix['projection'][2]['role']}` with `message_class={prefix['projection'][2]['message_class']!r}` and text `{prefix['projection'][2]['text']!r}`.
- `_compute_idle_from_log()` returns `{prefix['compute_idle_from_log']}`.
- `pi_current_turn_state_before(..., EOF)` returns pending `{prefix['pi_current_turn_state_before_eof']['pending']}` and idle `{prefix['pi_current_turn_state_before_eof']['idle']}`.
- Broker reducer after session/user/length rows: `busy={prefix['broker_turn_state_after_rows']['busy']}`, `turn_open={prefix['broker_turn_state_after_rows']['turn_open']}`, `turn_has_completion_candidate={prefix['broker_turn_state_after_rows']['turn_has_completion_candidate']}`.
- Runtime/readiness projection from that log+broker state: `busy={prefix['runtime_readiness_from_log_and_broker']['api_sessions_would_project_busy']}`, `direct_send={prefix['runtime_readiness_from_log_and_broker']['api_sessions_would_project_sendable_remote_ready']}`.

### Synthetic continuation: same prefix plus compaction rows and assistant `toolUse` continuation

- Last assistant continuation event: role `{cont['projection'][-1]['role']}`, `message_class={cont['projection'][-1]['message_class']!r}`, text `{cont['projection'][-1]['text']!r}`.
- `_compute_idle_from_log()` returns `{cont['compute_idle_from_log']}`.
- `pi_current_turn_state_before(..., EOF)` returns pending `{cont['pi_current_turn_state_before_eof']['pending']}` and idle `{cont['pi_current_turn_state_before_eof']['idle']}`.
- Broker reducer after all rows: `busy={cont['broker_turn_state_after_rows']['busy']}`, `turn_open={cont['broker_turn_state_after_rows']['turn_open']}`, pending `{cont['broker_turn_state_after_rows']['pending_calls']}`.
- Runtime/readiness projection after continuation: `busy={cont['runtime_readiness_from_log_and_broker']['api_sessions_would_project_busy']}`, `direct_send={cont['runtime_readiness_from_log_and_broker']['api_sessions_would_project_sendable_remote_ready']}`.

### Control: assistant `stopReason:\"stop\"` with visible text

- `pi_assistant_is_final_turn_end()` on the stop row: `{control['pi_assistant_is_final_turn_end_by_row'][2]['value']}`.
- Projection class: `{control['projection'][2]['message_class']!r}`.
- `_compute_idle_from_log()` returns `{control['compute_idle_from_log']}`.

## Interpretation

Current HEAD treats a Pi assistant row with visible text and `stopReason:\"length\"` as a final answer. The same row closes broker turn state, makes `_compute_idle_from_log()` idle, makes `pi_current_turn_state_before()` idle, and makes runtime readiness sendable. The synthetic continuation then reopens/busies the same turn when a later `toolUse` row appears. That is a transient false-idle window at the compaction/continuation boundary, violating the binary busy/idle invariant: the browser can consider the session idle/sendable while Pi is still continuing the turn.

The `stopReason:\"stop\"` control remains final/idle under the same mechanisms, so the proof isolates `length` rather than visible text itself.

## Files written

- `prove_pi_length_text_false_idle.py`
- `proof-output.json`
- `proof-summary.txt`
- `VERIFICATION-REPORT.md`

No source files or tests are modified by this artifact proof.
"""
    REPORT_MD.write_text(text, encoding="utf-8")


def main() -> None:
    prefix_rows = [_session_row(), _user_row(), _assistant_length_text_row()]
    continuation_rows = prefix_rows + _compaction_rows() + [_assistant_tooluse_continuation_row()]
    control_rows = [_session_row(), _user_row(), _assistant_stop_text_row()]
    with tempfile.TemporaryDirectory(prefix="pi-length-proof-", dir=ARTIFACT_DIR) as tmp_raw:
        tmp = Path(tmp_raw)
        results: dict[str, Any] = {
            "repo": _repo_state(),
            "cases": {
                "length_text_prefix": _analyze_case("length_text_prefix", prefix_rows, tmp / "length-prefix.jsonl"),
                "length_text_then_tooluse_continuation": _analyze_case(
                    "length_text_then_tooluse_continuation", continuation_rows, tmp / "length-continuation.jsonl"
                ),
                "stop_text_control": _analyze_case("stop_text_control", control_rows, tmp / "stop-control.jsonl"),
            },
        }
    results["verdict"] = _verdict(results)
    OUTPUT_JSON.write_text(json.dumps(_json_safe(results), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_summary(results)
    _write_report(results)
    print(f"verdict={results['verdict']}")
    print(f"wrote={OUTPUT_JSON}")
    print(f"wrote={SUMMARY_TXT}")
    print(f"wrote={REPORT_MD}")


if __name__ == "__main__":
    main()
