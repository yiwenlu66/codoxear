#!/usr/bin/env python3
"""Artifact-only proof for post-log-bind backend death.

This driver starts no Codoxear server, broker, sessiond, tmux, or backend
process. It exercises the production lifecycle/message components in-process:

1. Discover/register a fake web-owned session whose sidecar points at an
   existing Pi log with a user message and no terminal assistant/error row.
2. Read the same transcript through the real message tail/search route handlers.
3. Simulate the production broker exit cleanup for a post-bind death by removing
   the socket and sidecar, then run the real prune coordinator with dead pids.
4. Read sessions/messages/launch-ledger state again.

Expected defect signal: before prune the user-only transcript is reachable;
after prune the session is gone, message routes return unknown session, and the
launch ledger has no visible failed/recovery row because all current failure
paths are gated on log_path is None.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from codoxear.launch_ledger import latest_launch_attempt
from codoxear.launch_ledger import launch_attempt_row
from codoxear.launch_ledger import launch_attempt_transcript_for_session_id
from codoxear.launch_ledger import record_launch_attempt
from codoxear.launch_ledger import submitted_user_messages
from codoxear.launch_ledger import launch_failure_tail
from codoxear.message_routes import MessageRouteDeps
from codoxear.message_routes import handle_messages_search
from codoxear.message_routes import handle_messages_tail
from codoxear.session_discovery import DiscoveryDeps
from codoxear.session_discovery import discover_sessions
from codoxear.session_discovery_registry import SessionDiscoveryRegistryCoordinator
from codoxear.session_prune import SessionPruneCoordinator
from codoxear.util import read_launch_attempts

ARTIFACT_DIR = Path(__file__).resolve().parent
WORK_DIR = ARTIFACT_DIR / "workdir"
RAW_DIR = ARTIFACT_DIR / "raw-api"
LAUNCH_ATTEMPTS = WORK_DIR / "launch_attempts.jsonl"
SENTINEL = "POST_LOG_BOUND_DEATH_SENTINEL"
SESSION_ID = "post-log-death"
THREAD_ID = "post-log-bound-thread"
LAUNCH_ID = "launch-post-log-bound"
BROKER_PID = 111_111
AGENT_PID = 222_222
START_TS = 1_783_200_000.0


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_sidecar(sock: Path, *, root: Path, log_path: Path) -> None:
    meta = {
        "session_id": THREAD_ID,
        "agent_backend": "pi",
        "owner": "web",
        "broker_pid": BROKER_PID,
        "codex_pid": AGENT_PID,
        "cwd": str(root),
        "start_ts": START_TS,
        "updated_ts": START_TS + 10,
        "log_path": str(log_path),
        "launch_id": LAUNCH_ID,
        "spawn_nonce": "spawn-post-log-bound",
        "model_provider": "proof-provider",
        "preferred_auth_method": "apikey",
        "model": "proof-model",
        "reasoning_effort": "low",
        "control_protocol_version": 2,
        "control_capabilities": {"sync_send": True, "key_write_errors": True},
    }
    sock.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


def write_user_only_pi_log(path: Path, *, cwd: Path) -> None:
    rows = [
        {"type": "session", "id": THREAD_ID, "cwd": str(cwd)},
        {
            "type": "message",
            "createdAt": "2026-07-06T00:00:00.000Z",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": SENTINEL}],
            },
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def discovery_deps(*, live_pids: set[int], sock_error: BaseException | None = None) -> DiscoveryDeps:
    def sock_call(_sock: Path, _req: dict[str, Any], _timeout_s: float) -> dict[str, Any]:
        if sock_error is not None:
            raise sock_error
        return {"busy": True, "queue_len": 0, "interrupted_idle": False, "token": None}

    return DiscoveryDeps(
        pid_alive=lambda pid: int(pid) in live_pids,
        proc_find_open_rollout_log=lambda _proc_root, _root_pid, _agent_backend, _cwd, _ignored_paths: None,
        read_session_meta_or_none=lambda _log_path, _agent_backend, _context: None,
        coerce_main_thread_log=lambda thread_id, log_path: (thread_id, log_path),
        session_transport=lambda meta: (meta.get("transport"), meta.get("tmux_session"), meta.get("tmux_window")),
        session_run_settings=lambda meta, _log_path, _agent_backend: (
            meta.get("model_provider"),
            meta.get("preferred_auth_method"),
            meta.get("model"),
            meta.get("reasoning_effort"),
        ),
        sock_call=sock_call,
        broker_busy_queue_from_state=lambda state: (bool(state.get("busy")), int(state.get("queue_len", 0))),
        broker_interrupted_idle_from_state=lambda state: bool(state.get("interrupted_idle")),
        sock_error_definitely_stale=lambda exc: isinstance(exc, FileNotFoundError),
        token_update_finder=lambda _log_path: None,
    )


class CaptureHandler:
    def __init__(self) -> None:
        self.responses: list[dict[str, Any]] = []
        self.unauthorized = False

    def _unauthorized(self) -> None:
        self.unauthorized = True
        self.responses.append({"status": 401, "payload": {"error": "unauthorized"}})


class FakeManager:
    def __init__(self, sessions: dict[str, Any], *, launch_attempts_path: Path) -> None:
        self.sessions = sessions
        self.launch_attempts_path = launch_attempts_path

    def refresh_session_meta(self, _session_id: str) -> None:
        return None

    def get_session(self, session_id: str) -> Any | None:
        return self.sessions.get(session_id)

    def _attach_notification_texts(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return events

    def mark_log_delta(self, _session_id: str, *, objs: list[dict[str, Any]], new_off: int) -> None:
        return None


def make_message_deps(*, launch_attempts_path: Path) -> MessageRouteDeps:
    def json_response(handler: CaptureHandler, status: int, payload: dict[str, Any]) -> None:
        handler.responses.append({"status": status, "payload": payload})

    return MessageRouteDeps(
        require_auth=lambda _handler: True,
        json_response=json_response,
        launch_attempt_transcript_for_session_id=lambda session_id: launch_attempt_transcript_for_session_id(
            session_id,
            path=launch_attempts_path,
            default_agent_backend="pi",
            unattended_default_idle_minutes=30,
            unattended_default_max_injections=10,
        ),
        transcript_export_max_bytes=8 * 1024 * 1024,
        transcript_search_max_line_bytes=128 * 1024,
        decode_message_cursor=lambda cursor, *, kind, session: int(str(cursor).split(":")[-1]),
        encode_message_cursor=lambda *, kind, session, pos: f"{kind}:{pos}",
        record_metric=lambda _name, _value: None,
        message_runtime_snapshot=lambda session_id, s, **_kwargs: ({"busy": bool(s.busy)}, bool(s.busy), int(s.queue_len), s.token),
    )


def route_tail(manager: FakeManager, deps: MessageRouteDeps, session_id: str) -> dict[str, Any]:
    h = CaptureHandler()
    handle_messages_tail(h, session_id=session_id, query="limit=80", manager=manager, deps=deps)
    return h.responses[-1]


def route_search(manager: FakeManager, deps: MessageRouteDeps, session_id: str) -> dict[str, Any]:
    h = CaptureHandler()
    handle_messages_search(
        h,
        session_id=session_id,
        query=f"q={SENTINEL}&limit=20&count_max=100&text_max=200",
        manager=manager,
        deps=deps,
    )
    return h.responses[-1]


def session_rows_api_equivalent(sessions: dict[str, Any], *, launch_attempts_path: Path) -> dict[str, Any]:
    active_rows = []
    for sid, s in sorted(sessions.items()):
        active_rows.append(
            {
                "session_id": sid,
                "thread_id": s.thread_id,
                "agent_backend": s.agent_backend,
                "owned": s.owned,
                "cwd": s.cwd,
                "log_path": str(s.log_path) if s.log_path else None,
                "log_exists": bool(s.log_path and s.log_path.exists()),
                "busy": bool(s.busy),
                "queue_len": int(s.queue_len),
                "launch_id": s.launch_id,
            }
        )
    launch_records = list(read_launch_attempts(path=launch_attempts_path, max_records=100, max_age_s=10 * 365 * 24 * 3600))
    launch_rows = [
        row
        for row in (
            launch_attempt_row(
                rec,
                default_agent_backend="pi",
                unattended_default_idle_minutes=30,
                unattended_default_max_injections=10,
            )
            for rec in launch_records
        )
        if row is not None
    ]
    return {
        "active_rows": active_rows,
        "launch_attempt_rows": launch_rows,
        "combined_session_ids": [row["session_id"] for row in active_rows] + [row["session_id"] for row in launch_rows],
        "launch_records": launch_records,
    }


def source_evidence() -> list[dict[str, Any]]:
    checks = [
        ("codoxear/broker.py", "stage=\"agent_exit_before_log_bind\""),
        ("codoxear/broker.py", "and (st2.log_path is None or not st2.log_path.exists())"),
        ("codoxear/session_prune.py", "session.owned and session.log_path is None"),
        ("codoxear/session_discovery.py", "broker_exit_before_log_bind"),
        ("codoxear/session_discovery.py", "if (log_path is None) and (not deps.pid_alive(codex_pid))"),
        ("codoxear/session_control.py", "self._drop_dead_session(session_id, sock, clear_deleted_state=True)"),
        ("codoxear/message_routes.py", "launch_attempt_transcript_for_session_id(session_id)"),
    ]
    out: list[dict[str, Any]] = []
    for rel, needle in checks:
        path = REPO_ROOT / rel
        lines = path.read_text(encoding="utf-8").splitlines()
        found = None
        for idx, line in enumerate(lines, start=1):
            if needle in line:
                found = {"file": rel, "line": idx, "needle": needle, "text": line.strip()}
                break
        out.append(found or {"file": rel, "needle": needle, "missing": True})
    return out


def main() -> int:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    if RAW_DIR.exists():
        shutil.rmtree(RAW_DIR)
    WORK_DIR.mkdir(parents=True)
    RAW_DIR.mkdir(parents=True)

    sock_dir = WORK_DIR / "socks"
    sock_dir.mkdir()
    log_path = WORK_DIR / "pi-sessions" / "post-log-bound.jsonl"
    write_user_only_pi_log(log_path, cwd=WORK_DIR)
    sock = sock_dir / f"{SESSION_ID}.sock"
    sock.touch(mode=0o600)
    write_sidecar(sock, root=WORK_DIR, log_path=log_path)

    # Mimic the durable launch ledger state of a web launch that already bound a log.
    record_launch_attempt(
        {
            "launch_id": LAUNCH_ID,
            "state": "log_bound",
            "stage": "log_bound",
            "agent_backend": "pi",
            "cwd": str(WORK_DIR),
            "created_ts": START_TS,
            "updated_ts": START_TS + 10,
            "broker_pid": BROKER_PID,
            "agent_pid": AGENT_PID,
            "log_path": str(log_path),
            "spawn_nonce": "spawn-post-log-bound",
        },
        path=LAUNCH_ATTEMPTS,
        stderr=sys.stderr,
    )

    sessions: dict[str, Any] = {}
    lock = threading.RLock()
    recorded_failures: list[dict[str, Any]] = []
    deleted_state: list[str] = []
    unhidden: list[str] = []
    recent_cwds: list[dict[str, Any]] = []

    initial_discovery = discover_sessions(
        sock_dir,
        proc_root=WORK_DIR / "proc",
        hidden_sessions=set(),
        deps=discovery_deps(live_pids={BROKER_PID, AGENT_PID}),
    )
    registry = SessionDiscoveryRegistryCoordinator(
        lock=lock,
        sessions=lambda: sessions,
        pending_attachment_ids=lambda: set(),
        commit_unknown_sends=lambda: {},
        reset_log_caches=lambda session, off: setattr(session, "meta_log_off", off),
        record_launch_attempt=lambda rec: recorded_failures.append(record_launch_attempt(rec, path=LAUNCH_ATTEMPTS, stderr=sys.stderr)),
        prune_stale_socket_without_metadata=lambda sid, _sock: deleted_state.append(sid),
        unhide_session=lambda sid: unhidden.append(sid),
        unlink_quiet=lambda p: p.unlink(missing_ok=True),
        remember_recent_cwd=lambda cwd, ts=None: recent_cwds.append({"cwd": cwd, "ts": ts}) or True,
        save_recent_cwds=lambda: None,
        stderr=sys.stderr,
    )
    registry.apply_result(initial_discovery)

    manager = FakeManager(sessions, launch_attempts_path=LAUNCH_ATTEMPTS)
    deps = make_message_deps(launch_attempts_path=LAUNCH_ATTEMPTS)

    before_sessions = session_rows_api_equivalent(sessions, launch_attempts_path=LAUNCH_ATTEMPTS)
    before_tail = route_tail(manager, deps, SESSION_ID)
    before_search = route_search(manager, deps, SESSION_ID)

    # Production post-bind broker exit cleanup unlinks socket and sidecar. No broad
    # process kill is involved: these are proof-owned files only.
    sock.unlink(missing_ok=True)
    sock.with_suffix(".json").unlink(missing_ok=True)

    prune = SessionPruneCoordinator(
        lock=lock,
        sessions=lambda: sessions,
        sock_call=lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("dead fake socket")),
        broker_busy_queue_from_state=lambda state: (bool(state.get("busy")), int(state.get("queue_len", 0))),
        broker_interrupted_idle_from_state=lambda state: bool(state.get("interrupted_idle")),
        sock_error_definitely_stale=lambda exc: isinstance(exc, FileNotFoundError),
        pid_alive=lambda _pid: False,
        latest_launch_attempt=lambda launch_id: latest_launch_attempt(launch_id, path=LAUNCH_ATTEMPTS),
        submitted_user_messages=submitted_user_messages,
        launch_failure_tail=launch_failure_tail,
        which_tmux=lambda _cmd: None,
        tmux_pane_snapshot=lambda *_args, **_kwargs: {},
        clean_optional_text=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
        record_launch_attempt=lambda rec: recorded_failures.append(record_launch_attempt(rec, path=LAUNCH_ATTEMPTS, stderr=sys.stderr)),
        clear_deleted_session_state=lambda sid: deleted_state.append(sid),
        unlink_quiet=lambda p: p.unlink(missing_ok=True),
        stderr=sys.stderr,
    )
    prune.prune_dead_sessions()

    after_sessions = session_rows_api_equivalent(sessions, launch_attempts_path=LAUNCH_ATTEMPTS)
    after_tail = route_tail(manager, deps, SESSION_ID)
    after_search = route_search(manager, deps, SESSION_ID)
    ledger_after = list(read_launch_attempts(path=LAUNCH_ATTEMPTS, max_records=100, max_age_s=10 * 365 * 24 * 3600))

    observations = {
        "before_session_listed": SESSION_ID in before_sessions["combined_session_ids"],
        "before_tail_status": before_tail["status"],
        "before_tail_roles": [event.get("role") for event in before_tail["payload"].get("events", [])],
        "before_tail_texts": [event.get("text") for event in before_tail["payload"].get("events", [])],
        "before_search_status": before_search["status"],
        "before_search_match_count": before_search["payload"].get("match_count"),
        "after_session_listed": SESSION_ID in after_sessions["combined_session_ids"],
        "after_tail_status": after_tail["status"],
        "after_tail_error": after_tail["payload"].get("error"),
        "after_search_status": after_search["status"],
        "after_search_error": after_search["payload"].get("error"),
        "failure_records_written_by_prune": recorded_failures,
        "visible_launch_attempt_rows_after": after_sessions["launch_attempt_rows"],
        "launch_ledger_states_after": [rec.get("state") for rec in ledger_after],
        "orphan_log_still_exists": log_path.exists(),
        "orphan_log_path": str(log_path),
        "deleted_state_calls": deleted_state,
    }
    defect = (
        observations["before_session_listed"]
        and observations["before_tail_status"] == 200
        and SENTINEL in observations["before_tail_texts"]
        and observations["before_search_status"] == 200
        and observations["before_search_match_count"] == 1
        and not observations["after_session_listed"]
        and observations["after_tail_status"] == 404
        and observations["after_search_status"] == 404
        and not observations["failure_records_written_by_prune"]
        and not observations["visible_launch_attempt_rows_after"]
        and observations["orphan_log_still_exists"]
    )

    output = {
        "verdict": "DEFECT" if defect else "SCOUT",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "head": os.popen("git rev-parse HEAD").read().strip(),
        "isolation": "in-process only; no Codoxear server/broker/session/backend/tmux process started; proof-owned temp files under artifact dir",
        "scenario": {
            "session_id": SESSION_ID,
            "launch_id": LAUNCH_ID,
            "backend": "pi",
            "log_shape": "session header + user message only; no terminal assistant/error/no-response/interruption row",
            "simulated_death": "socket and sidecar removed, pids reported dead, real SessionPruneCoordinator invoked",
        },
        "initial_discovery": {
            "registrations": [asdict(reg) for reg in initial_discovery.registrations],
            "stale_actions": [asdict(action) for action in initial_discovery.stale_actions],
        },
        "observations": observations,
        "source_evidence": source_evidence(),
        "raw_files": {
            "before_sessions": str(RAW_DIR / "before-sessions.json"),
            "before_tail": str(RAW_DIR / "before-tail.json"),
            "before_search": str(RAW_DIR / "before-search.json"),
            "after_sessions": str(RAW_DIR / "after-sessions.json"),
            "after_tail": str(RAW_DIR / "after-tail.json"),
            "after_search": str(RAW_DIR / "after-search.json"),
            "launch_ledger_after": str(RAW_DIR / "launch-ledger-after.json"),
        },
    }

    write_json(RAW_DIR / "before-sessions.json", before_sessions)
    write_json(RAW_DIR / "before-tail.json", before_tail)
    write_json(RAW_DIR / "before-search.json", before_search)
    write_json(RAW_DIR / "after-sessions.json", after_sessions)
    write_json(RAW_DIR / "after-tail.json", after_tail)
    write_json(RAW_DIR / "after-search.json", after_search)
    write_json(RAW_DIR / "launch-ledger-after.json", ledger_after)
    write_json(ARTIFACT_DIR / "proof-output.json", output)

    summary = (
        f"Verdict: {output['verdict']}\n"
        f"Before: listed={observations['before_session_listed']}, tail={observations['before_tail_status']} "
        f"roles={observations['before_tail_roles']} search_matches={observations['before_search_match_count']}\n"
        f"After: listed={observations['after_session_listed']}, tail={observations['after_tail_status']} "
        f"search={observations['after_search_status']} failure_records={len(recorded_failures)} "
        f"visible_launch_rows={len(observations['visible_launch_attempt_rows_after'])}\n"
        f"Mechanism: current broker/discovery/prune failure persistence is gated on log_path is None. "
        f"A post-log-bound death leaves the user-only log orphaned at {log_path} with no API-visible session, "
        f"no launch recovery row, and no transcript terminal outcome.\n"
    )
    (ARTIFACT_DIR / "proof-summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    return 0 if defect else 2


if __name__ == "__main__":
    raise SystemExit(main())
