"""One end-to-end behavioral summary for the session's four critical fixes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
import time

import codoxear.broker as broker_module
from codoxear.broker_turn_state import State
from codoxear.message_cursor import decode_message_cursor, encode_message_cursor
from codoxear.message_routes import MessageRouteDeps, handle_messages_tail
from codoxear.session_listing import build_active_session_rows_snapshot, build_public_session_row
from codoxear.session_model import Session
from codoxear.session_runtime import session_run_settings_from_meta
from codoxear.session_store import SessionStore, SessionStorePaths
from codoxear.static_routes import STATIC_DIR, static_route_asset


class _MessagesHandler:
    def _unauthorized(self) -> None:
        raise AssertionError("the summary route request must be authenticated")


class _MessagesManager:
    def __init__(self, session: Session) -> None:
        self.session = session

    def refresh_session_meta(self, _session_id: str) -> None:
        return None

    def get_session(self, _session_id: str) -> Session:
        return self.session

    def mark_log_delta(self, *_args: object, **_kwargs: object) -> None:
        return None

    def _attach_notification_texts(self, events: list[dict[str, object]]) -> list[dict[str, object]]:
        return events


def _listing_store(tmp_path: Path) -> SessionStore:
    return SessionStore(
        paths=SessionStorePaths(
            aliases=tmp_path / "aliases.json",
            sidebar_meta=tmp_path / "sidebar.json",
            hidden_sessions=tmp_path / "hidden.json",
            files=tmp_path / "files.json",
            queues=tmp_path / "queues.json",
            pending_attachments=tmp_path / "pending.json",
            commit_unknown_sends=tmp_path / "unknown.json",
            recent_cwds=tmp_path / "recent.json",
            unattended=tmp_path / "unattended.json",
        ),
        file_history_max=5,
        recent_cwd_max=5,
        unattended_default_idle_minutes=5,
        unattended_default_max_injections=10,
        clean_alias=lambda value: value if isinstance(value, str) else "",
        clean_priority_offset=lambda value: float(value or 0.0),
        clean_snooze_until=lambda value: float(value) if value not in (None, "", 0) else None,
        clean_dependency_session_id=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
        clean_recent_cwd=lambda value: value.strip() if isinstance(value, str) and value.strip() else None,
        clean_commit_unknown_send_record=lambda value: value if isinstance(value, dict) else None,
    )


def _write_50mb_log(path: Path) -> None:
    """Make a 50 MiB logical JSONL with a scan-stopping 12 MiB tail."""
    target_size = 50 * 1024 * 1024
    tail_size = 12 * 1024 * 1024
    latest = json.dumps(
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "latest"}],
                "phase": "final_answer",
            },
            "ts": 1.0,
        }
    ).encode("utf-8") + b"\n"
    filler = b'{"type":"debug","payload":"' + b"x" * (64 * 1024) + b'"}\n'
    with path.open("wb") as stream:
        stream.seek(target_size - tail_size)
        while stream.tell() + len(filler) + len(latest) <= target_size:
            stream.write(filler)
        remaining = target_size - stream.tell() - len(latest)
        if remaining:
            stream.write(b"x" * (remaining - 1) + b"\n")
        stream.write(latest)
    assert path.stat().st_size == target_size


def test_session_critical_summary(tmp_path: Path, monkeypatch) -> None:
    """Prove marker binding, cached message I/O, live Pi effort, and PDF loading."""
    # dab6445c: a live Pi marker must win over the obsolete --session JSONL.
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    old_log = sessions_dir / "old.jsonl"
    new_log = sessions_dir / "new.jsonl"
    old_log.write_text('{"type":"session","id":"old"}\n', encoding="utf-8")
    new_log.write_text('{"type":"session","id":"new"}\n', encoding="utf-8")
    marker = tmp_path / "pi-active.json"
    marker.write_text(json.dumps({"version": 1, "sessionFile": str(new_log)}), encoding="utf-8")
    Path(f"{marker}.caps").write_text(
        json.dumps(
            {
                "bridgeVersion": 2,
                "pid": os.getpid(),
                "features": ["thinking"],
                "commands": [{"name": "effort"}],
                "reasoning_effort": "high",
            }
        ),
        encoding="utf-8",
    )

    broker = broker_module.Broker.__new__(broker_module.Broker)
    broker._stop = threading.Event()
    broker._lock = threading.Lock()
    broker.state = State(
        codex_pid=os.getpid(),
        pty_master_fd=-1,
        cwd=str(tmp_path),
        start_ts=time.time(),
        codex_home=tmp_path,
        sessions_dir=sessions_dir,
        declared_log_path=old_log,
        reasoning_effort="low",
    )
    broker.sessions_dir = sessions_dir
    broker.pi_active_session_marker_path = marker
    broker._refresh_pi_active_session_observability = lambda **_kwargs: None
    broker._write_meta = lambda: None
    bound_paths: list[Path] = []

    def register_live_log(*, log_path: Path) -> None:
        bound_paths.append(log_path)
        broker._stop.set()

    broker._maybe_register_or_switch_rollout = register_live_log
    monkeypatch.setattr(broker_module, "AGENT_BACKEND", "pi")
    monkeypatch.setattr(broker_module.time, "sleep", lambda _seconds: None)
    watcher = threading.Thread(target=broker._discover_log_watcher, daemon=True)
    watcher.start()
    watcher.join(timeout=1)
    assert not watcher.is_alive()
    assert bound_paths == [new_log]

    # 333dfe8d: the bridge's live caps override launch intent and stale replay,
    # then the actual listing projection exposes that effective value.
    assert broker.state is not None
    assert broker.state.live_run_settings == {"reasoning_effort": "high"}
    resolved = session_run_settings_from_meta(
        meta={
            "broker_pid": os.getpid(),
            "reasoning_effort": "low",
            "live_run_settings": broker.state.live_run_settings,
        },
        log_path=new_log,
        agent_backend="pi",
        clean_optional_text=lambda value: value if isinstance(value, str) else None,
        normalize_requested_preferred_auth_method=lambda value: value if isinstance(value, str) else None,
        display_reasoning_effort=lambda value: value if isinstance(value, str) else None,
        display_pi_reasoning_effort=lambda value: value if isinstance(value, str) else None,
        normalize_requested_cc_reasoning_effort=lambda value: value if isinstance(value, str) else None,
        read_run_settings_from_log=lambda *_args, **_kwargs: (None, None, "max"),
    )
    assert resolved == (None, None, None, "high")
    pi_session = Session(
        session_id="pi-summary",
        thread_id="pi-thread",
        broker_pid=os.getpid(),
        codex_pid=os.getpid(),
        agent_backend="pi",
        owned=False,
        start_ts=1.0,
        cwd=str(tmp_path),
        log_path=new_log,
        sock_path=tmp_path / "pi.sock",
        reasoning_effort=resolved[3],
        pi_thinking_command=broker.state.pi_thinking_command,
    )
    listing = build_active_session_rows_snapshot(
        sessions=[pi_session],
        queues={},
        unattended={},
        aliases={},
        store=_listing_store(tmp_path),
        now_ts=2.0,
        unattended_default_idle_minutes=5,
        unattended_default_max_injections=10,
        clean_unattended_cooldown_minutes=lambda value: int(value),
        clean_unattended_remaining_injections=lambda value, *, allow_zero=False: int(value),
        provider_choice_for_settings=lambda **_kwargs: "",
        resolve_session_cwd=Path,
        priority_half_life_seconds=60.0,
        priority_bucket_seconds=1.0,
        subagent_runs={},
    )
    public_pi_row = build_public_session_row(listing.rows[0], git_branch=None, busy=False)
    assert public_pi_row["reasoning_effort"] == "high"

    # 685782e8: the message-tail route makes one bounded scan, then serves its
    # unchanged revision from cache within the interactive I/O budget.
    large_log = tmp_path / "50mb.jsonl"
    _write_50mb_log(large_log)
    messages_session = Session(
        session_id="messages-summary",
        thread_id="messages-thread",
        broker_pid=1,
        codex_pid=1,
        agent_backend="codex",
        owned=False,
        start_ts=0.0,
        cwd=str(tmp_path),
        log_path=large_log,
        sock_path=tmp_path / "messages.sock",
    )
    responses: list[tuple[int, dict[str, object]]] = []
    secret = b"session-critical-summary"
    deps = MessageRouteDeps(
        require_auth=lambda _handler: True,
        set_auth_cookie=lambda _handler: None,
        json_response=lambda _handler, status, body: responses.append((status, body)),
        launch_attempt_transcript_for_session_id=lambda _session_id: None,
        transcript_export_max_bytes=50 * 1024 * 1024,
        transcript_search_max_line_bytes=64 * 1024,
        encode_message_cursor=lambda *, kind, session, pos: encode_message_cursor(kind=kind, session=session, pos=pos, secret=secret),
        decode_message_cursor=lambda token, *, kind, session: decode_message_cursor(token, kind=kind, session=session, secret=secret),
        record_metric=lambda _name, _value: None,
        message_runtime_snapshot=lambda *_args, **_kwargs: ({}, False, 0, None),
    )
    manager = _MessagesManager(messages_session)
    started = time.perf_counter()
    handle_messages_tail(_MessagesHandler(), session_id=messages_session.session_id, query="limit=60", manager=manager, deps=deps)
    first_elapsed = time.perf_counter() - started
    first_response = responses.pop()
    started = time.perf_counter()
    handle_messages_tail(_MessagesHandler(), session_id=messages_session.session_id, query="limit=60", manager=manager, deps=deps)
    cached_elapsed = time.perf_counter() - started
    second_response = responses.pop()
    assert first_response == second_response
    assert first_response[0] == 200
    assert [event["text"] for event in first_response[1]["events"]] == ["latest"]
    assert first_elapsed < 0.100
    assert cached_elapsed < 0.005

    # 46fce5a3: resolve the public /pdf.mjs route and execute the served ESM
    # bundle; consumers receive the PDF.js module API used by the viewer.
    pdf_asset = static_route_asset("/pdf.mjs")
    assert pdf_asset is not None
    program = f"""
import * as pdfjs from {json.dumps((STATIC_DIR / pdf_asset).as_uri())};
process.stdout.write(JSON.stringify({{ hasGetDocument: typeof pdfjs.getDocument === "function" }}));
"""
    completed = subprocess.run(
        ["node", "--input-type=module"],
        input=program,
        check=True,
        capture_output=True,
        text=True,
        timeout=8,
    )
    assert json.loads(completed.stdout) == {"hasGetDocument": True}
