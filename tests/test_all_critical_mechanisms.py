"""One behavioral pin for every critical shipping mechanism from this session.

Each case executes the owning production path (Python route/coordinator or the
browser module in a Node VM).  The commit comment on each test identifies the
change whose user-visible contract it preserves.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import codoxear.broker as broker_module
import codoxear.rollout_log as rollout_log
from codoxear.broker_turn_state import State
from codoxear.message_cursor import decode_message_cursor, encode_message_cursor
from codoxear.message_routes import MessageRouteDeps, handle_messages_tail
from codoxear.session_model import Session
from codoxear.session_runtime import session_run_settings_from_meta

ROOT = Path(__file__).resolve().parents[1]


def _test_module(name: str):
    """Load an existing behavioral harness without relying on package layout."""
    path = ROOT / "tests" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"critical_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# dab6445c — live Pi marker beats an obsolete launch-declared JSONL path.
def test_pi_broker_marker_wins_over_stale_declared_log_path(tmp_path: Path, monkeypatch) -> None:
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()
    stale = sessions_dir / "stale.jsonl"
    live = sessions_dir / "live.jsonl"
    stale.write_text(json.dumps({"type": "session", "id": "stale"}) + "\n", encoding="utf-8")
    live.write_text(json.dumps({"type": "session", "id": "live"}) + "\n", encoding="utf-8")
    marker = tmp_path / "active.json"
    marker.write_text(json.dumps({"version": 1, "sessionFile": str(live)}), encoding="utf-8")

    spawned = broker_module.Broker.__new__(broker_module.Broker)
    spawned._stop = threading.Event()
    spawned._lock = threading.Lock()
    spawned.state = State(
        codex_pid=os.getpid(), pty_master_fd=-1, cwd=str(tmp_path), start_ts=time.time(),
        codex_home=tmp_path, sessions_dir=sessions_dir, declared_log_path=stale,
    )
    spawned.sessions_dir = sessions_dir
    spawned.pi_active_session_marker_path = marker
    spawned._refresh_pi_active_session_observability = lambda **_kwargs: None
    spawned._write_meta = lambda: None
    chosen: list[Path] = []

    def register(*, log_path: Path) -> None:
        chosen.append(log_path)
        spawned._stop.set()

    spawned._maybe_register_or_switch_rollout = register
    monkeypatch.setattr(broker_module, "AGENT_BACKEND", "pi")
    monkeypatch.setattr(broker_module.time, "sleep", lambda _seconds: None)
    worker = threading.Thread(target=spawned._discover_log_watcher, daemon=True)
    worker.start()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert chosen == [live]


# 685782e8 — bounded tail scan and its unchanged-revision cache meet the I/O floor.
def test_messages_tail_reads_100mb_once_then_uses_cache_under_budget(tmp_path: Path, monkeypatch) -> None:
    log_path = tmp_path / "100mb.jsonl"
    target_size = 100 * 1024 * 1024
    event = {
        "type": "response_item",
        "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "latest"}], "phase": "final_answer"},
        "ts": 1.0,
    }
    line = json.dumps(event).encode() + b"\n"
    filler = b'{"type":"debug","payload":"' + b"x" * 65500 + b'"}\n'
    with log_path.open("wb") as stream:
        stream.seek(target_size - 12 * 1024 * 1024)
        while stream.tell() + len(filler) + len(line) <= target_size:
            stream.write(filler)
        remaining = target_size - stream.tell() - len(line)
        stream.write(b"x" * (remaining - 1) + b"\n")
        stream.write(line)
    assert log_path.stat().st_size == target_size

    session = Session("critical-perf", "thread", 1, 1, "codex", False, 0.0, str(tmp_path), log_path, tmp_path / "sock")
    responses: list[tuple[int, dict]] = []
    secret = b"critical-mechanisms"
    deps = MessageRouteDeps(
        require_auth=lambda _handler: True, set_auth_cookie=lambda _handler: None,
        json_response=lambda _handler, status, body: responses.append((status, body)),
        launch_attempt_transcript_for_session_id=lambda _sid: None,
        transcript_export_max_bytes=50 * 1024 * 1024, transcript_search_max_line_bytes=64 * 1024,
        encode_message_cursor=lambda *, kind, session, pos: encode_message_cursor(kind=kind, session=session, pos=pos, secret=secret),
        decode_message_cursor=lambda token, *, kind, session: decode_message_cursor(token, kind=kind, session=session, secret=secret),
        record_metric=lambda _name, _value: None, message_runtime_snapshot=lambda *_args, **_kwargs: ({}, False, 0, None),
    )

    class Manager:
        def refresh_session_meta(self, _sid): pass
        def get_session(self, _sid): return session
        def mark_log_delta(self, *_args, **_kwargs): pass
        def _attach_notification_texts(self, events): return events
    class Handler:
        def _unauthorized(self): raise AssertionError("route unexpectedly unauthenticated")

    rollout_log._TAIL_PAGE_CACHE.clear()
    rollout_log._TAIL_PAGE_CACHE_ORDER.clear()
    page_reads: list[Path] = []
    read_chat_page_reverse = rollout_log._read_chat_page_reverse

    def count_tail_page_read(path: Path, **kwargs):
        page_reads.append(path)
        return read_chat_page_reverse(path, **kwargs)

    monkeypatch.setattr(rollout_log, "_read_chat_page_reverse", count_tail_page_read)
    start = time.perf_counter()
    handle_messages_tail(Handler(), session_id=session.session_id, query="limit=60", manager=Manager(), deps=deps)
    first = time.perf_counter() - start
    first_response = responses.pop()
    start = time.perf_counter()
    handle_messages_tail(Handler(), session_id=session.session_id, query="limit=60", manager=Manager(), deps=deps)
    cached = time.perf_counter() - start
    second_response = responses.pop()
    assert first_response == second_response
    assert first_response[0] == 200
    assert [item["text"] for item in first_response[1]["events"]] == ["latest"]
    assert page_reads == [log_path]
    # A busy test runner may deschedule a request; cache identity, not a
    # sub-millisecond wall-clock sample, proves the unchanged log was scanned
    # once. These remain generous route-latency ceilings.
    assert first < 1.0
    assert cached < 0.1


# 333dfe8d — a running Pi bridge supplies newer effort than delayed JSONL replay.
def test_live_pi_bridge_effort_overrides_stale_log_replay(tmp_path: Path) -> None:
    log_path = tmp_path / "pi.jsonl"
    log_path.write_text("{}\n", encoding="utf-8")
    result = session_run_settings_from_meta(
        meta={"broker_pid": os.getpid(), "reasoning_effort": "low", "live_run_settings": {"reasoning_effort": "high"}},
        log_path=log_path, agent_backend="pi", clean_optional_text=lambda value: value if isinstance(value, str) else None,
        normalize_requested_preferred_auth_method=lambda value: value if isinstance(value, str) else None,
        display_reasoning_effort=lambda value: value if isinstance(value, str) else None,
        display_pi_reasoning_effort=lambda value: value if isinstance(value, str) else None,
        normalize_requested_cc_reasoning_effort=lambda value: value if isinstance(value, str) else None,
        read_run_settings_from_log=lambda *_args, **_kwargs: (None, None, "max"),
    )
    assert result == (None, None, None, "high")


# d6d3a1da — A's completion drains B's pending merged unattended update.
def test_unattended_inflight_a_then_b_reconciles_to_b() -> None:
    module = _test_module("test_unattended_edit_reconcile")
    module.test_in_flight_config_a_finishes_before_pending_config_b_applies()


# 54203bb4 — busy send choice keeps direct send, queue, and cancel distinct.
def test_busy_send_choice_now_later_cancel_are_distinct() -> None:
    module = _test_module("test_send_choice")
    module.test_busy_send_choice_routes_now_later_and_cancel_through_distinct_actions()


# 1cdba92b — keyboard dispatch is scoped to the topmost nested dialog.
def test_nested_dialog_keyboard_activates_inner_button() -> None:
    source = (ROOT / "codoxear/static/app_modal.js").read_text(encoding="utf-8")
    script = f"""
const vm = require('vm'); const clicks = [];
const button = {{ textContent: 'Proceed', disabled: false, hidden: false, getAttribute: () => null,
  getClientRects: () => [{{}}], click: () => clicks.push('inner') }};
const outer = {{ style: {{ display: 'flex' }}, querySelectorAll: () => [{{ textContent: 'Delete', disabled: false, hidden: false, getAttribute: () => null, getClientRects: () => [{{}}], click: () => clicks.push('outer') }}] }};
const inner = {{ style: {{ display: 'flex' }}, querySelectorAll: () => [button] }};
const ctx = {{ window: {{}} }}; vm.createContext(ctx); vm.runInContext({json.dumps(source)}, ctx);
const handler = ctx.window.CodoxearModal.createModalKeyboardHandler({{ modalIsolationTargets: [inner, outer], isTextEntryElement: () => false }});
const event = {{ key: 'p', altKey:false, ctrlKey:false, metaKey:false, isComposing:false, preventDefault(){{}}, stopPropagation(){{}} }};
process.stdout.write(JSON.stringify({{ activated: handler(event), clicks }}));
"""
    result = json.loads(subprocess.run(["node", "-e", script], check=True, capture_output=True, text=True).stdout)
    assert result == {"activated": True, "clicks": ["inner"]}


# 0a13843b — persisted voice opt-in reconnects after controller recreation.
def test_voice_enabled_state_survives_controller_recreation() -> None:
    module = _test_module("test_voice_resume")
    result = module.run_voice_resume_harness()
    assert result["enabledAfterReload"] is True
    assert result["afterPageReload"] == 1


# 4d8c4549 — nested fence delimiters remain literal code inside the outer fence.
def test_markdown_nested_fences_render_as_literal_inner_fence() -> None:
    module = _test_module("test_app_markdown_extended")
    html = module.render_markdown("````markdown\n```js\nconst x = 1;\n```\n````")
    assert '<pre><code class="language-markdown">```js\nconst x = 1;\n```' in html


# 46fce5a3 — the vendored PDF.js module opens an actual one-page PDF.
def test_pdf_get_document_loads_minimal_pdf() -> None:
    module = _test_module("test_pdf_viewer_pipeline")
    module.test_vendored_pdfjs_parses_page_and_extracts_text_in_node()


# a52f1582 — Pi intercom delivery records feed the browser notification panel.
def test_pi_intercom_messages_appear_in_notification_feed(tmp_path: Path) -> None:
    module = _test_module("test_notification_feed_all_backends")
    module.test_notification_feed_and_read_state_cover_pi_codex_and_claude_code(tmp_path)


# 414c5562 — a dead broker becomes a retained lost tombstone before sidecar cleanup.
def test_watchdog_dead_broker_sidecars_become_lost_tombstones(tmp_path: Path) -> None:
    module = _test_module("test_broker_watchdog")
    module.test_watchdog_projects_lost_tombstone_then_prunes_stale_sidecar_after_grace(tmp_path)


# 51e4a30e — browser offline state is immediately visible rather than silent.
def test_offline_navigator_state_shows_banner() -> None:
    module = _test_module("test_offline_resilience")
    rendered = module.render_network_status(on_line=False)
    assert rendered["offline"] is True
    assert rendered["hidden"] is False
    assert rendered["text"] == "Offline — waiting for a network connection. Updates retry automatically."
