from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import pytest

from codoxear.diagnostics_routes import handle_diagnostics_get_route
from codoxear.server import SessionManager, _match_session_route
from codoxear.session_cleaners import (
    clean_alias, clean_dependency_session_id, clean_priority_offset,
    clean_snooze_until, clean_snoozed_indefinitely,
)
from codoxear.session_snapshot import IdentityRemap
from codoxear.session_ui_state import SessionUiStateCoordinator
from test_diagnostics_routes import Manager as DiagnosticsManager, Handler as DiagnosticsHandler, _deps as diagnostics_deps
from test_session_sidebar_priority import _store, _session, _list_coordinator
from test_session_snapshot_runtime import _Harness as SnapshotHarness


def ui_harness(tmp_path, *, now=100000.0, half_life=28800.0, bucket=30.0):
    store = _store(tmp_path)
    sessions = {
        sid: _session(sid=sid, start_ts=now - age)
        for sid, age in (("dragged", 50000), ("above", 28800), ("below", 57600))
    }
    lock = threading.Lock()
    saves = []

    def save():
        assert not lock.locked()
        saves.append(True)
        store.save_sidebar_meta(store.sidebar_meta)

    ui = SessionUiStateCoordinator(
        lock=lock, sessions=lambda: sessions,
        aliases=lambda: store.aliases, set_aliases=lambda value: setattr(store, "aliases", value),
        sidebar_meta=lambda: store.sidebar_meta, set_sidebar_meta=lambda value: setattr(store, "sidebar_meta", value),
        hidden_sessions=lambda: store.hidden_sessions, set_hidden_sessions=lambda value: setattr(store, "hidden_sessions", value),
        save_aliases=lambda: None, save_sidebar_meta=save, save_hidden_sessions=lambda: None,
        clean_alias=clean_alias, clean_priority_offset=clean_priority_offset,
        clean_snooze_until=clean_snooze_until, clean_dependency_session_id=clean_dependency_session_id,
    )
    listing = _list_coordinator(
        store=store, sessions=sessions, lock=lock, now=lambda: now,
        priority_half_life_seconds=half_life, priority_bucket_seconds=bucket,
    )
    manager = SessionManager.__new__(SessionManager)
    manager._coordinators = SimpleNamespace(ui_state=ui, listing=listing)
    return SimpleNamespace(store=store, sessions=sessions, ui=ui, listing=listing, manager=manager, saves=saves, now=now)


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_indefinite_cleaner_requires_boolean(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        clean_snoozed_indefinitely(value)


def test_validator_rejects_conflict_without_mutating_alias_or_meta(tmp_path):
    h = ui_harness(tmp_path)
    h.store.aliases["dragged"] = "original"
    h.store.sidebar_meta["dragged"] = {"priority_offset": 0.2}
    with pytest.raises(ValueError, match="indefinite snooze cannot have snooze_until"):
        h.manager.edit_session(
            "dragged", name="changed", priority_offset=0.5,
            snooze_until=h.now + 100, snoozed_indefinitely=True, dependency_session_id=None,
        )
    assert h.store.aliases == {"dragged": "original"}
    assert h.store.sidebar_meta == {"dragged": {"priority_offset": 0.2}}
    assert h.saves == []


def test_sparse_persistence_and_timestamp_replaces_indefinite(tmp_path):
    h = ui_harness(tmp_path)
    result = h.manager.sidebar_meta_set(
        "dragged", priority_offset=0.2, snooze_until=None,
        snoozed_indefinitely=True, dependency_session_id="above",
    )
    assert result["snoozed_indefinitely"] is True
    assert h.store.load_sidebar_meta() == {
        "dragged": {"priority_offset": 0.2, "snoozed_indefinitely": True, "dependency_session_id": "above"}
    }
    assert json.loads(h.store.paths.sidebar_meta.read_text()) == h.store.sidebar_meta
    h.manager.sidebar_meta_set(
        "dragged", priority_offset=0.2, snooze_until=h.now + 100, dependency_session_id=None,
    )
    assert h.store.load_sidebar_meta() == {"dragged": {"priority_offset": 0.2, "snooze_until": h.now + 100}}
    assert h.manager.sidebar_meta_get("dragged")["snoozed_indefinitely"] is False
    assert "snoozed_indefinitely" not in json.loads(h.store.paths.sidebar_meta.read_text())["dragged"]
    h.store.paths.sidebar_meta.write_text(json.dumps({"dragged": {"snoozed_indefinitely": False}}))
    assert h.store.load_sidebar_meta() == {"dragged": {"priority_offset": 0.0}}


def test_expiry_prune_never_changes_an_indefinite_snooze(tmp_path):
    h = ui_harness(tmp_path)
    meta = {"priority_offset": 0.2, "snoozed_indefinitely": True}
    h.store.sidebar_meta["dragged"] = dict(meta)
    state = h.store.sidebar_state_for_session("dragged", active_session_ids=set(h.sessions), now_ts=1e30)
    assert state.snoozed_indefinitely is True
    assert state.snooze_until is None
    assert state.dirty is False
    assert h.store.sidebar_meta["dragged"] == meta


@pytest.mark.parametrize("indefinite,timestamp,expected", [
    (True, None, True), (False, 100100.0, True), (False, 99999.0, False), (False, None, False),
])
def test_listing_and_diagnostics_share_snooze_derivation(tmp_path, indefinite, timestamp, expected):
    h = ui_harness(tmp_path)
    h.manager.sidebar_meta_set(
        "dragged", priority_offset=0.2, snooze_until=timestamp,
        snoozed_indefinitely=indefinite, dependency_session_id=None,
    )
    row = next(row for row in h.listing.list_sessions() if row["session_id"] == "dragged")
    manager = DiagnosticsManager(h.sessions["dragged"])
    manager.sidebar = h.manager.sidebar_meta_get("dragged")
    responses = []
    deps = replace(diagnostics_deps(responses, now=h.now),
                   sidebar_time_priority_from_elapsed_seconds=lambda elapsed: row["time_priority"])
    assert handle_diagnostics_get_route(
        DiagnosticsHandler(), path="/api/sessions/dragged/diagnostics", manager=manager,
        deps=deps, match_session_route=_match_session_route,
    )
    status, diagnostic = responses[0]
    assert status == 200
    assert row["snoozed_indefinitely"] is diagnostic["snoozed_indefinitely"] is indefinite
    assert row["snoozed"] is diagnostic["snoozed"] is expected
    assert row["final_priority"] == diagnostic["final_priority"]
    if expected:
        assert row["final_priority"] == 0.0


def test_registry_id_remap_moves_whole_indefinite_meta(tmp_path):
    h = SnapshotHarness(tmp_path)
    h.add_live_session("new", broker_pid=30, thread_id="thread", start_ts=2.0)
    meta = {"priority_offset": 0.4, "snoozed_indefinitely": True, "dependency_session_id": "other"}
    h.session_store.sidebar_meta["old"] = dict(meta)
    result = h.coordinator.remap_session_identity(IdentityRemap(
        old_session_id="old", new_session_id="new", new_agent_backend="pi",
        old_thread_id="thread", new_thread_id="thread", new_start_ts=2.0,
    ))
    assert result.sidebar_meta_moved is True
    assert "old" not in h.session_store.sidebar_meta
    assert h.session_store.sidebar_meta["new"] == meta
    assert h.saved.count("sidebar_meta") == 1
