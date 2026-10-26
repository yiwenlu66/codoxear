from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from codoxear.session_snapshot import IdentityRemap
from codoxear.session_snapshot import RESTORING_TTL_SECONDS
from codoxear.session_snapshot import RestoringMark
from codoxear.session_snapshot import SessionSnapshotStore
from codoxear.session_snapshot import SnapshotCorrupt
from codoxear.session_snapshot import SnapshotEntry
from codoxear.session_snapshot import SnapshotState
from codoxear.session_snapshot import SnapshotWriteRefused
from codoxear.session_snapshot import snapshot_path


def _entry(session_id: str, *, thread_id: str | None = None, log_path: str | None = None, **overrides) -> SnapshotEntry:
    values = dict(
        session_id=session_id,
        thread_id=thread_id or f"thread-{session_id}",
        agent_backend="pi",
        log_path=log_path or f"/logs/{session_id}.jsonl",
        cwd="/repo",
        model_provider="dexgem",
        preferred_auth_method=None,
        model="kimi",
        reasoning_effort="high",
        service_tier=None,
        tmux=True,
        saved_at=0.0,
        start_ts=1.0,
    )
    values.update(overrides)
    return SnapshotEntry(**values)


def _store(tmp_path: Path, *, clock: list[float] | None = None) -> SessionSnapshotStore:
    ticks = iter(clock or [100.0 + i for i in range(40)])
    return SessionSnapshotStore(snapshot_path(tmp_path), now=lambda: next(ticks))


def _keys(store: SessionSnapshotStore) -> set[tuple[str, str]]:
    return set(store.read().entries)


def test_missing_file_reads_as_empty_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.read() == SnapshotState(saved_at=None, entries={})
    assert not store.exists()


def test_upsert_round_trips_entries_keyed_by_transcript_with_private_permissions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    result = store.upsert([_entry("broker-1"), _entry("broker-2", tmux=False)])

    assert result.changed == [("pi", "thread-broker-1"), ("pi", "thread-broker-2")]
    assert result.remaps == []
    mode = stat.S_IMODE(os.stat(store.path).st_mode)
    assert mode == 0o600
    on_disk = json.loads(store.path.read_text(encoding="utf-8"))
    assert on_disk["version"] == 1
    assert [row["session_id"] for row in on_disk["entries"]] == ["broker-1", "broker-2"]
    assert "pending_remaps" not in on_disk
    state = store.read()
    assert state.saved_at is not None
    entry = state.entries[("pi", "thread-broker-2")]
    assert entry.tmux is False
    assert entry.session_id == "broker-2"
    assert entry.start_ts == 1.0
    assert entry.restoring is None
    assert state.entries[("pi", "thread-broker-1")].launch_fields() == _entry("broker-1").launch_fields()


def test_upsert_is_a_no_op_for_unchanged_launch_fields(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert([_entry("broker-1")])
    first = store.read()
    before = store.path.stat().st_mtime_ns

    assert store.upsert([_entry("broker-1", saved_at=999.0)]).changed == []
    assert store.path.stat().st_mtime_ns == before
    assert store.read() == first


def test_upsert_updates_changed_launch_fields_only(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert([_entry("broker-1"), _entry("broker-2")])
    unchanged_saved_at = store.read().entries[("pi", "thread-broker-2")].saved_at

    result = store.upsert([_entry("broker-1", reasoning_effort="low"), _entry("broker-2")])
    assert result.changed == [("pi", "thread-broker-1")]
    state = store.read()
    assert state.entries[("pi", "thread-broker-1")].reasoning_effort == "low"
    assert state.entries[("pi", "thread-broker-2")].saved_at == unchanged_saved_at


def test_upsert_never_removes_unrelated_entries(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert([_entry("broker-1"), _entry("broker-2")])

    store.upsert([_entry("broker-3")])
    assert _keys(store) == {("pi", "thread-broker-1"), ("pi", "thread-broker-2"), ("pi", "thread-broker-3")}

    assert store.upsert([]).changed == []
    assert len(_keys(store)) == 3


def test_rule_a_same_transcript_new_registry_id_updates_in_place_and_reports_remap(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert([_entry("broker-1", thread_id="t1", log_path="/logs/t1.jsonl"), _entry("broker-9")])

    resumed = _entry("broker-2", thread_id="t1", log_path="/logs/t1-new.jsonl", start_ts=2.0)
    result = store.upsert([resumed])

    assert result.changed == [("pi", "t1")]
    assert result.remaps == [IdentityRemap(old_session_id="broker-1", new_session_id="broker-2", old_thread_id="t1", new_thread_id="t1", new_agent_backend="pi", new_start_ts=2.0)]
    state = store.read()
    assert _keys(store) == {("pi", "t1"), ("pi", "thread-broker-9")}
    assert state.entries[("pi", "t1")].session_id == "broker-2"
    assert state.entries[("pi", "t1")].start_ts == 2.0
    assert state.entries[("pi", "t1")].log_path == "/logs/t1-new.jsonl"


def test_rule_b_same_incarnation_moving_to_a_new_thread_retires_the_old_key(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert([_entry("broker-1", thread_id="t-old", start_ts=5.0)])

    result = store.upsert([_entry("broker-1", thread_id="t-new", log_path="/logs/t-new.jsonl", start_ts=5.0)])

    assert result.remaps == []
    assert _keys(store) == {("pi", "t-new")}


def test_pid_reuse_does_not_clobber_a_dead_entry(tmp_path: Path) -> None:
    """Critic repro B: a new unrelated broker with the same pid-derived id must not overwrite the dead session's entry."""
    store = _store(tmp_path)
    store.upsert([_entry("broker-100", thread_id="thread-A", start_ts=1.0)])

    result = store.upsert([_entry("broker-100", thread_id="thread-B", log_path="/logs/B.jsonl", start_ts=2.0)])

    assert result.remaps == []
    state = store.read()
    assert _keys(store) == {("pi", "thread-A"), ("pi", "thread-B")}
    assert state.entries[("pi", "thread-A")].session_id == "broker-100"
    assert state.entries[("pi", "thread-A")].start_ts == 1.0


def test_rule_c_resume_target_or_shared_log_continues_under_a_new_thread_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert([_entry("broker-1", thread_id="t1", log_path="/logs/t1.jsonl", agent_backend="codex"), _entry("broker-9")])

    by_resume = _entry("broker-2", thread_id="t1-renamed", log_path="/logs/t1-new.jsonl", agent_backend="codex", resume_session_id="t1", start_ts=2.0)
    result = store.upsert([by_resume])
    assert result.remaps == [IdentityRemap(old_session_id="broker-1", new_session_id="broker-2", old_thread_id="t1", new_thread_id="t1-renamed", new_agent_backend="codex", new_start_ts=2.0)]
    assert _keys(store) == {("codex", "t1-renamed"), ("pi", "thread-broker-9")}

    by_log = _entry("broker-3", thread_id="t1-again", log_path="/logs/t1-new.jsonl", agent_backend="codex", start_ts=3.0)
    result = store.upsert([by_log])
    assert result.remaps == [IdentityRemap(old_session_id="broker-2", new_session_id="broker-3", old_thread_id="t1-renamed", new_thread_id="t1-again", new_agent_backend="codex", new_start_ts=3.0)]
    assert _keys(store) == {("codex", "t1-again"), ("pi", "thread-broker-9")}

    other_backend = _entry("broker-4", thread_id="t1-again", log_path="/logs/t1-new.jsonl", agent_backend="pi", start_ts=4.0)
    store.upsert([other_backend])
    assert _keys(store) == {("codex", "t1-again"), ("pi", "t1-again"), ("pi", "thread-broker-9")}


def test_upsert_clears_a_restoring_mark_on_supersede(tmp_path: Path) -> None:
    store = _store(tmp_path)
    old = _entry("broker-1", thread_id="t1")
    store.upsert([old])
    assert store.mark_restoring(old.key, expected_session_id="broker-1", mark=RestoringMark(broker_pid=77, launched_at=10.0)) is True
    assert store.read().entries[old.key].restoring == RestoringMark(broker_pid=77, launched_at=10.0)

    result = store.upsert([_entry("broker-2", thread_id="t1", start_ts=2.0)])
    assert result.remaps[0].new_session_id == "broker-2"
    assert store.read().entries[old.key].restoring is None


def test_mark_restoring_is_refused_once_the_entry_moved_or_vanished(tmp_path: Path) -> None:
    store = _store(tmp_path)
    old = _entry("broker-1", thread_id="t1")
    store.upsert([old])
    store.upsert([_entry("broker-2", thread_id="t1", start_ts=2.0)])

    assert store.mark_restoring(old.key, expected_session_id="broker-1", mark=RestoringMark(broker_pid=None, launched_at=1.0)) is False
    assert store.read().entries[old.key].restoring is None
    assert store.mark_restoring(("pi", "nope"), expected_session_id="broker-1", mark=RestoringMark(broker_pid=None, launched_at=1.0)) is False


def test_restoring_mark_expires_after_ttl() -> None:
    mark = RestoringMark(broker_pid=None, launched_at=1000.0)
    assert mark.active(1000.0) is True
    assert mark.active(1000.0 + RESTORING_TTL_SECONDS - 1) is True
    assert mark.active(1000.0 + RESTORING_TTL_SECONDS) is False
    assert mark.active(999.0) is False


def test_empty_write_over_non_empty_snapshot_is_refused_unless_explicit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert([_entry("broker-1")])
    previous = store.read()

    with pytest.raises(SnapshotWriteRefused):
        store._write(previous, SnapshotState(saved_at=1.0, entries={}), allow_empty=False)
    assert store.read() == previous

    assert store.remove([("pi", "thread-broker-1")]) == [("pi", "thread-broker-1")]
    assert store.read().entries == {}


def test_remove_matching_accepts_session_id_or_thread_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert([_entry("broker-1", thread_id="t1"), _entry("broker-2", thread_id="t2"), _entry("broker-3", thread_id="t3")])
    saved_at = store.read().saved_at

    assert store.remove_matching("broker-2") == [("pi", "t2")]
    assert store.remove_matching("t3") == [("pi", "t3")]
    assert store.remove_matching("unknown") == []
    assert store.remove([("pi", "nope")]) == []
    state = store.read()
    assert set(state.entries) == {("pi", "t1")}
    assert state.saved_at == saved_at


def test_remove_incarnation_only_touches_that_broker_process(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert([_entry("broker-100", thread_id="A", start_ts=1.0), _entry("broker-100", thread_id="B", log_path="/logs/B.jsonl", start_ts=2.0)])

    assert store.remove_incarnation("broker-100", 2.0) == [("pi", "B")]
    assert _keys(store) == {("pi", "A")}
    assert store.remove_incarnation("broker-100", 3.0) == []


def test_read_is_cached_until_the_file_changes(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.upsert([_entry("broker-1")])
    first = store.read()
    assert store.read() is first

    other = SessionSnapshotStore(store.path, now=lambda: 500.0)
    other.upsert([_entry("broker-2")])
    assert set(store.read().entries) == {("pi", "thread-broker-1"), ("pi", "thread-broker-2")}


@pytest.mark.parametrize(
    "raw",
    [
        "{not json",
        "[]",
        json.dumps({"version": 2, "entries": []}),
        json.dumps({"version": 1, "entries": {}}),
        json.dumps({"version": 1, "entries": [{"session_id": "x"}]}),
        json.dumps({"version": 1, "entries": [{**_entry("broker-1").to_dict(), "restoring": {"broker_pid": 1}}]}),
    ],
)
def test_corrupt_snapshot_raises_a_clear_error_and_is_not_overwritten_by_upsert(tmp_path: Path, raw: str) -> None:
    store = _store(tmp_path)
    store.path.write_text(raw, encoding="utf-8")

    with pytest.raises(SnapshotCorrupt):
        store.read()
    with pytest.raises(SnapshotCorrupt):
        store.upsert([_entry("broker-1")])
    assert store.path.read_text(encoding="utf-8") == raw


def test_snapshot_written_before_the_identity_redesign_still_loads(tmp_path: Path) -> None:
    store = _store(tmp_path)
    legacy = {k: v for k, v in _entry("broker-1", thread_id="t1").to_dict().items() if k not in {"start_ts", "resume_session_id", "restoring"}}
    store.path.write_text(json.dumps({"version": 1, "saved_at": 1.0, "entries": [legacy], "pending_remaps": [{"old_session_id": "x"}]}), encoding="utf-8")

    state = store.read()
    entry = state.entries[("pi", "t1")]
    assert entry.session_id == "broker-1"
    assert entry.start_ts == 0.0
    assert entry.resume_session_id is None
    assert entry.restoring is None


def test_entry_dict_round_trip_preserves_optional_fields() -> None:
    entry = _entry(
        "broker-7",
        preferred_auth_method="apikey",
        service_tier="fast",
        agent_backend="codex",
        tmux=False,
        saved_at=12.5,
        resume_session_id="older",
        restoring=RestoringMark(broker_pid=9, launched_at=3.0),
    )
    assert SnapshotEntry.from_dict(entry.to_dict()) == entry
    assert SnapshotEntry.from_dict({**entry.to_dict(), "model": "  "}).model is None
    assert SnapshotEntry.from_dict({**entry.to_dict(), "restoring": {"broker_pid": True, "launched_at": 3.0}}).restoring == RestoringMark(broker_pid=None, launched_at=3.0)
    assert entry.key == ("codex", "thread-broker-7")
