"""Behavior of quick-session workspace allocation, naming, and containment."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from codoxear import quick_workspace as qw
from codoxear.launch_ledger import launch_attempt_row
from codoxear.session_cleaners import clean_alias
from codoxear.session_cleaners import clean_dependency_session_id
from codoxear.session_cleaners import clean_priority_offset
from codoxear.session_cleaners import clean_recent_cwd
from codoxear.session_cleaners import clean_snooze_until
from codoxear.session_store import SessionStore
from codoxear.session_store import SessionStorePaths


def test_one_slot_number_names_the_label_and_the_directory() -> None:
    assert qw.quick_dir_name(1) == "quick-01"
    assert qw.quick_display_name(1) == "Quick Session 01"
    assert qw.quick_dir_name(99) == "quick-99"
    assert qw.quick_cwd(7, root=Path("/home/u/codoxear-quick")) == Path("/home/u/codoxear-quick/quick-07")


def test_padding_is_a_minimum_width_so_slot_100_is_not_an_error() -> None:
    assert qw.quick_display_name(100) == "Quick Session 100"
    assert qw.quick_dir_name(100) == "quick-100"
    assert qw.quick_cwd(1000, root=Path("/home/u/codoxear-quick")).name == "quick-1000"


def test_allocate_is_monotonic_persistent_and_creates_no_directory() -> None:
    with TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "codoxear-quick"
        state = base / "app" / "quick_sessions.json"
        reserved = [qw.allocate(state_path=state, root=root) for _ in range(3)]
        assert [item["slot"] for item in reserved] == [1, 2, 3]
        assert reserved[0] == {
            "slot": 1,
            "name": "Quick Session 01",
            "cwd": str(root / "quick-01"),
        }
        # The launch path creates the workspace; an unused reservation leaves no
        # filesystem state at all.
        assert not root.exists()
        assert json.loads(state.read_text()) == {"next_slot": 4}


def test_allocate_never_lands_on_an_existing_workspace() -> None:
    with TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "codoxear-quick"
        root.mkdir()
        for slot in (1, 2, 5):
            (root / qw.quick_dir_name(slot)).mkdir()
        state = base / "quick_sessions.json"
        state.write_text(json.dumps({"next_slot": 1}))
        assert qw.allocate(state_path=state, root=root)["slot"] == 6
        state.write_text(json.dumps({"next_slot": 9}))
        assert qw.allocate(state_path=state, root=root)["slot"] == 9


def test_a_malformed_counter_cannot_collide_with_existing_workspaces() -> None:
    with TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "codoxear-quick"
        root.mkdir()
        (root / "quick-04").mkdir()
        state = base / "quick_sessions.json"
        for payload in ("not json", json.dumps({"next_slot": "x"}), json.dumps({"next_slot": 0}), json.dumps([1])):
            state.write_text(payload)
            assert qw.allocate(state_path=state, root=root)["slot"] == 5
            state.unlink()


def test_containment_and_slot_parsing() -> None:
    root = Path("/home/u/codoxear-quick")
    assert qw.is_quick_cwd("/home/u/codoxear-quick/quick-07", root=root)
    assert qw.is_quick_cwd("/home/u/codoxear-quick/quick-07/sub", root=root)
    assert not qw.is_quick_cwd("/home/u/codoxear-quick-extra/quick-07", root=root)
    assert not qw.is_quick_cwd("/home/u/repo", root=root)
    assert not qw.is_quick_cwd("", root=root)
    assert qw.slot_from_cwd("/home/u/codoxear-quick/quick-07", root=root) == 7
    assert qw.slot_from_cwd("/home/u/codoxear-quick/quick-07/sub", root=root) is None
    assert qw.slot_from_cwd("/home/u/codoxear-quick/notes", root=root) is None
    assert qw.quick_display_name_for_cwd("/home/u/codoxear-quick/quick-100", root=root) == "Quick Session 100"
    assert qw.quick_display_name_for_cwd("/home/u/repo", root=root) == ""


def test_containment_survives_a_symlinked_root() -> None:
    with TemporaryDirectory() as tmp:
        base = Path(tmp)
        real = base / "volume" / "codoxear-quick"
        real.mkdir(parents=True)
        alias = base / "codoxear-quick"
        alias.symlink_to(real)
        slot_dir = real / "quick-03"
        slot_dir.mkdir()
        # A session records the resolved path; the configured root is the symlink.
        assert qw.is_quick_cwd(str(slot_dir), root=alias)
        assert qw.slot_from_cwd(str(slot_dir), root=alias) == 3
        assert qw.quick_display_name_for_cwd(str(slot_dir), root=alias) == "Quick Session 03"


def test_release_trims_only_an_empty_quick_workspace() -> None:
    with TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "codoxear-quick"
        empty = root / "quick-02"
        empty.mkdir(parents=True)
        used = root / "quick-03"
        used.mkdir()
        (used / "notes.txt").write_text("keep")
        other = base / "repo"
        other.mkdir()
        assert qw.release_workspace(str(empty), root=root) == ""
        assert not empty.exists()
        assert qw.release_workspace(str(used), root=root) == str(used)
        assert (used / "notes.txt").read_text() == "keep"
        assert qw.release_workspace(str(other), root=root) == ""
        assert other.is_dir()


def _store(base: Path, quick_root: Path) -> SessionStore:
    return SessionStore(
        paths=SessionStorePaths(
            aliases=base / "aliases.json",
            sidebar_meta=base / "sidebar.json",
            hidden_sessions=base / "hidden.json",
            files=base / "files.json",
            queues=base / "queues.json",
            pending_attachments=base / "pending.json",
            commit_unknown_sends=base / "commit_unknown.json",
            recent_cwds=base / "recent_cwds.json",
            unattended=base / "unattended.json",
            drafts=base / "drafts.json",
        ),
        file_history_max=20,
        recent_cwd_max=8,
        unattended_default_idle_minutes=10,
        unattended_default_max_injections=3,
        clean_alias=clean_alias,
        clean_priority_offset=clean_priority_offset,
        clean_snooze_until=clean_snooze_until,
        clean_dependency_session_id=clean_dependency_session_id,
        clean_recent_cwd=clean_recent_cwd,
        clean_commit_unknown_send_record=lambda raw: None,
        quick_root=quick_root,
    )


def test_quick_workspaces_never_reach_the_recent_cwd_picker() -> None:
    with TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "codoxear-quick"
        store = _store(base, root)
        assert store.note_recent_cwd(str(root / "quick-01"), 10.0) is False
        assert store.note_recent_cwd("/repo", 20.0) is True
        # Entries written before this rule existed are filtered on read too, so
        # they cannot crowd real directories out of the capped list.
        store.recent_cwds[str(root / "quick-09")] = 30.0
        assert store.list_recent_cwds(limit=5) == ["/repo"]


def test_a_provisional_launch_row_names_itself_from_its_slot() -> None:
    # A launch-attempt row has no alias by construction, so it would otherwise
    # show the bare directory name until the session row's rename lands.
    root = Path("/home/u/codoxear-quick")
    record = {
        "launch_id": "launch-1",
        "state": "starting",
        "cwd": str(root / "quick-05"),
        "created_ts": 1.0,
        "updated_ts": 1.0,
        "agent_backend": "pi",
    }
    row = launch_attempt_row(
        record,
        default_agent_backend="pi",
        unattended_default_idle_minutes=10,
        unattended_default_max_injections=3,
        quick_root=root,
    )
    assert row is not None and row["display_name"] == "Quick Session 05"
    plain = launch_attempt_row(
        {**record, "cwd": "/home/u/repo"},
        default_agent_backend="pi",
        unattended_default_idle_minutes=10,
        unattended_default_max_injections=3,
        quick_root=root,
    )
    assert plain is not None and plain["display_name"] == "repo"
