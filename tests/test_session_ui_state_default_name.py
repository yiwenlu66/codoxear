"""The alias store's default-name assignment, exercised through the real store coordinator."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from codoxear.session_cleaners import (
    clean_alias,
    clean_dependency_session_id,
    clean_priority_offset,
    clean_snooze_until,
    clean_snoozed_indefinitely,
)
from codoxear.session_ui_state import SessionUiStateCoordinator


def _session(*, cwd: str, observed: str = "") -> SimpleNamespace:
    return SimpleNamespace(cwd=cwd, backend_session_name=observed)


def _harness(*, sessions: dict[str, SimpleNamespace] | None = None) -> SimpleNamespace:
    lock = threading.Lock()
    sessions = sessions if sessions is not None else {"s1": _session(cwd="/home/tester/repo")}
    aliases: dict[str, str] = {}
    saves: list[dict[str, str]] = []

    def save_aliases() -> None:
        assert not lock.locked(), "the store is saved outside the lock"
        saves.append(dict(aliases))

    ui = SessionUiStateCoordinator(
        lock=lock,
        sessions=lambda: sessions,
        aliases=lambda: aliases,
        set_aliases=lambda value: None,
        sidebar_meta=lambda: {},
        set_sidebar_meta=lambda value: None,
        hidden_sessions=lambda: set(),
        set_hidden_sessions=lambda value: None,
        save_aliases=save_aliases,
        save_sidebar_meta=lambda: None,
        save_hidden_sessions=lambda: None,
        clean_alias=clean_alias,
        clean_priority_offset=clean_priority_offset,
        clean_snooze_until=clean_snooze_until,
        clean_dependency_session_id=clean_dependency_session_id,
    )
    return SimpleNamespace(ui=ui, sessions=sessions, aliases=aliases, saves=saves)


def test_an_unnamed_session_is_named_after_its_directory() -> None:
    h = _harness()
    assert h.ui.alias_assign_default("s1") == "repo"
    assert h.aliases == {"s1": "repo"}
    assert h.saves == [{"s1": "repo"}]


def test_a_name_is_assigned_once() -> None:
    h = _harness()
    h.ui.alias_assign_default("s1")
    h.ui.alias_set("s1", "build pipeline")
    assert h.ui.alias_assign_default("s1") == "build pipeline"
    assert h.aliases == {"s1": "build pipeline"}


def test_an_explicit_clear_is_never_re_named() -> None:
    h = _harness()
    h.ui.alias_set("s1", "")
    assert h.ui.alias_assign_default("s1") == ""
    assert h.aliases == {"s1": ""}
    assert h.saves == [{"s1": ""}]  # the clear, not a name


def test_live_assigned_names_are_avoided_however_they_were_set() -> None:
    sessions = {
        "named": _session(cwd="/elsewhere/other"),
        "terminal": _session(cwd="/elsewhere/third", observed="repo-2"),
        "unnamed": _session(cwd="/home/tester/repo"),
        "target": _session(cwd="/home/tester/repo"),
    }
    h = _harness(sessions=sessions)
    h.aliases["named"] = "repo"
    # Two names are taken: one the user chose in codoxear, one typed in the
    # backend's own TUI. A session with no name yet does not occupy one.
    assert h.ui.alias_assign_default("target") == "repo-3"
    assert h.aliases["target"] == "repo-3"


def test_numbering_follows_birth_order_not_reconcile_order() -> None:
    sessions = {
        "older": _session(cwd="/home/tester/repo"),
        "newer": _session(cwd="/home/tester/repo"),
    }
    h = _harness(sessions=sessions)
    assert h.ui.alias_assign_default("older") == "repo"
    assert h.ui.alias_assign_default("newer") == "repo-2"


def test_unknown_session_raises_and_writes_nothing() -> None:
    h = _harness()
    with pytest.raises(KeyError):
        h.ui.alias_assign_default("nope")
    assert h.aliases == {}
    assert h.saves == []
