"""The alias store and the name projector compose into the default-name rule.

This is the unit-level shape of the production wiring: one registry lock, one
alias map, one projector. It pins that two sessions in the same directory get
distinct default names (and that the projector's write and the store's assign
share the store without deadlocking), which the end-to-end Docker verifier can
only show for a single session.
"""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from codoxear.session_cleaners import (
    clean_alias,
    clean_dependency_session_id,
    clean_priority_offset,
    clean_snooze_until,
)
from codoxear.session_name_projection import SessionNameProjection
from codoxear.session_name_projection import SessionNameProjectionDeps
from codoxear.session_ui_state import SessionUiStateCoordinator


class _Harness:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.sessions = {
            "s1": self._session("s1", thread_id="t1"),
            "s2": self._session("s2", thread_id="t2"),
        }
        self.aliases: dict[str, str] = {}
        self.applied: dict[str, dict[str, Any]] = {}
        self.pushes: list[dict[str, Any]] = []
        self.alias_saves = 0
        self.applied_saves = 0
        self.ui = SessionUiStateCoordinator(
            lock=self.lock,
            sessions=lambda: self.sessions,
            aliases=lambda: self.aliases,
            set_aliases=lambda value: None,
            sidebar_meta=lambda: {},
            set_sidebar_meta=lambda value: None,
            hidden_sessions=lambda: set(),
            set_hidden_sessions=lambda value: None,
            save_aliases=self._save_aliases,
            save_sidebar_meta=lambda: None,
            save_hidden_sessions=lambda: None,
            clean_alias=clean_alias,
            clean_priority_offset=clean_priority_offset,
            clean_snooze_until=clean_snooze_until,
            clean_dependency_session_id=clean_dependency_session_id,
        )
        self.projection = SessionNameProjection(
            SessionNameProjectionDeps(
                lock=self.lock,
                sessions=lambda: self.sessions,
                alias_get=self.ui.alias_get,
                alias_set=self.ui.alias_set,
                alias_cleared=self.ui.alias_cleared,
                alias_assign_default=self.ui.alias_assign_default,
                sock_call=self._sock_call,
                applied=lambda: self.applied,
                set_applied=lambda value: setattr(self, "applied", value),
                save_applied=self._save_applied,
                now=lambda: 1000.0,
                stderr=None,
            )
        )

    @staticmethod
    def _session(session_id: str, *, thread_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            agent_backend="pi",
            thread_id=thread_id,
            cwd="/home/tester/repo",
            sock_path=Path(f"/tmp/{session_id}.sock"),
            log_path=None,
            session_name_control=True,
            backend_session_name="",
        )

    def _sock_call(self, sock: Any, req: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        self.pushes.append({"sock": sock, **req})
        return {"ok": True, "backend": "pi", "name": req.get("name", "")}

    def _save_aliases(self) -> None:
        assert not self.lock.locked(), "aliases are saved outside the registry lock"
        self.alias_saves += 1

    def _save_applied(self) -> None:
        assert not self.lock.locked(), "applied records are saved outside the registry lock"
        self.applied_saves += 1


def test_two_sessions_in_one_directory_get_distinct_default_names() -> None:
    h = _Harness()

    assert h.projection.reconcile("s1")["name"] == "repo"
    assert h.projection.reconcile("s2")["name"] == "repo-2"
    assert h.aliases == {"s1": "repo", "s2": "repo-2"}
    assert [push["name"] for push in h.pushes] == ["repo", "repo-2"]
    assert [push["thread_id"] for push in h.pushes] == ["t1", "t2"]

    # Both names are settled: another refresh pushes nothing at all.
    assert h.projection.reconcile("s1")["reason"] == "unchanged"
    assert h.projection.reconcile("s2")["reason"] == "unchanged"
    assert len(h.pushes) == 2


def test_a_terminal_title_supersedes_a_default_but_never_a_user_name() -> None:
    """Both writes land in the same store, so the two names cannot disagree."""
    h = _Harness()
    h.projection.reconcile("s1")

    # A name the user chose in codoxear is the authority: a stale backend title
    # does not displace it.
    h.ui.alias_set("s1", "build pipeline")
    assert h.projection.reconcile("s1")["name"] == "build pipeline"
    assert h.pushes[-1]["name"] == "build pipeline"
    h.sessions["s1"].backend_session_name = "typed in the terminal"
    assert h.projection.reconcile("s1")["reason"] == "unchanged"
    assert h.ui.alias_get("s1") == "build pipeline"

    # A name codoxear only assigned yields to the one the user types in the
    # backend's own TUI — adoption, which is a local store write, not a push.
    h.projection.reconcile("s2")
    assert h.ui.alias_get("s2") == "repo"  # s1 no longer holds the plain name
    h.sessions["s2"].backend_session_name = "typed in the terminal"
    pushes = len(h.pushes)
    assert h.projection.reconcile("s2")["reason"] == "adopted"
    assert h.ui.alias_get("s2") == "typed in the terminal"
    assert len(h.pushes) == pushes
