from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from codoxear.session_name_projection import SessionNameProjection
from codoxear.session_name_projection import SessionNameProjectionDeps
from codoxear.session_name_projection import identity_key


class Harness:
    def __init__(self, *, backend: str = "pi", thread_id: str = "thread-1", control: bool = True) -> None:
        self.lock = threading.Lock()
        self.session = SimpleNamespace(
            agent_backend=backend,
            thread_id=thread_id,
            sock_path=Path("/tmp/fake.sock"),
            log_path=Path("/tmp/fake.jsonl"),
            session_name_control=control,
            backend_session_name="",
        )
        self.sessions = {"s1": self.session}
        self.aliases: dict[str, str] = {}
        self.applied: dict[str, dict[str, Any]] = {}
        self.saves = 0
        self.calls: list[dict[str, Any]] = []
        self.replies: list[dict[str, Any]] = []
        self.alias_writes: list[tuple[str, str]] = []
        self.fail_next: list[Exception] = []
        self.now = 1000.0
        self.projection = SessionNameProjection(
            SessionNameProjectionDeps(
                lock=self.lock,
                sessions=lambda: self.sessions,
                alias_get=lambda sid: self.aliases.get(sid, ""),
                alias_set=self._alias_set,
                sock_call=self._sock_call,
                applied=lambda: self.applied,
                set_applied=self._set_applied,
                save_applied=self._save,
                now=lambda: self.now,
                stderr=None,
            )
        )

    def _alias_set(self, session_id: str, name: str) -> str:
        self.alias_writes.append((session_id, name))
        cleaned = " ".join(name.split()).strip()
        if cleaned:
            self.aliases[session_id] = cleaned
        else:
            self.aliases.pop(session_id, None)
        return self.aliases.get(session_id, "")

    def _sock_call(self, sock: Any, req: dict[str, Any], **_kwargs: Any) -> dict[str, Any]:
        self.calls.append({"sock": sock, **req})
        if self.fail_next:
            raise self.fail_next.pop(0)
        if self.replies:
            return self.replies.pop(0)
        name = req.get("name", "")
        return {"ok": True, "backend": req.get("backend", "pi"), "name": name}

    def _set_applied(self, value: dict[str, dict[str, Any]]) -> None:
        self.applied = value

    def _save(self) -> None:
        self.saves += 1


def test_identity_key_is_backend_plus_transcript() -> None:
    assert identity_key("pi", "abc") == "pi:abc"
    assert identity_key("codex", "  abc  ") == "codex:abc"
    assert identity_key("", "abc") == ""
    assert identity_key("pi", "") == ""


def test_rename_is_projected_once_per_name() -> None:
    h = Harness()
    h.aliases["s1"] = "Alpha"

    first = h.projection.reconcile("s1")
    assert first["reason"] == "applied"
    assert h.calls == [{"sock": Path("/tmp/fake.sock"), "cmd": "name", "name": "Alpha", "thread_id": "thread-1"}]
    assert h.applied["pi:thread-1"] == {"state": "applied", "name": "Alpha", "attempts": 1, "ts": 1000.0, "control": True}

    second = h.projection.reconcile("s1")
    assert second["reason"] == "unchanged"
    assert len(h.calls) == 1  # a poll never re-appends the same title

    h.aliases["s1"] = "Beta"
    assert h.projection.reconcile("s1")["reason"] == "applied"
    assert [call["name"] for call in h.calls] == ["Alpha", "Beta"]

    h.aliases["s1"] = "Alpha"
    assert h.projection.reconcile("s1")["reason"] == "applied"
    assert [call["name"] for call in h.calls] == ["Alpha", "Beta", "Alpha"]


def test_removing_the_alias_projects_a_clear_and_stays_cleared() -> None:
    h = Harness()
    h.aliases["s1"] = "Alpha"
    h.projection.reconcile("s1")

    h.aliases.pop("s1")
    assert h.projection.reconcile("s1")["reason"] == "cleared"
    assert h.calls[-1]["name"] == ""
    assert h.applied["pi:thread-1"]["state"] == "cleared"

    # A stale backend title must not be re-adopted after the user cleared it.
    h.session.backend_session_name = "Alpha"
    assert h.projection.reconcile("s1")["reason"] == "cleared"
    assert h.alias_writes == []
    assert h.aliases == {}


def test_unsupported_is_retried_only_after_the_backend_advertises_control() -> None:
    h = Harness(control=False)
    h.aliases["s1"] = "Alpha"
    h.replies = [{"ok": False, "backend": "pi", "unsupported": True, "reason": "unsupported"}]
    assert h.projection.reconcile("s1")["reason"] == "unsupported"
    assert h.applied["pi:thread-1"]["state"] == "unsupported"

    # Unchanged capability: no repeated hammering.
    assert h.projection.reconcile("s1")["reason"] == "unsupported"
    assert len(h.calls) == 1

    # A bridge /reload flips the capability; the same name is now projected.
    h.session.session_name_control = True
    assert h.projection.reconcile("s1")["reason"] == "applied"
    assert len(h.calls) == 2


def test_transport_failure_backs_off_then_recovers() -> None:
    h = Harness()
    h.aliases["s1"] = "Alpha"
    h.fail_next = [OSError("broker down")]
    failed = h.projection.reconcile("s1")
    assert failed["ok"] is False
    assert failed["reason"] == "transport"
    assert h.applied["pi:thread-1"]["state"] == "failed"

    # Inside the first backoff window: no call.
    h.now += 5.0
    assert h.projection.reconcile("s1")["reason"] == "retry-later"
    assert len(h.calls) == 1

    # After it, the projection retries and lands.
    h.now += 20.0
    assert h.projection.reconcile("s1")["reason"] == "applied"
    assert len(h.calls) == 2


def test_pending_write_is_not_retried_until_the_name_changes() -> None:
    h = Harness(backend="cc")
    h.aliases["s1"] = "Alpha"
    h.replies = [{"ok": True, "backend": "cc", "pending": True}]
    assert h.projection.reconcile("s1")["reason"] == "pending"
    assert h.applied["cc:thread-1"]["state"] == "pending"
    assert h.projection.reconcile("s1")["reason"] == "unchanged"
    assert len(h.calls) == 1


def test_clear_the_backend_refuses_marks_the_identity_cleared() -> None:
    h = Harness(backend="codex")
    h.aliases["s1"] = "Alpha"
    h.projection.reconcile("s1")

    h.aliases.pop("s1")
    h.replies = [{"ok": False, "backend": "codex", "reason": "unsupported-clear"}]
    assert h.projection.reconcile("s1")["reason"] == "unsupported-clear"
    assert h.applied["codex:thread-1"]["state"] == "cleared"

    # The old codex title still exists; it is not adopted back.
    h.session.backend_session_name = "Alpha"
    assert h.projection.reconcile("s1")["reason"] == "cleared"
    assert h.aliases == {}


def test_a_terminal_set_title_is_adopted_when_no_alias_exists() -> None:
    h = Harness()
    h.session.backend_session_name = "terminal name"
    result = h.projection.reconcile("s1")
    assert result["reason"] == "adopted"
    assert h.aliases == {"s1": "terminal name"}
    assert h.applied["pi:thread-1"] == {
        "state": "adopted",
        "name": "terminal name",
        "attempts": 0,
        "ts": 1000.0,
        "control": True,
    }
    # Adoption is a local write, not a round trip to the backend.
    assert h.calls == []
    assert h.projection.reconcile("s1")["reason"] == "unchanged"


def test_adoption_waits_for_a_bound_transcript() -> None:
    h = Harness(thread_id="")
    h.session.backend_session_name = "terminal name"
    assert h.projection.reconcile("s1")["reason"] == "no-thread"
    assert h.aliases == {}
    assert h.calls == []


def test_a_new_transcript_identity_receives_the_name_again() -> None:
    h = Harness()
    h.aliases["s1"] = "Alpha"
    h.projection.reconcile("s1")
    assert len(h.calls) == 1

    # Pi /new: same registry session and alias, new transcript.
    h.session.thread_id = "thread-2"
    assert h.projection.reconcile("s1")["reason"] == "applied"
    assert h.calls[-1]["thread_id"] == "thread-2"
    # The previous transcript is no longer the session's identity; its record
    # survives the hourly grace window, then is pruned (a stale key would only
    # cost one redundant push).
    assert set(h.applied) == {"pi:thread-1", "pi:thread-2"}


def test_pruning_drops_identities_no_longer_present() -> None:
    h = Harness()
    h.aliases["s1"] = "Alpha"
    h.projection.reconcile("s1")
    assert "pi:thread-1" in h.applied

    h.sessions.clear()
    h.aliases.clear()
    # Reconcile of an unknown session raises; a store for a live session prunes
    # every identity that is no longer present.
    try:
        h.projection.reconcile("s1")
    except KeyError:
        pass
    h.sessions["s2"] = SimpleNamespace(
        agent_backend="pi",
        thread_id="thread-9",
        sock_path=Path("/tmp/fake.sock"),
        log_path=None,
        session_name_control=True,
        backend_session_name="",
    )
    h.aliases["s2"] = "Gamma"
    assert h.projection.reconcile("s2")["reason"] == "applied"
    # A just-written record survives a transient unbound state; an hour-old
    # orphan is dropped.
    assert set(h.applied) == {"pi:thread-1", "pi:thread-9"}
    h.now += 3700.0
    h.aliases["s2"] = "Delta"
    assert h.projection.reconcile("s2")["reason"] == "applied"
    assert set(h.applied) == {"pi:thread-9"}
