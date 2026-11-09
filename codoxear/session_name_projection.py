from __future__ import annotations

"""Project the codoxear session name onto each backend's own session title.

The alias store is the single authority for a session's name. Every backend also
keeps a title of its own (Pi's ``session_info`` row, Codex's thread name,
Claude Code's ``custom-title``); this module makes that title a projection of the
alias and nothing more.

Rules, in order:

1. A non-empty alias is written to the backend — but only once per
   ``(backend, transcript identity, name)`` key, so a broker restart or a
   session-list poll never re-appends the same title.
2. Removing the alias projects a clear where the backend supports one (Pi). A
   backend with no clear path (Codex refuses empty titles, Claude Code has no
   programmatic clear) keeps its last title, and that identity is marked
   ``cleared`` so the title is not silently re-adopted as if the user had set it.
3. With no alias and no user clear, a title the user set in the terminal is
   adopted into the alias store — once. Adoption never fights a user clear.
4. A backend that cannot be reached is retried on a bounded backoff. It never
   blocks the rename: the alias is already saved and the failure is reported.
"""

from dataclasses import dataclass
from typing import Any, Callable, MutableMapping

from .agent_backend import normalize_agent_backend
from .session_cleaners import clean_alias


# Retry backoff for a transport failure, by attempt count. Bounded so a broker
# that comes back later still receives the name, without polling it every tick.
_RETRY_DELAYS = (15.0, 60.0, 300.0)


@dataclass(frozen=True)
class SessionNameProjectionDeps:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Any]]
    alias_get: Callable[[str], str]
    alias_cleared: Callable[[str], bool]
    alias_set: Callable[[str, str], str]
    sock_call: Callable[..., dict[str, Any]]
    applied: Callable[[], MutableMapping[str, dict[str, Any]]]
    set_applied: Callable[[dict[str, dict[str, Any]]], None]
    save_applied: Callable[[], None]
    now: Callable[[], float]
    stderr: Any


def identity_key(agent_backend: Any, thread_id: Any) -> str:
    raw_backend = str(agent_backend or "").strip().lower()
    thread = str(thread_id or "").strip()
    if not raw_backend or not thread:
        return ""
    try:
        backend = normalize_agent_backend(raw_backend)
    except ValueError:
        return ""
    return f"{backend}:{thread}"


class SessionNameProjection:
    def __init__(self, deps: SessionNameProjectionDeps) -> None:
        self.deps = deps
        self._inflight: set[str] = set()

    # -- state helpers -------------------------------------------------------

    def _session_snapshot(self, session_id: str) -> dict[str, Any] | None:
        with self.deps.lock:
            session = self.deps.sessions().get(session_id)
            if session is None:
                return None
            return {
                "backend": normalize_agent_backend(getattr(session, "agent_backend", None), default=""),
                "thread_id": str(getattr(session, "thread_id", "") or "").strip(),
                "log_path": getattr(session, "log_path", None),
                "sock_path": getattr(session, "sock_path", None),
                "control": bool(getattr(session, "session_name_control", False)),
                "observed": str(getattr(session, "backend_session_name", "") or "").strip(),
            }

    def _record(self, key: str) -> dict[str, Any]:
        with self.deps.lock:
            value = self.deps.applied().get(key)
            return dict(value) if isinstance(value, dict) else {}

    def _store(self, key: str, record: dict[str, Any]) -> None:
        with self.deps.lock:
            applied = dict(self.deps.applied())
            if record is None:
                applied.pop(key, None)
            else:
                applied[key] = record
            applied = self._prune_locked(applied)
            self.deps.set_applied(applied)
        self.deps.save_applied()

    def _prune_locked(self, applied: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        live: set[str] = set()
        for session in self.deps.sessions().values():
            key = identity_key(getattr(session, "agent_backend", None), getattr(session, "thread_id", None))
            if key:
                live.add(key)
        if not live:
            return applied
        # A session can be momentarily unbound (Pi /new rebinds, a sidecar is
        # rewritten); keeping its record briefly avoids one redundant re-push,
        # while an hour-old orphan is genuinely gone.
        now = self.deps.now()
        return {
            key: value
            for key, value in applied.items()
            if key in live or (now - float(value.get("ts") or 0.0)) < 3600.0
        }

    def _begin(self, key: str) -> bool:
        with self.deps.lock:
            if key in self._inflight:
                return False
            self._inflight.add(key)
            return True

    def _end(self, key: str) -> None:
        with self.deps.lock:
            self._inflight.discard(key)

    # -- the projection ------------------------------------------------------

    def reconcile(self, session_id: str) -> dict[str, Any]:
        """Bring the backend's title in line with the alias store."""
        snapshot = self._session_snapshot(session_id)
        if snapshot is None:
            raise KeyError("unknown session")
        backend = snapshot["backend"]
        key = identity_key(backend, snapshot["thread_id"])
        if not key:
            return {"ok": True, "backend": backend, "reason": "no-thread"}
        desired = clean_alias(self.deps.alias_get(session_id))

        if not self._begin(key):
            return {"ok": True, "backend": backend, "reason": "inflight"}
        try:
            return self._reconcile_locked_identity(session_id, snapshot, key, desired)
        finally:
            self._end(key)

    def _reconcile_locked_identity(
        self,
        session_id: str,
        snapshot: dict[str, Any],
        key: str,
        desired: str,
    ) -> dict[str, Any]:
        backend = snapshot["backend"]
        record = self._record(key)
        state = record.get("state") if isinstance(record.get("state"), str) else ""
        record_name = record.get("name") if isinstance(record.get("name"), str) else ""

        if desired:
            if record_name == desired and state in ("applied", "adopted", "pending"):
                return {"ok": True, "backend": backend, "reason": "unchanged", "name": desired}
            if record_name == desired and state == "unsupported":
                # Only a capability change since the failure re-arms the
                # projection; otherwise this would retry on every poll.
                previous_control = record.get("control") is True
                if not snapshot["control"] or previous_control == snapshot["control"]:
                    return {"ok": False, "backend": backend, "reason": "unsupported", "name": desired}
            if record_name == desired and state == "failed":
                delay = _retry_delay(record)
                if (self.deps.now() - float(record.get("ts") or 0.0)) < delay:
                    return {"ok": False, "backend": backend, "reason": "retry-later", "name": desired}
            return self._push(session_id, snapshot, key, desired, record)

        # No name for this session. That is not the same as "the user removed it":
        # the alias store can be empty because it was never adopted, or because a
        # save lost it. Only an explicit clear (an entry holding "") may erase a
        # backend title — a wrong clear destroys a name the user set in the
        # terminal, and the recorded `cleared` state then blocks re-adoption.
        if not self.deps.alias_cleared(session_id):
            observed = snapshot["observed"]
            if observed and record_name != observed:
                return self._adopt(session_id, backend, key, observed)
            return {"ok": True, "backend": backend, "reason": "no-alias"}
        if state == "cleared":
            return {"ok": True, "backend": backend, "reason": "cleared"}
        if record_name or state == "failed":
            if state == "failed":
                delay = _retry_delay(record)
                if (self.deps.now() - float(record.get("ts") or 0.0)) < delay:
                    return {"ok": False, "backend": backend, "reason": "retry-later"}
            # The user removed the name: project the clear once.
            return self._push(session_id, snapshot, key, "", record)
        observed = snapshot["observed"]
        if observed and record_name != observed:
            return self._adopt(session_id, backend, key, observed)
        return {"ok": True, "backend": backend, "reason": "unchanged"}

    def _push(
        self,
        session_id: str,
        snapshot: dict[str, Any],
        key: str,
        name: str,
        previous: dict[str, Any],
    ) -> dict[str, Any]:
        backend = snapshot["backend"]
        sock = snapshot.get("sock_path")
        if sock is None:
            return self._record_failure(key, name, previous, {"reason": "transport", "detail": "no control socket"}, backend)
        try:
            reply = self.deps.sock_call(
                sock,
                {"cmd": "name", "name": name, "thread_id": snapshot["thread_id"]},
                timeout_s=2.0,
            )
        except Exception as exc:
            return self._record_failure(key, name, previous, {"reason": "transport", "detail": f"{type(exc).__name__}: {exc}"}, backend)
        if not isinstance(reply, dict):
            return self._record_failure(key, name, previous, {"reason": "transport", "detail": "invalid broker reply"}, backend)
        reason = reply.get("reason") if isinstance(reply.get("reason"), str) else ""

        if reply.get("ok") is True:
            if reply.get("pending") is True:
                self._store(key, {"state": "pending", "name": name, "attempts": 0, "ts": self.deps.now(), "control": snapshot["control"]})
                return {"ok": True, "backend": backend, "reason": "pending", "name": name}
            effective = reply.get("name") if isinstance(reply.get("name"), str) else name
            if not name:
                # A projected clear: the alias stays empty and the title is gone
                # (or the backend cannot clear, in which case the identity stays
                # cleared so an old title is never re-adopted).
                self._store(key, {"state": "cleared", "name": "", "attempts": 0, "ts": self.deps.now(), "control": snapshot["control"]})
                return {"ok": True, "backend": backend, "reason": "cleared"}
            self._store(key, {"state": "applied", "name": effective, "attempts": 1, "ts": self.deps.now(), "control": snapshot["control"]})
            return {"ok": True, "backend": backend, "reason": "applied", "name": effective}

        if reply.get("unsupported") is True or reason in ("unsupported", "unsupported-clear"):
            # Terminal for this key until the capability or the name changes:
            # an old Pi bridge that later reloads advertises the capability and
            # the next poll retries automatically.
            state_value = "cleared" if not name else "unsupported"
            self._store(key, {"state": state_value, "name": name, "attempts": 0, "ts": self.deps.now(), "control": snapshot["control"]})
            return {"ok": False, "backend": backend, "reason": reason or "unsupported", "name": name}

        return self._record_failure(key, name, previous, {"reason": reason or "error", "detail": str(reply.get("error") or reply.get("detail") or "")}, backend)

    def _record_failure(
        self,
        key: str,
        name: str,
        previous: dict[str, Any],
        failure: dict[str, str],
        backend: str,
    ) -> dict[str, Any]:
        attempts = previous.get("attempts") if isinstance(previous.get("attempts"), int) else 0
        attempts = int(attempts) + 1
        control = previous.get("control") is True
        self._store(key, {"state": "failed", "name": name, "attempts": attempts, "ts": self.deps.now(), "control": control})
        return {
            "ok": False,
            "backend": backend,
            "reason": failure.get("reason") or "error",
            "name": name,
            "detail": failure.get("detail") or "",
        }

    def _adopt(self, session_id: str, backend: str, key: str, observed: str) -> dict[str, Any]:
        """Take a terminal-set title as the display name when none exists."""
        try:
            alias = self.deps.alias_set(session_id, observed)
        except KeyError:
            return {"ok": True, "backend": backend, "reason": "unknown-session"}
        if not alias:
            return {"ok": True, "backend": backend, "reason": "unchanged"}
        self._store(key, {"state": "adopted", "name": alias, "attempts": 0, "ts": self.deps.now(), "control": True})
        return {"ok": True, "backend": backend, "reason": "adopted", "name": alias}


def _retry_delay(record: dict[str, Any]) -> float:
    attempts = record.get("attempts")
    index = int(attempts) - 1 if isinstance(attempts, int) and attempts > 0 else 0
    index = min(max(index, 0), len(_RETRY_DELAYS) - 1)
    return _RETRY_DELAYS[index]
