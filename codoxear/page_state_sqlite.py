from __future__ import annotations

import functools
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SessionRef = tuple[str, str]


def _db_locked(fn):
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return fn(self, *args, **kwargs)

    return wrapped


@dataclass(frozen=True)
class DurableSessionRecord:
    backend: str
    session_id: str
    cwd: str | None = None
    source_path: str | None = None
    title: str | None = None
    first_user_message: str | None = None
    created_at: float | None = None
    updated_at: float | None = None
    pending_startup: bool = False


@dataclass(frozen=True)
class LegacyImportReport:
    imported_aliases: int = 0
    imported_sidebar_meta: int = 0
    imported_files: int = 0
    imported_queues: int = 0
    imported_recent_cwds: int = 0
    imported_cwd_groups: int = 0
    imported_hidden_keys: int = 0
    imported_voice_settings: int = 0
    imported_push_subscriptions: int = 0
    imported_delivery_ledger: int = 0
    unmapped_rows: int = 0


def _normalize_backend(value: Any) -> str:
    raw = str(value or "").strip().lower()
    return "pi" if raw == "pi" else "codex"


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _legacy_runtime_lookup(
    runtime_map: dict[str, SessionRef], legacy_key: Any
) -> SessionRef | None:
    key = _clean_text(legacy_key)
    if key is None:
        return None
    ref = runtime_map.get(key)
    if ref is not None:
        return ref
    if key.startswith("sid:"):
        return runtime_map.get(key[4:])
    return None


class PageStateDB:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        self._migrate()

    @_db_locked
    def close(self) -> None:
        self._conn.close()

    @_db_locked
    def is_empty(self) -> bool:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM session_ui_state"
        ).fetchone()
        if row is not None and int(row["n"] or 0) > 0:
            return False
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM session_files"
        ).fetchone()
        if row is not None and int(row["n"] or 0) > 0:
            return False
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM session_queue_items"
        ).fetchone()
        if row is not None and int(row["n"] or 0) > 0:
            return False
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM recent_cwds"
        ).fetchone()
        if row is not None and int(row["n"] or 0) > 0:
            return False
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM cwd_groups"
        ).fetchone()
        if row is not None and int(row["n"] or 0) > 0:
            return False
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM app_kv"
        ).fetchone()
        if row is not None and int(row["n"] or 0) > 0:
            return False
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM push_subscriptions"
        ).fetchone()
        if row is not None and int(row["n"] or 0) > 0:
            return False
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM delivery_ledger"
        ).fetchone()
        return not (row is not None and int(row["n"] or 0) > 0)

    @_db_locked
    def _migrate(self) -> None:
        with self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                  backend TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  cwd TEXT,
                  source_path TEXT,
                  title TEXT,
                  first_user_message TEXT,
                  created_at REAL,
                  updated_at REAL,
                  pending_startup INTEGER NOT NULL DEFAULT 0,
                  PRIMARY KEY (backend, session_id)
                );

                CREATE TABLE IF NOT EXISTS session_ui_state (
                  backend TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  alias TEXT,
                  focused INTEGER NOT NULL DEFAULT 0,
                  hidden INTEGER NOT NULL DEFAULT 0,
                  priority_offset REAL NOT NULL DEFAULT 0,
                  snooze_until REAL,
                  dependency_backend TEXT,
                  dependency_session_id TEXT,
                  PRIMARY KEY (backend, session_id)
                );

                CREATE TABLE IF NOT EXISTS hidden_session_keys (
                  key TEXT PRIMARY KEY
                );

                CREATE TABLE IF NOT EXISTS session_files (
                  backend TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  path TEXT NOT NULL,
                  ordinal INTEGER NOT NULL,
                  last_used_ts REAL,
                  PRIMARY KEY (backend, session_id, path)
                );

                CREATE TABLE IF NOT EXISTS session_queue_items (
                  backend TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  ordinal INTEGER NOT NULL,
                  text TEXT NOT NULL,
                  PRIMARY KEY (backend, session_id, ordinal)
                );

                CREATE TABLE IF NOT EXISTS recent_cwds (
                  cwd TEXT PRIMARY KEY,
                  last_used_ts REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cwd_groups (
                  cwd TEXT PRIMARY KEY,
                  label TEXT,
                  collapsed INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS app_kv (
                  namespace TEXT NOT NULL,
                  key TEXT NOT NULL,
                  value_json TEXT NOT NULL,
                  PRIMARY KEY (namespace, key)
                );

                CREATE TABLE IF NOT EXISTS push_subscriptions (
                  id TEXT PRIMARY KEY,
                  payload_json TEXT NOT NULL,
                  updated_ts REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS delivery_ledger (
                  message_id TEXT PRIMARY KEY,
                  payload_json TEXT NOT NULL,
                  updated_ts REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS legacy_import_unmapped (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  source_name TEXT NOT NULL,
                  legacy_key TEXT NOT NULL,
                  payload_json TEXT NOT NULL,
                  imported_at REAL NOT NULL
                );
                """
            )
            cols = {
                str(row[1])
                for row in self._conn.execute("PRAGMA table_info(sessions)")
            }
            if "pending_startup" not in cols:
                self._conn.execute(
                    "ALTER TABLE sessions ADD COLUMN pending_startup INTEGER NOT NULL DEFAULT 0"
                )

    @_db_locked
    def load_sessions(self) -> dict[SessionRef, DurableSessionRecord]:
        out: dict[SessionRef, DurableSessionRecord] = {}
        for row in self._conn.execute(
            "SELECT backend, session_id, cwd, source_path, title, first_user_message, created_at, updated_at, pending_startup FROM sessions"
        ):
            ref = (_normalize_backend(row["backend"]), str(row["session_id"]))
            out[ref] = DurableSessionRecord(
                backend=ref[0],
                session_id=ref[1],
                cwd=_clean_text(row["cwd"]),
                source_path=_clean_text(row["source_path"]),
                title=_clean_text(row["title"]),
                first_user_message=_clean_text(row["first_user_message"]),
                created_at=(float(row["created_at"]) if row["created_at"] is not None else None),
                updated_at=(float(row["updated_at"]) if row["updated_at"] is not None else None),
                pending_startup=bool(row["pending_startup"] or 0),
            )
        return out

    @_db_locked
    def save_sessions(self, rows: dict[SessionRef, DurableSessionRecord]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM sessions")
            for (backend, session_id), row in sorted(rows.items()):
                self._conn.execute(
                    """
                    INSERT INTO sessions (
                      backend, session_id, cwd, source_path, title,
                      first_user_message, created_at, updated_at, pending_startup
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        backend,
                        session_id,
                        row.cwd,
                        row.source_path,
                        row.title,
                        row.first_user_message,
                        row.created_at,
                        row.updated_at,
                        1 if row.pending_startup else 0,
                    ),
                )

    @_db_locked
    def upsert_session(self, row: DurableSessionRecord) -> None:
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO sessions (
                  backend, session_id, cwd, source_path, title,
                  first_user_message, created_at, updated_at, pending_startup
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(backend, session_id) DO UPDATE SET
                  cwd=excluded.cwd,
                  source_path=excluded.source_path,
                  title=excluded.title,
                  first_user_message=excluded.first_user_message,
                  created_at=excluded.created_at,
                  updated_at=excluded.updated_at,
                  pending_startup=excluded.pending_startup
                """,
                (
                    row.backend,
                    row.session_id,
                    row.cwd,
                    row.source_path,
                    row.title,
                    row.first_user_message,
                    row.created_at,
                    row.updated_at,
                    1 if row.pending_startup else 0,
                ),
            )

    @_db_locked
    def delete_session(self, ref: SessionRef) -> None:
        with self._conn:
            self._conn.execute(
                "DELETE FROM sessions WHERE backend = ? AND session_id = ?",
                (ref[0], ref[1]),
            )

    @_db_locked
    def known_session_refs(self) -> set[SessionRef]:
        refs: set[SessionRef] = set()
        queries = [
            "SELECT backend, session_id FROM sessions",
            "SELECT backend, session_id FROM session_ui_state",
            "SELECT backend, session_id FROM session_files",
            "SELECT backend, session_id FROM session_queue_items",
        ]
        for query in queries:
            for row in self._conn.execute(query):
                refs.add((_normalize_backend(row["backend"]), str(row["session_id"])))
        return refs

    @_db_locked
    def load_session_ui_state(self) -> tuple[dict[SessionRef, str], dict[SessionRef, dict[str, Any]], set[str]]:
        aliases: dict[SessionRef, str] = {}
        sidebar_meta: dict[SessionRef, dict[str, Any]] = {}
        hidden_keys: set[str] = set()
        for row in self._conn.execute(
            "SELECT backend, session_id, alias, focused, hidden, priority_offset, snooze_until, dependency_backend, dependency_session_id FROM session_ui_state"
        ):
            ref = (_normalize_backend(row["backend"]), str(row["session_id"]))
            alias = _clean_text(row["alias"])
            if alias is not None:
                aliases[ref] = alias
            entry: dict[str, Any] = {
                "priority_offset": float(row["priority_offset"] or 0.0),
            }
            if row["snooze_until"] is not None:
                entry["snooze_until"] = float(row["snooze_until"])
            dependency_id = _clean_text(row["dependency_session_id"])
            if dependency_id is not None:
                entry["dependency_session_id"] = dependency_id
            if int(row["focused"] or 0):
                entry["focused"] = True
            if int(row["hidden"] or 0):
                entry["hidden"] = True
            sidebar_meta[ref] = entry
        for row in self._conn.execute("SELECT key FROM hidden_session_keys"):
            key = _clean_text(row["key"])
            if key is not None:
                hidden_keys.add(key)
        return aliases, sidebar_meta, hidden_keys

    @_db_locked
    def save_session_ui_state(
        self,
        aliases: dict[SessionRef, str],
        sidebar_meta: dict[SessionRef, dict[str, Any]],
        hidden_keys: set[str],
    ) -> None:
        refs = set(aliases.keys()) | set(sidebar_meta.keys())
        with self._conn:
            self._conn.execute("DELETE FROM session_ui_state")
            self._conn.execute("DELETE FROM hidden_session_keys")
            for backend, session_id in sorted(refs):
                entry = dict(sidebar_meta.get((backend, session_id), {}))
                self._conn.execute(
                    """
                    INSERT INTO session_ui_state (
                      backend, session_id, alias, focused, hidden,
                      priority_offset, snooze_until, dependency_backend, dependency_session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        backend,
                        session_id,
                        aliases.get((backend, session_id)),
                        1 if entry.get("focused") else 0,
                        1 if entry.get("hidden") else 0,
                        float(entry.get("priority_offset") or 0.0),
                        entry.get("snooze_until"),
                        backend if _clean_text(entry.get("dependency_session_id")) else None,
                        _clean_text(entry.get("dependency_session_id")),
                    ),
                )
            for key in sorted(hidden_keys):
                self._conn.execute(
                    "INSERT INTO hidden_session_keys (key) VALUES (?)",
                    (key,),
                )

    @_db_locked
    def load_files(self) -> dict[SessionRef, list[str]]:
        out: dict[SessionRef, list[tuple[int, str]]] = {}
        for row in self._conn.execute(
            "SELECT backend, session_id, path, ordinal FROM session_files ORDER BY backend, session_id, ordinal ASC"
        ):
            ref = (_normalize_backend(row["backend"]), str(row["session_id"]))
            out.setdefault(ref, []).append((int(row["ordinal"]), str(row["path"])))
        return {ref: [path for _ordinal, path in rows] for ref, rows in out.items()}

    @_db_locked
    def save_files(self, files: dict[SessionRef, list[str]]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM session_files")
            for (backend, session_id), rows in sorted(files.items()):
                for ordinal, path in enumerate(rows):
                    self._conn.execute(
                        "INSERT INTO session_files (backend, session_id, path, ordinal, last_used_ts) VALUES (?, ?, ?, ?, ?)",
                        (backend, session_id, path, ordinal, time.time()),
                    )

    @_db_locked
    def load_queues(self) -> dict[SessionRef, list[str]]:
        out: dict[SessionRef, list[tuple[int, str]]] = {}
        for row in self._conn.execute(
            "SELECT backend, session_id, ordinal, text FROM session_queue_items ORDER BY backend, session_id, ordinal ASC"
        ):
            ref = (_normalize_backend(row["backend"]), str(row["session_id"]))
            out.setdefault(ref, []).append((int(row["ordinal"]), str(row["text"])))
        return {ref: [text for _ordinal, text in rows] for ref, rows in out.items()}

    @_db_locked
    def save_queues(self, queues: dict[SessionRef, list[str]]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM session_queue_items")
            for (backend, session_id), rows in sorted(queues.items()):
                for ordinal, text in enumerate(rows):
                    self._conn.execute(
                        "INSERT INTO session_queue_items (backend, session_id, ordinal, text) VALUES (?, ?, ?, ?)",
                        (backend, session_id, ordinal, text),
                    )

    @_db_locked
    def load_recent_cwds(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in self._conn.execute("SELECT cwd, last_used_ts FROM recent_cwds"):
            out[str(row["cwd"])] = float(row["last_used_ts"])
        return out

    @_db_locked
    def save_recent_cwds(self, recent_cwds: dict[str, float]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM recent_cwds")
            for cwd, ts in sorted(recent_cwds.items()):
                self._conn.execute(
                    "INSERT INTO recent_cwds (cwd, last_used_ts) VALUES (?, ?)",
                    (cwd, float(ts)),
                )

    @_db_locked
    def load_cwd_groups(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in self._conn.execute("SELECT cwd, label, collapsed FROM cwd_groups"):
            entry = {
                "label": _clean_text(row["label"]) or "",
                "collapsed": bool(int(row["collapsed"] or 0)),
            }
            out[str(row["cwd"])] = entry
        return out

    @_db_locked
    def save_cwd_groups(self, cwd_groups: dict[str, dict[str, Any]]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM cwd_groups")
            for cwd, entry in sorted(cwd_groups.items()):
                self._conn.execute(
                    "INSERT INTO cwd_groups (cwd, label, collapsed) VALUES (?, ?, ?)",
                    (cwd, _clean_text(entry.get("label")), 1 if entry.get("collapsed") else 0),
                )

    @_db_locked
    def load_app_kv(self, namespace: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for row in self._conn.execute(
            "SELECT key, value_json FROM app_kv WHERE namespace = ?",
            (namespace,),
        ):
            try:
                out[str(row["key"])] = json.loads(str(row["value_json"]))
            except json.JSONDecodeError:
                continue
        return out

    @_db_locked
    def save_app_kv(self, namespace: str, values: dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM app_kv WHERE namespace = ?", (namespace,))
            for key, value in sorted(values.items()):
                self._conn.execute(
                    "INSERT INTO app_kv (namespace, key, value_json) VALUES (?, ?, ?)",
                    (namespace, key, json.dumps(value, ensure_ascii=False, sort_keys=True)),
                )

    @_db_locked
    def load_push_subscriptions(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in self._conn.execute(
            "SELECT payload_json FROM push_subscriptions ORDER BY updated_ts ASC, id ASC"
        ):
            try:
                payload = json.loads(str(row["payload_json"]))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                out.append(payload)
        return out

    @_db_locked
    def load_delivery_ledger(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in self._conn.execute(
            "SELECT message_id, payload_json FROM delivery_ledger ORDER BY updated_ts ASC, message_id ASC"
        ):
            try:
                payload = json.loads(str(row["payload_json"]))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                out[str(row["message_id"])] = payload
        return out

    @_db_locked
    def save_push_subscriptions(self, rows: list[dict[str, Any]]) -> None:
        self._save_push_subscriptions(rows)

    @_db_locked
    def save_delivery_ledger(self, rows: dict[str, dict[str, Any]]) -> None:
        self._save_delivery_ledger(rows)

    @_db_locked
    def import_legacy_app_dir(self, app_dir: Path) -> LegacyImportReport:
        source = Path(app_dir)
        runtime_map = self._legacy_runtime_map(source / "socks")
        report = LegacyImportReport()

        aliases = self._read_json(source / "session_aliases.json", default={})
        sidebar = self._read_json(source / "session_sidebar.json", default={})
        hidden = self._read_json(source / "hidden_sessions.json", default=[])
        files = self._read_json(source / "session_files.json", default={})
        queues = self._read_json(source / "session_queues.json", default={})
        recent_cwds = self._read_json(source / "recent_cwds.json", default={})
        cwd_groups = self._read_json(source / "cwd_groups.json", default={})
        voice_settings = self._read_json(source / "voice_settings.json", default={})
        push_subscriptions = self._read_json(source / "push_subscriptions.json", default=[])
        delivery_ledger = self._read_json(source / "voice_delivery_ledger.json", default={})

        alias_rows: dict[SessionRef, str] = {}
        sidebar_rows: dict[SessionRef, dict[str, Any]] = {}
        file_rows: dict[SessionRef, list[str]] = {}
        queue_rows: dict[SessionRef, list[str]] = {}
        hidden_keys: set[str] = set()
        now = time.time()
        unmapped_rows = 0

        if isinstance(aliases, dict):
            for legacy_key, alias in aliases.items():
                ref = _legacy_runtime_lookup(runtime_map, legacy_key)
                if ref is None:
                    self._record_unmapped("session_aliases.json", str(legacy_key), alias, now)
                    unmapped_rows += 1
                    continue
                cleaned = _clean_text(alias)
                if cleaned is not None:
                    alias_rows[ref] = cleaned
                    report = LegacyImportReport(**{**report.__dict__, "imported_aliases": report.imported_aliases + 1})

        if isinstance(sidebar, dict):
            for legacy_key, value in sidebar.items():
                ref = _legacy_runtime_lookup(runtime_map, legacy_key)
                if ref is None:
                    self._record_unmapped("session_sidebar.json", str(legacy_key), value, now)
                    unmapped_rows += 1
                    continue
                if not isinstance(value, dict):
                    continue
                entry: dict[str, Any] = {"priority_offset": float(value.get("priority_offset") or 0.0)}
                if value.get("snooze_until") is not None:
                    entry["snooze_until"] = value.get("snooze_until")
                dep = _clean_text(value.get("dependency_session_id"))
                if dep is not None:
                    entry["dependency_session_id"] = dep
                sidebar_rows[ref] = entry
                report = LegacyImportReport(**{**report.__dict__, "imported_sidebar_meta": report.imported_sidebar_meta + 1})

        if isinstance(files, dict):
            for legacy_key, rows in files.items():
                ref = _legacy_runtime_lookup(runtime_map, legacy_key)
                if ref is None:
                    self._record_unmapped("session_files.json", str(legacy_key), rows, now)
                    unmapped_rows += 1
                    continue
                if not isinstance(rows, list):
                    continue
                file_rows[ref] = [str(row) for row in rows if isinstance(row, str) and row.strip()]
                report = LegacyImportReport(**{**report.__dict__, "imported_files": report.imported_files + len(file_rows[ref])})

        if isinstance(queues, dict):
            for legacy_key, rows in queues.items():
                ref = _legacy_runtime_lookup(runtime_map, legacy_key)
                if ref is None:
                    self._record_unmapped("session_queues.json", str(legacy_key), rows, now)
                    unmapped_rows += 1
                    continue
                if not isinstance(rows, list):
                    continue
                queue_rows[ref] = [str(row) for row in rows if isinstance(row, str) and row.strip()]
                report = LegacyImportReport(**{**report.__dict__, "imported_queues": report.imported_queues + len(queue_rows[ref])})

        if isinstance(hidden, list):
            hidden_keys = {str(row).strip() for row in hidden if isinstance(row, str) and row.strip()}
            report = LegacyImportReport(**{**report.__dict__, "imported_hidden_keys": len(hidden_keys)})

        cleaned_recent: dict[str, float] = {}
        if isinstance(recent_cwds, dict):
            for cwd, ts in recent_cwds.items():
                try:
                    cleaned_recent[str(cwd)] = float(ts)
                except (TypeError, ValueError):
                    continue
            report = LegacyImportReport(**{**report.__dict__, "imported_recent_cwds": len(cleaned_recent)})

        cleaned_groups: dict[str, dict[str, Any]] = {}
        if isinstance(cwd_groups, dict):
            for cwd, entry in cwd_groups.items():
                if not isinstance(entry, dict):
                    continue
                cleaned_groups[str(cwd)] = {
                    "label": _clean_text(entry.get("label")) or "",
                    "collapsed": bool(entry.get("collapsed")),
                }
            report = LegacyImportReport(**{**report.__dict__, "imported_cwd_groups": len(cleaned_groups)})

        self.save_session_ui_state(alias_rows, sidebar_rows, hidden_keys)
        self.save_files(file_rows)
        self.save_queues(queue_rows)
        self.save_recent_cwds(cleaned_recent)
        self.save_cwd_groups(cleaned_groups)

        if isinstance(voice_settings, dict):
            self.save_app_kv("voice_settings", voice_settings)
            report = LegacyImportReport(**{**report.__dict__, "imported_voice_settings": len(voice_settings)})

        if isinstance(push_subscriptions, list):
            self._save_push_subscriptions(push_subscriptions)
            report = LegacyImportReport(**{**report.__dict__, "imported_push_subscriptions": len(push_subscriptions)})

        if isinstance(delivery_ledger, dict):
            self._save_delivery_ledger(delivery_ledger)
            report = LegacyImportReport(**{**report.__dict__, "imported_delivery_ledger": len(delivery_ledger)})

        return LegacyImportReport(**{**report.__dict__, "unmapped_rows": unmapped_rows})

    @_db_locked
    def _save_push_subscriptions(self, rows: list[Any]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM push_subscriptions")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                row_id = _clean_text(row.get("id"))
                if row_id is None:
                    continue
                updated_ts = row.get("updated_ts")
                try:
                    ts = float(updated_ts) if updated_ts is not None else time.time()
                except (TypeError, ValueError):
                    ts = time.time()
                self._conn.execute(
                    "INSERT INTO push_subscriptions (id, payload_json, updated_ts) VALUES (?, ?, ?)",
                    (row_id, json.dumps(row, ensure_ascii=False, sort_keys=True), ts),
                )

    @_db_locked
    def _save_delivery_ledger(self, rows: dict[str, Any]) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM delivery_ledger")
            for message_id, row in rows.items():
                if not isinstance(message_id, str):
                    continue
                if not isinstance(row, dict):
                    continue
                updated = row.get("updated_ts") or row.get("created_ts") or time.time()
                try:
                    ts = float(updated)
                except (TypeError, ValueError):
                    ts = time.time()
                self._conn.execute(
                    "INSERT INTO delivery_ledger (message_id, payload_json, updated_ts) VALUES (?, ?, ?)",
                    (message_id, json.dumps(row, ensure_ascii=False, sort_keys=True), ts),
                )

    @_db_locked
    def _record_unmapped(self, source_name: str, legacy_key: str, payload: Any, imported_at: float) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO legacy_import_unmapped (source_name, legacy_key, payload_json, imported_at) VALUES (?, ?, ?, ?)",
                (source_name, legacy_key, json.dumps(payload, ensure_ascii=False, sort_keys=True), imported_at),
            )

    @_db_locked
    def _read_json(self, path: Path, *, default: Any) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return default

    @_db_locked
    def _legacy_runtime_map(self, socks_dir: Path) -> dict[str, SessionRef]:
        out: dict[str, SessionRef] = {}
        for meta_path in sorted(socks_dir.glob("*.json")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(meta, dict):
                continue
            runtime_id = meta_path.stem
            durable_id = _clean_text(meta.get("session_id"))
            if durable_id is None:
                continue
            backend = _normalize_backend(meta.get("agent_backend") or meta.get("backend"))
            out[runtime_id] = (backend, durable_id)
        return out


def import_legacy_app_dir_to_db(*, source_app_dir: Path, db_path: Path) -> LegacyImportReport:
    db = PageStateDB(db_path)
    try:
        return db.import_legacy_app_dir(source_app_dir)
    finally:
        db.close()
