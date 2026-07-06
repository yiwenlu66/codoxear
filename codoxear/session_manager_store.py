from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from .session_store import SessionStore
from .session_store import SessionStorePaths


def session_store_paths(
    *,
    aliases: Path,
    sidebar_meta: Path,
    hidden_sessions: Path,
    files: Path,
    queues: Path,
    pending_attachments: Path,
    commit_unknown_sends: Path,
    recent_cwds: Path,
    unattended: Path,
    staged_attachments: Path | None = None,
    uploads_root: Path | None = None,
) -> SessionStorePaths:
    return SessionStorePaths(
        aliases=aliases,
        sidebar_meta=sidebar_meta,
        hidden_sessions=hidden_sessions,
        files=files,
        queues=queues,
        pending_attachments=pending_attachments,
        commit_unknown_sends=commit_unknown_sends,
        recent_cwds=recent_cwds,
        unattended=unattended,
        staged_attachments=staged_attachments,
        uploads_root=uploads_root,
    )


def create_session_store(
    *,
    paths: SessionStorePaths,
    file_history_max: int,
    recent_cwd_max: int,
    unattended_default_idle_minutes: int,
    unattended_default_max_injections: int,
    clean_alias: Callable[[Any], Any],
    clean_priority_offset: Callable[[Any], Any],
    clean_snooze_until: Callable[[Any], Any],
    clean_dependency_session_id: Callable[[Any], Any],
    clean_recent_cwd: Callable[[Any], Any],
    clean_commit_unknown_send_record: Callable[[Any], dict[str, Any] | None],
) -> SessionStore:
    return SessionStore(
        paths=paths,
        file_history_max=file_history_max,
        recent_cwd_max=recent_cwd_max,
        unattended_default_idle_minutes=unattended_default_idle_minutes,
        unattended_default_max_injections=unattended_default_max_injections,
        clean_alias=clean_alias,
        clean_priority_offset=clean_priority_offset,
        clean_snooze_until=clean_snooze_until,
        clean_dependency_session_id=clean_dependency_session_id,
        clean_recent_cwd=clean_recent_cwd,
        clean_commit_unknown_send_record=clean_commit_unknown_send_record,
    )


def copy_session_store_state(*, source: SessionStore, target: SessionStore) -> None:
    target.unattended = source.unattended
    target.aliases = source.aliases
    target.sidebar_meta = source.sidebar_meta
    target.hidden_sessions = source.hidden_sessions
    target.files = source.files
    target.queues = source.queues
    target.pending_attachment_ids = source.pending_attachment_ids
    target.staged_attachments = source.staged_attachments
    target.commit_unknown_sends = source.commit_unknown_sends
    target.recent_cwds = source.recent_cwds


def session_store_for_manager(
    *,
    existing: Any,
    paths: SessionStorePaths,
    create_store: Callable[[SessionStorePaths], SessionStore],
) -> SessionStore:
    if isinstance(existing, SessionStore) and existing.paths == paths:
        return existing
    store = create_store(paths)
    if isinstance(existing, SessionStore):
        copy_session_store_state(source=existing, target=store)
    return store
