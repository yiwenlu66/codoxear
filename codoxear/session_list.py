from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, MutableMapping

from .session_listing import build_active_session_rows_snapshot
from .session_listing import build_launch_attempt_rows
from .session_listing import build_orphan_recovery_rows
from .session_listing import sort_session_rows
from .session_model import Session
from .session_runtime import ListingRuntimeProbes
from .session_runtime import build_runtime_enriched_session_rows
from .session_store import SessionStore


@dataclass(frozen=True)
class SessionListCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    queues: Callable[[], MutableMapping[str, Any]]
    unattended: Callable[[], MutableMapping[str, Any]]
    aliases: Callable[[], MutableMapping[str, Any]]
    hidden_sessions: Callable[[], set[str]]
    commit_unknown_sends: Callable[[], MutableMapping[str, Any]]
    store: SessionStore
    discover_existing_if_stale: Callable[[], None]
    prune_dead_sessions: Callable[[], None]
    update_meta_counters: Callable[[], None]
    save_files: Callable[[], None]
    save_sidebar_meta: Callable[[], None]
    save_recent_cwds: Callable[[], None]
    now: Callable[[], float]
    runtime_probes: ListingRuntimeProbes
    include_launch_attempts: Callable[[], bool]
    read_launch_attempts: Callable[[], Iterable[dict[str, Any]]]
    launch_attempt_row: Callable[[dict[str, Any]], dict[str, Any] | None]
    clean_unattended_cooldown_minutes: Callable[[Any], int]
    clean_unattended_remaining_injections: Callable[..., int]
    provider_choice_for_settings: Callable[..., str]
    resolve_session_cwd: Callable[[str], Path]
    unattended_default_idle_minutes: int
    unattended_default_max_injections: int
    priority_half_life_seconds: float
    priority_bucket_seconds: float

    def list_sessions(self) -> list[dict[str, Any]]:
        self.discover_existing_if_stale()
        self.prune_dead_sessions()
        self.update_meta_counters()

        files_dirty = False
        sidebar_dirty = False
        recent_cwd_dirty = False
        now_ts = self.now()
        with self.lock:
            snapshot = build_active_session_rows_snapshot(
                sessions=list(self.sessions().values()),
                queues=self.queues(),
                unattended=self.unattended(),
                aliases=self.aliases(),
                store=self.store,
                now_ts=now_ts,
                unattended_default_idle_minutes=self.unattended_default_idle_minutes,
                unattended_default_max_injections=self.unattended_default_max_injections,
                clean_unattended_cooldown_minutes=self.clean_unattended_cooldown_minutes,
                clean_unattended_remaining_injections=self.clean_unattended_remaining_injections,
                provider_choice_for_settings=self.provider_choice_for_settings,
                resolve_session_cwd=self.resolve_session_cwd,
                priority_half_life_seconds=self.priority_half_life_seconds,
                priority_bucket_seconds=self.priority_bucket_seconds,
            )
            items = snapshot.rows
            files_dirty = files_dirty or snapshot.files_dirty
            sidebar_dirty = sidebar_dirty or snapshot.sidebar_dirty
            recent_cwd_dirty = recent_cwd_dirty or snapshot.recent_cwd_dirty

        runtime_result = build_runtime_enriched_session_rows(
            staged_rows=items,
            sessions=self.sessions(),
            lock=self.lock,
            store=self.store,
            probes=self.runtime_probes,
            now_ts=now_ts,
            provider_choice_for_settings=lambda model_provider, preferred_auth_method: self.provider_choice_for_settings(
                model_provider=model_provider,
                preferred_auth_method=preferred_auth_method,
            ),
            priority_half_life_seconds=self.priority_half_life_seconds,
            priority_bucket_seconds=self.priority_bucket_seconds,
        )
        out = runtime_result.rows
        recent_cwd_dirty = recent_cwd_dirty or runtime_result.recent_cwd_dirty

        if self.include_launch_attempts():
            with self.lock:
                hidden_failure_ids = set(self.hidden_sessions())
                active_launch_ids = {
                    s.launch_id
                    for s in self.sessions().values()
                    if isinstance(s.launch_id, str) and s.launch_id
                }
                active_spawn_nonces = {
                    s.spawn_nonce
                    for s in self.sessions().values()
                    if isinstance(s.spawn_nonce, str) and s.spawn_nonce
                }
            out.extend(
                build_launch_attempt_rows(
                    records=self.read_launch_attempts(),
                    hidden_failure_ids=hidden_failure_ids,
                    active_launch_ids=active_launch_ids,
                    active_spawn_nonces=active_spawn_nonces,
                    row_from_record=self.launch_attempt_row,
                )
            )

        with self.lock:
            active_ids = set(self.sessions().keys())
            commit_unknown_snapshot = {
                str(sid): dict(record) if isinstance(record, dict) else record
                for sid, record in self.commit_unknown_sends().items()
            }
            queue_snapshot = {
                str(sid): list(queue) if isinstance(queue, list) else queue
                for sid, queue in self.queues().items()
            }
        existing_out_ids = {str(item.get("session_id")) for item in out if isinstance(item, dict)}
        out.extend(
            build_orphan_recovery_rows(
                active_session_ids=active_ids,
                commit_unknown_sends=commit_unknown_snapshot,
                queues=queue_snapshot,
                existing_session_ids=existing_out_ids,
                now_ts=now_ts,
                unattended_default_idle_minutes=self.unattended_default_idle_minutes,
                unattended_default_max_injections=self.unattended_default_max_injections,
            )
        )

        if files_dirty:
            self.save_files()
        if sidebar_dirty:
            self.save_sidebar_meta()
        if recent_cwd_dirty:
            self.save_recent_cwds()
        sort_session_rows(out)
        return out
