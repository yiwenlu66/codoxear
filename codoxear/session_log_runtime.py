from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, MutableMapping

from .agent_backend import get_agent_backend
from .session_log_projection import LogDerivedSessionObservation
from .session_log_projection import log_identity
from .session_log_projection import log_revision
from .session_model import Session
from .util import _codex_sessions_dir_for_log
from .util import scan_active_cc_subagents
from .util import scan_active_codex_subagents
from .util import scan_active_pi_subagents
from .session_runtime import suppress_session_interrupted_idle
from .token_signal import TOKEN_NONE
from .token_signal import TokenObservation
from .token_signal import coerce_token_observation


@dataclass(frozen=True)
class SessionLogRuntimeCoordinator:
    lock: Any
    sessions: Callable[[], MutableMapping[str, Session]]
    analyze_log_chunk: Callable[..., tuple[Any, Any, Any, Any, Any, Any, Any, Any]]
    turn_context_run_settings: Callable[[Any], tuple[str | None, str | None]]
    compute_idle_from_log: Callable[[Path], bool | None]
    read_jsonl_from_offset: Callable[..., tuple[list[dict[str, Any]], int]]
    find_latest_token_update: Callable[[Path], dict[str, Any] | None]

    def observation_from_rows(
        self,
        *,
        agent_backend: str,
        log_path: Path,
        revision: tuple[int, int, int, int],
        start_off: int,
        end_off: int,
        objs: list[dict[str, Any]],
        invalidate_idle_cache: bool = True,
    ) -> LogDerivedSessionObservation:
        """Normalize one backend-native JSONL range into the registry schema."""
        _thinking, _thinking_tokens, _tools, _system, last_ts, token_update, _chat_events, _turn_state = self.analyze_log_chunk(objs)
        provider, model, effort = get_agent_backend(agent_backend).run_settings_from_log_rows(
            objs,
            turn_context_run_settings=self.turn_context_run_settings,
        )
        return LogDerivedSessionObservation(
            log_path=log_path,
            revision=revision,
            start_off=start_off,
            end_off=end_off,
            token=coerce_token_observation(token_update),
            model_provider=provider,
            model=model,
            reasoning_effort=effort,
            last_conversation_ts=float(last_ts) if isinstance(last_ts, (int, float)) else None,
            invalidate_idle_cache=invalidate_idle_cache,
        )

    def commit_log_observation(self, session_id: str, observation: LogDerivedSessionObservation) -> bool:
        """Atomically commit a current, monotonic JSONL observation.

        The session binding path and the file identity must still match what
        the reader observed. An unchanged exact revision is accepted; ordinary
        append growth is also accepted when it covers every byte the reader
        actually consumed. Within one path/device/inode generation, a commit
        whose end precedes the last accepted end is rejected. A changed
        path/inode (rebind) or a current same-file revision smaller than the
        prior end (truncation) starts a legitimate new offset generation.

        Source priority is unchanged: full-refresh observations may carry
        already-reconciled effective settings; raw Pi log observations always
        own provider/model but cannot replace a live bridge-owned effort.
        """
        identity = log_identity(observation.log_path, observation.revision)
        with self.lock:
            current_revision = log_revision(observation.log_path)
            if current_revision is None:
                return False
            current_identity = log_identity(observation.log_path, current_revision)
            revision_unchanged = current_revision == observation.revision
            observed_growth = int(observation.end_off) > int(observation.revision[2])
            append_only_growth = (
                observed_growth
                and current_identity == identity
                and int(current_revision[2]) >= int(observation.end_off)
                and int(current_revision[3]) >= int(observation.revision[3])
            )
            if not revision_unchanged and not append_only_growth:
                return False
            session = self.sessions().get(session_id)
            if session is None or session.log_path != observation.log_path:
                return False

            prior_identity = session.log_projection_identity
            prior_end = int(session.log_projection_end)
            reset_generation = prior_identity != identity or int(observation.revision[2]) < prior_end
            if not reset_generation and int(observation.end_off) < prior_end:
                return False

            if reset_generation:
                session.log_projection_identity = identity
                session.log_projection_end = 0
                if prior_identity == identity and int(observation.revision[2]) < prior_end:
                    session.last_chat_ts = None
                    session.last_chat_history_scanned = False
                    session.run_settings_log_revision = None

            if observation.last_conversation_ts is not None:
                session.last_chat_ts = (
                    observation.last_conversation_ts
                    if session.last_chat_ts is None
                    else max(session.last_chat_ts, observation.last_conversation_ts)
                )
            if observation.history_scanned:
                session.last_chat_history_scanned = True
            if observation.token.observed:
                session.token = observation.token.public_token
            if observation.replace_settings or observation.model_provider is not None:
                session.model_provider = observation.model_provider
            if observation.replace_settings or observation.model is not None:
                session.model = observation.model
            bridge_live = (
                session.agent_backend == "pi"
                and bool(session.pi_thinking_command)
                and self._pid_alive(session.broker_pid)
            )
            if observation.replace_settings:
                session.reasoning_effort = observation.reasoning_effort
            elif observation.reasoning_effort is not None and (observation.effective_settings or not bridge_live):
                session.reasoning_effort = observation.reasoning_effort
            if observation.settings_revision is not None:
                session.run_settings_log_revision = observation.settings_revision
            if observation.invalidate_idle_cache:
                session.idle_cache_log_off = -1
            session.log_projection_revision = current_revision
            session.log_projection_end = int(observation.end_off)
            return True

    @staticmethod
    def _pid_alive(pid: Any) -> bool:
        try:
            pid_int = int(pid)
        except (TypeError, ValueError):
            return False
        if pid_int <= 0:
            return False
        try:
            import os

            os.kill(pid_int, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except Exception:
            return False
        return True

    def update_meta_counters(self) -> None:
        with self.lock:
            items = list(self.sessions().items())
        for sid, session in items:
            log_path = session.log_path
            if log_path is None or (not log_path.exists()):
                continue
            source_revision = log_revision(log_path)
            if source_revision is None:
                continue
            size = int(source_revision[2])
            offset = int(session.meta_log_off)
            reset_last_chat = False
            if size < offset:
                offset = 0
                reset_last_chat = True

            # Stale interrupted-idle guard: ``interrupted_idle_log_off`` is the
            # log byte offset captured when the broker last confirmed an
            # interrupt. Content at or beyond it arrived after the interrupt,
            # so it is post-interrupt activity that proves the turn resumed and
            # invalidates the stored interrupted-idle override. Advance the read
            # cursor past any pre-baseline bytes so the first chunk processed is
            # unambiguously post-interrupt; this keeps the interrupted turn's
            # own non-final tail (which keeps the override alive) separate from
            # genuine resumed activity. The skipped bytes only affect meta
            # counters, which are reset to zero below whenever the session is
            # not busy (the interrupted-idle case), so nothing is lost.
            interrupted_idle_active = bool(session.interrupted_idle)
            interrupted_idle_baseline = int(session.interrupted_idle_log_off) if interrupted_idle_active else 0
            clear_interrupted_idle = False
            post_baseline = False
            if interrupted_idle_active and 0 < interrupted_idle_baseline <= size:
                if offset < interrupted_idle_baseline:
                    offset = interrupted_idle_baseline
                post_baseline = True

            total_thinking = 0
            total_thinking_tokens = 0
            total_tools = 0
            total_system = 0
            turn_open = bool(session.meta_turn_open)
            counters_reset = False
            codex_reasoning_total = session.meta_codex_reasoning_total
            latest_chat_ts: float | None = None
            latest_token_observation: TokenObservation = TOKEN_NONE
            loops = 0
            while offset < size and loops < 16:
                objs, new_offset = self.read_jsonl_from_offset(log_path, offset, max_bytes=256 * 1024)
                if new_offset <= offset:
                    break
                (
                    delta_thinking,
                    delta_thinking_tokens,
                    delta_tools,
                    delta_system,
                    chunk_chat_ts,
                    token_update,
                    chat_events,
                    chunk_turn_state,
                ) = self.analyze_log_chunk(
                    objs,
                    initial_turn_open=turn_open,
                    initial_codex_reasoning_total=codex_reasoning_total,
                )
                codex_reasoning_total = chunk_turn_state.codex_reasoning_total
                token_observation = coerce_token_observation(token_update)
                if chunk_turn_state.counters_reset:
                    # A user row arrived while the scanner's persisted turn was
                    # closed. Discard both prior-session counters and activity
                    # from an older turn earlier in this multi-chunk scan.
                    total_thinking = delta_thinking
                    total_thinking_tokens = delta_thinking_tokens
                    total_tools = delta_tools
                    total_system = delta_system
                    counters_reset = True
                else:
                    total_thinking += delta_thinking
                    total_thinking_tokens += delta_thinking_tokens
                    total_tools += delta_tools
                    total_system += delta_system
                turn_open = bool(chunk_turn_state.turn_open)
                if chunk_chat_ts is not None:
                    latest_chat_ts = chunk_chat_ts if latest_chat_ts is None else max(latest_chat_ts, chunk_chat_ts)
                if token_observation.observed:
                    latest_token_observation = token_observation
                # Any user/assistant turn activity in a post-baseline chunk
                # proves the turn resumed after the interrupt. Visible
                # conversation rows surface as chat events; reasoning/tool-only
                # rows surface as thinking/tool counter deltas. Together they
                # cover every form of resumed turn activity (a lone
                # token_count after interrupt does not, and must not, clear).
                if post_baseline and (
                    any(isinstance(e, dict) and e.get("role") in ("user", "assistant") for e in chat_events)
                    or delta_thinking > 0
                    or delta_tools > 0
                ):
                    clear_interrupted_idle = True
                offset = new_offset
                loops += 1

            fallback_token: dict[str, Any] | None = None
            if not latest_token_observation.observed and session.token is None:
                fallback_token = self.find_latest_token_update(log_path)
            if source_revision is not None:
                observation = LogDerivedSessionObservation(
                    log_path=log_path,
                    revision=source_revision,
                    start_off=0 if reset_last_chat else int(session.meta_log_off),
                    end_off=min(offset, source_revision[2]),
                    token=(
                        latest_token_observation
                        if latest_token_observation.observed
                        else coerce_token_observation(fallback_token)
                    ),
                    last_conversation_ts=latest_chat_ts,
                )
                self.commit_log_observation(sid, observation)

            with self.lock:
                current = self.sessions().get(sid)
                if not current or current.log_path != log_path:
                    continue
                current.meta_codex_reasoning_total = codex_reasoning_total
                if current.busy:
                    if counters_reset:
                        current.meta_thinking = total_thinking
                        current.meta_thinking_tokens = total_thinking_tokens
                        current.meta_tools = total_tools
                        current.meta_system = total_system
                    else:
                        current.meta_thinking += total_thinking
                        current.meta_thinking_tokens += total_thinking_tokens
                        current.meta_tools += total_tools
                        current.meta_system += total_system
                    current.meta_turn_open = turn_open
                else:
                    # Episode-aware reset: while the session has active
                    # subagents, the user's work episode continues in the
                    # background even though the main turn is closed. Preserve
                    # the counters so the next delivery turn resumes them
                    # monotonically instead of showing a mid-episode reset.
                    if current.agent_backend == "pi":
                        subagents_active = bool(scan_active_pi_subagents().get(str(log_path)))
                    elif current.agent_backend == "codex":
                        sessions_dir = _codex_sessions_dir_for_log(log_path)
                        subagents_active = bool(
                            sessions_dir is not None
                            and scan_active_codex_subagents(sessions_dirs=(sessions_dir,)).get(current.thread_id)
                        )
                    elif current.agent_backend == "cc":
                        subagents_active = bool(
                            current.owned
                            and current.thread_id
                            and scan_active_cc_subagents(parent_broker_pids={current.thread_id: current.broker_pid}).get(current.thread_id)
                        )
                    else:
                        subagents_active = False
                    if not subagents_active:
                        current.meta_thinking = 0
                        current.meta_thinking_tokens = 0
                        current.meta_tools = 0
                        current.meta_system = 0
                        current.meta_turn_open = False
                if clear_interrupted_idle and current.interrupted_idle:
                    suppress_session_interrupted_idle(current)
                current.meta_log_off = offset if offset >= 0 else current.meta_log_off

    def mark_log_delta(
        self,
        session_id: str,
        *,
        objs: list[dict[str, Any]],
        new_off: int,
        start_off: int = 0,
        expected_log_path: Path | None = None,
        revision: tuple[int, int, int, int] | None = None,
    ) -> bool:
        with self.lock:
            current = self.sessions().get(session_id)
            if current is None:
                return False
            agent_backend = current.agent_backend
            log_path = expected_log_path if expected_log_path is not None else current.log_path
        if log_path is None:
            return False
        observed_revision = revision if revision is not None else log_revision(log_path)
        if observed_revision is None:
            return False
        observation = self.observation_from_rows(
            agent_backend=agent_backend,
            log_path=log_path,
            revision=observed_revision,
            start_off=start_off,
            end_off=new_off,
            objs=objs,
        )
        return self.commit_log_observation(session_id, observation)

    def idle_from_log(self, session_id: str) -> bool:
        with self.lock:
            session = self.sessions().get(session_id)
            if not session:
                raise KeyError("unknown session")
            log_path = session.log_path
        if log_path is None:
            raise FileNotFoundError(f"missing rollout log for session {session_id}")
        return self.idle_from_log_path(session_id, log_path)

    def idle_from_log_path(self, session_id: str, log_path: Path) -> bool:
        with self.lock:
            session = self.sessions().get(session_id)
            cache_matches_path = bool(session and session.log_path == log_path)
            cached_off = int(session.idle_cache_log_off) if cache_matches_path and session else -1
            cached_idle = session.idle_cache_value if cache_matches_path and session else None
        if not log_path.exists():
            raise FileNotFoundError(f"missing rollout log for session {session_id}")
        size = int(log_path.stat().st_size)
        if cache_matches_path and (size >= 0) and (cached_off == size) and isinstance(cached_idle, bool):
            return bool(cached_idle)
        idle = self.compute_idle_from_log(log_path)
        with self.lock:
            current = self.sessions().get(session_id)
            if current and current.log_path == log_path:
                current.idle_cache_log_off = size
                current.idle_cache_value = idle
        if idle is None:
            raise RuntimeError("unable to compute idle state from log")
        return bool(idle)
