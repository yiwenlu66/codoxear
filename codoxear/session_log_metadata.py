from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Callable, Iterable, Mapping, TextIO

from .agent_backend import normalize_agent_backend


def sessions_dir_for_backend(
    agent_backend: str,
    *,
    codex_sessions_dir: Path,
    pi_sessions_dir: Path,
    cc_sessions_dir: Path,
) -> Path:
    backend_name = normalize_agent_backend(agent_backend)
    if backend_name == "codex":
        return codex_sessions_dir
    if backend_name == "pi":
        return pi_sessions_dir
    if backend_name == "cc":
        return cc_sessions_dir
    raise ValueError(f"unsupported agent_backend: {backend_name}")


def iter_session_logs_for_backend(
    *,
    agent_backend: str,
    sessions_dir_for_backend_func: Callable[[str], Path],
    iter_session_logs: Callable[..., Iterable[Path]],
) -> list[Path]:
    backend_name = normalize_agent_backend(agent_backend)
    return list(iter_session_logs(sessions_dir_for_backend_func(backend_name), agent_backend=backend_name))


def find_session_log_for_session_id(
    session_id: str,
    *,
    agent_backend: str,
    sessions_dir_for_backend_func: Callable[[str], Path],
    find_session_log_for_session_id_func: Callable[..., Path | None],
) -> Path | None:
    backend_name = normalize_agent_backend(agent_backend)
    return find_session_log_for_session_id_func(sessions_dir_for_backend_func(backend_name), session_id, agent_backend=backend_name)


def find_new_session_log(
    *,
    agent_backend: str,
    after_ts: float,
    preexisting: set[Path],
    timeout_s: float,
    sessions_dir_for_backend_func: Callable[[str], Path],
    find_new_session_log_func: Callable[..., tuple[str, Path] | None],
) -> tuple[str, Path] | None:
    backend_name = normalize_agent_backend(agent_backend)
    return find_new_session_log_func(
        sessions_dir=sessions_dir_for_backend_func(backend_name),
        agent_backend=backend_name,
        after_ts=after_ts,
        preexisting=preexisting,
        timeout_s=timeout_s,
    )


def infer_session_meta_backend(log_path: Path, *, pi_sessions_dir: Path, cc_sessions_dir: Path) -> str:
    try:
        log_path.resolve().relative_to(pi_sessions_dir.resolve())
        return "pi"
    except Exception:
        try:
            log_path.resolve().relative_to(cc_sessions_dir.resolve())
            return "cc"
        except Exception:
            return "codex"


def read_session_meta(
    log_path: Path,
    *,
    agent_backend: str | None = None,
    pi_sessions_dir: Path,
    cc_sessions_dir: Path,
    read_session_meta_payload: Callable[..., dict[str, Any] | None],
) -> dict[str, Any]:
    if agent_backend is None:
        backend_name = infer_session_meta_backend(log_path, pi_sessions_dir=pi_sessions_dir, cc_sessions_dir=cc_sessions_dir)
    else:
        backend_name = normalize_agent_backend(agent_backend)
    payload = read_session_meta_payload(log_path, agent_backend=backend_name, timeout_s=0.0)
    if payload is None:
        raise ValueError(f"missing session metadata in {log_path}")
    return payload


def read_session_meta_or_none(
    log_path: Path,
    *,
    agent_backend: str | None = None,
    context: str,
    read_session_meta_func: Callable[..., dict[str, Any]],
    invalid_warnings: set[tuple[str, str]],
    stderr: TextIO = sys.stderr,
) -> dict[str, Any] | None:
    try:
        return read_session_meta_func(log_path, agent_backend=agent_backend)
    except (FileNotFoundError, ValueError) as exc:
        warning_key = (context, str(log_path))
        if warning_key not in invalid_warnings:
            invalid_warnings.add(warning_key)
            stderr.write(f"warning: {context}: ignoring invalid session metadata in {log_path}: {type(exc).__name__}: {exc}\n")
            stderr.flush()
        return None


def turn_context_run_settings(
    payload: Any,
    *,
    clean_optional_text: Callable[[Any], str | None],
    display_reasoning_effort: Callable[[Any], str | None],
) -> tuple[str | None, str | None]:
    if not isinstance(payload, dict):
        return None, None
    return (
        clean_optional_text(payload.get("model")),
        display_reasoning_effort(payload.get("reasoning_effort") or payload.get("effort")),
    )


def read_run_settings_from_log(
    log_path: Path,
    *,
    agent_backend: str,
    read_pi_run_settings: Callable[[Path], tuple[str | None, str | None, str | None]],
    read_cc_run_settings: Callable[[Path], tuple[str | None, str | None, str | None]],
    read_session_meta_or_none_func: Callable[..., dict[str, Any] | None],
    clean_optional_text: Callable[[Any], str | None],
    display_reasoning_effort: Callable[[Any], str | None],
    find_latest_turn_context: Callable[..., Any],
) -> tuple[str | None, str | None, str | None]:
    backend_name = normalize_agent_backend(agent_backend)
    if backend_name == "pi":
        return read_pi_run_settings(log_path)
    if backend_name == "cc":
        return read_cc_run_settings(log_path)
    meta = read_session_meta_or_none_func(log_path, agent_backend="codex", context="run settings")
    model_provider = clean_optional_text(meta.get("model_provider")) if meta is not None else None
    model = clean_optional_text(meta.get("model")) if meta is not None else None
    reasoning_effort = display_reasoning_effort(meta.get("reasoning_effort")) if meta is not None else None
    if model is None or reasoning_effort is None:
        ctx_model, ctx_effort = turn_context_run_settings(
            find_latest_turn_context(log_path, max_scan_bytes=8 * 1024 * 1024),
            clean_optional_text=clean_optional_text,
            display_reasoning_effort=display_reasoning_effort,
        )
        if model is None:
            model = ctx_model
        if reasoning_effort is None:
            reasoning_effort = ctx_effort
    return model_provider, model, reasoning_effort
