#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pty
import re
import signal
import socket
import sys
import termios
import threading
import time
import traceback
import tty
from pathlib import Path
from typing import Any

from codoxear.agent_backend import get_agent_backend
from codoxear.agent_backend import normalize_agent_backend
from codoxear.pi_log import pi_complete_jsonl_offset_before as _pi_complete_jsonl_offset_before
from codoxear.pi_log import pi_context_token_update as _pi_context_token_update
from codoxear.pi_log import pi_token_update as _pi_token_update
from codoxear import pty_util as _pty_util
from codoxear.broker_launch import SHELL_PRE_EXEC_MARKER
from codoxear.broker_launch import SHELL_PRE_EXEC_MARKER_BYTES
from codoxear.broker_launch import _agent_shell_command
from codoxear.broker_launch import _ensure_pi_bridge_args as _ensure_pi_bridge_args_impl
from codoxear.broker_launch import _ensure_pi_session_arg as _ensure_pi_session_arg_impl
from codoxear.broker_launch import _expand_cwd
from codoxear.broker_launch import _pi_active_session_marker_path
from codoxear.broker_launch import _pi_bridge_extension_path
from codoxear.broker_launch import _pi_new_session_log_path
from codoxear.broker_launch import _pi_session_dir_from_args
from codoxear.broker_launch import _pi_session_dir_name
from codoxear.broker_launch import _read_pi_active_session_marker
from codoxear.broker_launch import _reset_pi_active_session_marker
from codoxear.broker_launch import _resume_session_id_from_args as _resume_session_id_from_args_impl
from codoxear.broker_launch import _session_log_path_from_args
from codoxear.broker_launch import _shell_argv_for_command as _shell_argv_for_command_impl
from codoxear.broker_launch import _user_shell
from codoxear.broker_log_binding import _apply_broker_log_binding_to_state
from codoxear.broker_log_binding import _resolve_broker_log_binding
from codoxear.broker_log_binding import _seed_broker_log_state
from codoxear.broker_turn_state import INTERRUPT_HINT_TAIL_MAX
from codoxear.broker_turn_state import State
from codoxear.broker_turn_state import _apply_rollout_obj_to_state
from codoxear.broker_turn_state import _close_turn_state
from codoxear.broker_turn_state import _hint_seen_in_new_text
from codoxear.broker_turn_state import _mark_busy_state_idle
from codoxear.broker_turn_state import _mark_explicit_interrupt_request
from codoxear.broker_turn_state import _should_clear_busy_state as _should_clear_busy_state_impl
from codoxear.broker_turn_state import _strip_ansi
from codoxear.broker_turn_state import _update_busy_from_pty_text
from codoxear.control_socket import handle_control_socket_connection as _handle_control_socket_connection
from codoxear.util import append_launch_attempt as _append_launch_attempt
from codoxear.util import default_app_dir as _default_app_dir
from codoxear.util import find_new_session_log as _find_new_session_log
from codoxear.util import iter_session_logs as _iter_session_logs
from codoxear.util import launch_attempts_path as _launch_attempts_path
from codoxear.util import pid_alive as _pid_alive
from codoxear.util import process_group_alive as _process_group_alive
from codoxear.util import proc_find_open_rollout_log as _proc_find_open_rollout_log
from codoxear.util import _paths_match as _paths_match
from codoxear.util import read_launch_attempts as _read_launch_attempts
from codoxear.util import read_jsonl_from_offset as _read_jsonl_from_offset_impl
from codoxear.util import redacted_launch_attempt_persist_record as _redacted_launch_attempt_persist_record
from codoxear.util import session_id_from_rollout_path as _session_id_from_rollout_path
from codoxear.util import _send_socket_json_line as _send_socket_json_line
from codoxear.util import _socket_peer_disconnected as _socket_peer_disconnected


APP_DIR = _default_app_dir()
SOCK_DIR = APP_DIR / "socks"
LAUNCH_ATTEMPTS_PATH = _launch_attempts_path(APP_DIR)
PROC_ROOT = Path("/proc")

AGENT_BACKEND = normalize_agent_backend(os.environ.get("CODEX_WEB_AGENT_BACKEND"), default="codex")
BACKEND = get_agent_backend(AGENT_BACKEND)
AGENT_BIN = BACKEND.cli_bin()
OWNER_TAG = os.environ.get("CODEX_WEB_OWNER", "")
MODEL_PROVIDER_OVERRIDE = os.environ.get("CODEX_WEB_MODEL_PROVIDER", "").strip()
PREFERRED_AUTH_METHOD_OVERRIDE = os.environ.get("CODEX_WEB_PREFERRED_AUTH_METHOD", "").strip()
MODEL_OVERRIDE = os.environ.get("CODEX_WEB_MODEL", "").strip()
REASONING_EFFORT_OVERRIDE = os.environ.get("CODEX_WEB_REASONING_EFFORT", "").strip().lower()
SERVICE_TIER_OVERRIDE = os.environ.get("CODEX_WEB_SERVICE_TIER", "").strip().lower()
DEFAULT_AGENT_HOME = BACKEND.home()
DEBUG = os.environ.get("CODEX_WEB_BROKER_DEBUG", "0") == "1"
_BUSY_QUIET_RAW = os.environ.get("CODEX_WEB_BUSY_QUIET_SECONDS")
if _BUSY_QUIET_RAW is None or (not _BUSY_QUIET_RAW.strip()):
    _BUSY_QUIET_RAW = "3.0"
BUSY_QUIET_SECONDS = max(float(_BUSY_QUIET_RAW), 0.0)

_BUSY_INTERRUPT_GRACE_RAW = os.environ.get("CODEX_WEB_BUSY_INTERRUPT_GRACE_SECONDS")
if _BUSY_INTERRUPT_GRACE_RAW is None or (not _BUSY_INTERRUPT_GRACE_RAW.strip()):
    _BUSY_INTERRUPT_GRACE_RAW = "3.0"
BUSY_INTERRUPT_GRACE_SECONDS = max(float(_BUSY_INTERRUPT_GRACE_RAW), 0.0)

_SHELL_STARTUP_TIMEOUT_RAW = os.environ.get("CODEX_WEB_SHELL_STARTUP_TIMEOUT_SECONDS")
if _SHELL_STARTUP_TIMEOUT_RAW is None or (not _SHELL_STARTUP_TIMEOUT_RAW.strip()):
    _SHELL_STARTUP_TIMEOUT_RAW = "15.0"
SHELL_STARTUP_TIMEOUT_SECONDS = max(float(_SHELL_STARTUP_TIMEOUT_RAW), 0.0)
def _dprint(msg: str) -> None:
    if not DEBUG:
        return
    sys.stderr.write(msg.rstrip("\n") + "\n")
    sys.stderr.flush()


def _now() -> float:
    return time.time()


def _record_launch_attempt(record: dict[str, Any]) -> None:
    if OWNER_TAG != "web":
        return
    try:
        launch_id = record.get("launch_id")
        if isinstance(launch_id, str) and launch_id and "submitted_user_messages" not in record:
            for prev in _read_launch_attempts(path=LAUNCH_ATTEMPTS_PATH, max_records=100, max_age_s=24 * 3600):
                if prev.get("launch_id") != launch_id:
                    continue
                submitted = prev.get("submitted_user_messages")
                if isinstance(submitted, list) and submitted:
                    record = dict(record)
                    record["submitted_user_messages"] = submitted
                break
        rec = _append_launch_attempt(_redacted_launch_attempt_persist_record(record), path=LAUNCH_ATTEMPTS_PATH)
        if rec.get("state") == "failed":
            sys.stderr.write(
                "error: session launch failed: "
                f"{rec.get('launch_id')}: {rec.get('stage')}: {rec.get('error')}\n"
            )
            sys.stderr.flush()
    except Exception as e:
        sys.stderr.write(f"error: failed to write launch attempt record: {type(e).__name__}: {e}\n")
        sys.stderr.flush()


def _broker_launch_record(
    *,
    stage: str,
    error: str,
    cwd: str,
    start_ts: float,
    agent_pid: int | None = None,
    log_path: Path | None = None,
    exit_code: int | None = None,
) -> dict[str, Any]:
    return {
        "launch_id": (os.environ.get("CODEX_WEB_LAUNCH_ID") or "").strip() or None,
        "state": "failed",
        "stage": stage,
        "error": error,
        "agent_backend": AGENT_BACKEND,
        "cwd": cwd,
        "created_ts": start_ts,
        "updated_ts": time.time(),
        "broker_pid": os.getpid(),
        "agent_pid": agent_pid,
        "exit_code": exit_code,
        "log_path": str(log_path) if log_path else None,
        "transport": (os.environ.get("CODEX_WEB_TRANSPORT") or "").strip() or None,
        "tmux_session": (os.environ.get("CODEX_WEB_TMUX_SESSION") or "").strip() or None,
        "tmux_window": (os.environ.get("CODEX_WEB_TMUX_WINDOW") or "").strip() or None,
        "spawn_nonce": (os.environ.get("CODEX_WEB_SPAWN_NONCE") or "").strip() or None,
        "model_provider": MODEL_PROVIDER_OVERRIDE or None,
        "preferred_auth_method": PREFERRED_AUTH_METHOD_OVERRIDE or None,
        "model": MODEL_OVERRIDE or None,
        "reasoning_effort": REASONING_EFFORT_OVERRIDE or None,
        "service_tier": SERVICE_TIER_OVERRIDE or None,
        "resume_session_id": (os.environ.get("CODEX_WEB_RESUME_SESSION_ID") or "").strip() or None,
    }


























def _resume_session_id_from_args(args: list[str]) -> str | None:
    return _resume_session_id_from_args_impl(args, agent_backend=AGENT_BACKEND)


def _ensure_pi_bridge_args(*, args: list[str], marker_path: Path) -> list[str]:
    return _ensure_pi_bridge_args_impl(args=args, marker_path=marker_path, agent_backend=AGENT_BACKEND)


def _ensure_pi_session_arg(*, args: list[str], cwd: str, sessions_dir: Path) -> list[str]:
    return _ensure_pi_session_arg_impl(args=args, cwd=cwd, sessions_dir=sessions_dir, agent_backend=AGENT_BACKEND)


def _shell_argv_for_command(cmd: str) -> list[str]:
    return _shell_argv_for_command_impl(cmd, user_shell=_user_shell)


def _set_pdeathsig(sig: int) -> None:
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, sig, 0, 0, 0)
    except Exception:
        return


def _require_proc() -> None:
    if sys.platform.startswith("linux"):
        if not (PROC_ROOT / "self" / "fd").is_dir():
            sys.stderr.write("error: codoxear-broker requires /proc (missing /proc/self/fd).\n")
            raise SystemExit(2)
    elif sys.platform == "darwin":
        pass  # macOS is supported via lsof/pgrep
    else:
        sys.stderr.write(f"error: codoxear-broker requires Linux or macOS (unsupported: {sys.platform}).\n")
        raise SystemExit(2)










def _exec_agent(*, cwd: str, agent_args: list[str]) -> None:
    argv = [AGENT_BIN, *agent_args]
    os.chdir(cwd)
    os.execvpe(argv[0], argv, os.environ)

def _exec_agent_via_login_shell(*, cwd: str, agent_args: list[str], pty_slave_path: str) -> None:
    argv = [AGENT_BIN, *agent_args]
    cmd = _agent_shell_command(argv, pty_slave_path=pty_slave_path)
    shell_argv = _shell_argv_for_command(cmd)
    os.chdir(cwd)
    os.execvpe(shell_argv[0], shell_argv, os.environ)


def _enter_seq_bytes() -> bytes:
    return _seq_bytes(os.environ.get("CODEX_WEB_ENTER_SEQ", "\r"))


def _seq_bytes(raw: str) -> bytes:
    b = _pty_util.seq_bytes(raw)
    return b if b else b"\r"


def _encode_enter() -> bytes:
    b = _enter_seq_bytes()
    if DEBUG:
        _dprint(f"broker: enter_seq={b!r}")
    return b


def _write_all(fd: int, data: bytes) -> None:
    _pty_util.write_all(fd, data)


def _inject(fd: int, *, text: str, suffix: bytes, delay_s: float = 0.05) -> None:
    _pty_util.inject_bracketed_paste(fd, text=text, suffix=suffix, delay_s=delay_s)


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    _pty_util.set_winsize(fd, rows, cols)


def _term_size() -> tuple[int, int]:
    try:
        sz = os.get_terminal_size(sys.stdin.fileno())
        return int(sz.lines), int(sz.columns)
    except Exception:
        return 40, 120

def _claimed_log_paths_from_sock_meta(*, sock_dir: Path, exclude_sock: Path | None = None) -> set[Path]:
    out: set[Path] = set()
    if not sock_dir.exists():
        return out
    for meta_path in sock_dir.glob("*.json"):
        sock_path = meta_path.with_suffix(".sock")
        if exclude_sock is not None and _paths_match(sock_path, exclude_sock):
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(meta, dict):
            continue
        log_path_raw = meta.get("log_path")
        if not isinstance(log_path_raw, str) or not log_path_raw.strip():
            continue
        broker_pid = int(meta.get("broker_pid")) if isinstance(meta.get("broker_pid"), int) else 0
        agent_pid = int(meta.get("codex_pid")) if isinstance(meta.get("codex_pid"), int) else 0
        if (broker_pid > 0 or agent_pid > 0) and (not _pid_alive(broker_pid)) and (not _pid_alive(agent_pid)):
            continue
        path = Path(log_path_raw)
        try:
            out.add(path.resolve())
        except Exception:
            out.add(path)
    return out


_DETACH_TRIGGER_PHRASES: dict[str, tuple[str, ...]] = {"codex": ("To continue this session, run ",)}


def _detach_current_session_binding(st: "State") -> None:
    for p in (st.log_path, st.last_rollout_path, st.last_detected_rollout_path):
        if p is not None:
            st.ignored_rollout_paths.add(p)
    st.log_path = None
    st.session_id = None
    st.log_off = 0
    st.last_interrupt_request_ts = 0.0
    st.last_interrupted_idle_ts = 0.0
    st.last_rollout_path = None
    st.last_detected_rollout_path = None
    st.detach_trigger_tail = ""


def _detach_trigger_seen(*, agent_backend: str, tail: str, cleaned: str) -> bool:
    for phrase in _DETACH_TRIGGER_PHRASES.get(agent_backend, ()):
        if _hint_seen_in_new_text(tail=tail, cleaned=cleaned, phrase=phrase):
            return True
    return False


def _maybe_detach_on_session_switch_trigger(*, st: "State", tail: str, cleaned: str, agent_backend: str) -> bool:
    if not _detach_trigger_seen(agent_backend=agent_backend, tail=tail, cleaned=cleaned):
        return False
    _detach_current_session_binding(st)
    return True


def _read_jsonl_from_offset(path: Path, offset: int, max_bytes: int = 256 * 1024) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], offset
    return _read_jsonl_from_offset_impl(path, offset, max_bytes=max_bytes, advance_on_oversized_unterminated=False)


def _should_clear_busy_state(st: State, now_ts: float) -> bool:
    return _should_clear_busy_state_impl(
        st,
        now_ts,
        busy_quiet_seconds=BUSY_QUIET_SECONDS,
        busy_interrupt_grace_seconds=BUSY_INTERRUPT_GRACE_SECONDS,
    )










def _observe_shell_pre_exec_marker(st: State, chunk: bytes, *, now_ts: float) -> None:
    if st.shell_pre_exec_marker_seen:
        return
    combined = st.shell_pre_exec_marker_tail + chunk
    if SHELL_PRE_EXEC_MARKER_BYTES in combined:
        st.shell_pre_exec_marker_seen = True
        st.shell_pre_exec_marker_ts = now_ts
        st.shell_pre_exec_marker_tail = b""
        return
    keep = max(len(SHELL_PRE_EXEC_MARKER_BYTES) - 1, 0)
    st.shell_pre_exec_marker_tail = combined[-keep:] if keep else b""


class Broker:
    def __init__(self, *, cwd: str, codex_args: list[str]) -> None:
        self.cwd = cwd
        self.pi_active_session_marker_path = _pi_active_session_marker_path()
        if AGENT_BACKEND == "pi":
            _reset_pi_active_session_marker(self.pi_active_session_marker_path)
        base_args = _ensure_pi_session_arg(args=codex_args, cwd=self.cwd, sessions_dir=BACKEND.sessions_dir())
        base_args = _ensure_pi_bridge_args(args=base_args, marker_path=self.pi_active_session_marker_path)
        # Headless web sessions need different defaults for robust injection and log discovery.
        # These flags are only for the interactive Codex CLI.
        if OWNER_TAG == "web" and AGENT_BACKEND == "codex":
            forced = ["-c", "disable_response_storage=false", "-c", "disable_paste_burst=true"]
            self.codex_args = forced + base_args
        else:
            self.codex_args = base_args
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.state: State | None = None
        self._emulate_terminal = (os.environ.get("CODEX_WEB_EMULATE_TERMINAL", "0") == "1") or (not sys.stdin.isatty())
        self._term_query_buf = b""
        self._stdin_termios: list[Any] | None = None

        self.codex_home = DEFAULT_AGENT_HOME
        self.sessions_dir = BACKEND.sessions_dir()
        resume_env = str(os.environ.get("CODEX_WEB_RESUME_SESSION_ID") or "").strip()
        self._resume_session_id = resume_env or _resume_session_id_from_args(self.codex_args)

    def _teardown_managed_process_group(self, *, wait_seconds: float = 1.0) -> None:
        self._stop.set()
        with self._lock:
            st = self.state
        if not st:
            return
        root_pid = int(st.codex_pid)
        if not _process_group_alive(root_pid):
            return
        try:
            os.killpg(root_pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = _now() + max(wait_seconds, 0.0)
        while _process_group_alive(root_pid):
            if _now() >= deadline:
                break
            time.sleep(0.05)
        if not _process_group_alive(root_pid):
            return
        try:
            os.killpg(root_pid, signal.SIGKILL)
        except ProcessLookupError:
            return

    def _discover_log_watcher(self) -> None:
        try:
            while not self._stop.is_set():
                with self._lock:
                    st = self.state
                    if not st:
                        return
                    current_log_path = st.log_path
                    current_session_id = st.session_id
                    current_last_rollout_path = st.last_rollout_path
                    declared_log_path = st.declared_log_path
                    root_pid = int(st.codex_pid)
                    ignored_paths = set(st.ignored_rollout_paths)
                if root_pid > 0:
                    if AGENT_BACKEND == "pi":
                        if declared_log_path is not None and declared_log_path.exists():
                            if (
                                current_log_path is None
                                or (not _paths_match(declared_log_path, current_log_path))
                                or current_session_id is None
                                or current_last_rollout_path is None
                                or (not _paths_match(current_last_rollout_path, declared_log_path))
                            ):
                                self._maybe_register_or_switch_rollout(log_path=declared_log_path)
                                time.sleep(0.25)
                                continue
                        lp = _read_pi_active_session_marker(
                            self.pi_active_session_marker_path,
                            sessions_dir=self.sessions_dir,
                        )
                        if lp is not None and lp.exists():
                            if current_log_path is None or (not _paths_match(lp, current_log_path)):
                                self._maybe_register_or_switch_rollout(log_path=lp)
                                time.sleep(0.25)
                                continue
                    else:
                        lp = _proc_find_open_rollout_log(
                            proc_root=PROC_ROOT,
                            root_pid=root_pid,
                            agent_backend=AGENT_BACKEND,
                            cwd=self.cwd,
                            ignored_paths=ignored_paths,
                        )
                        if lp and lp.exists():
                            if current_log_path is None or (not _paths_match(lp, current_log_path)):
                                self._maybe_register_or_switch_rollout(log_path=lp)
                                time.sleep(0.25)
                                continue
                        if AGENT_BACKEND == "cc" and current_log_path is None:
                            found = _find_new_session_log(
                                sessions_dir=self.sessions_dir,
                                agent_backend=AGENT_BACKEND,
                                cwd=self.cwd,
                                after_ts=st.start_ts,
                                preexisting=st.known_rollout_paths,
                                exclude_paths=ignored_paths,
                                timeout_s=0.0,
                            )
                            if found is not None:
                                _sid, fallback_log_path = found
                                self._maybe_register_or_switch_rollout(log_path=fallback_log_path)
                                time.sleep(0.25)
                                continue
                    # Exit early if Codex is gone.
                    try:
                        wpid, _status = os.waitpid(root_pid, os.WNOHANG)
                        if wpid == root_pid:
                            return
                    except ChildProcessError:
                        return
                    except Exception:
                        raise
                time.sleep(0.25)
        except Exception:
            sys.stderr.write(f"error: log discover watcher crashed: {traceback.format_exc()}\n")
            try:
                self._teardown_managed_process_group()
            except Exception:
                traceback.print_exc()

    def _register_from_log(self, *, log_path: Path) -> bool:
        sid = self._session_id_from_rollout_path(log_path)
        if sid is None:
            raise RuntimeError(f"unable to determine session_id from rollout filename: {log_path}")
        if not sid:
            _dprint(f"broker: register_from_log: no session id: {log_path}")
            return False

        try:
            raw_off = int(log_path.stat().st_size)
        except Exception:
            raw_off = 0
        if AGENT_BACKEND == "pi" and raw_off > 0:
            try:
                off = _pi_complete_jsonl_offset_before(log_path, raw_off)
            except Exception:
                off = 0
        else:
            off = raw_off

        headless = (OWNER_TAG == "web")
        sock_path = SOCK_DIR / f"{sid}-{os.getpid()}.sock"
        with self._lock:
            st = self.state
            if not st:
                return False
            st.log_path = log_path
            st.session_id = sid
            if not headless:
                st.sock_path = sock_path
            st.log_off = off

        _dprint(f"broker: registered session_id={sid} log_path={log_path} sock_path={sock_path}")
        self._write_meta()
        if not headless:
            threading.Thread(target=self._sock_server, daemon=True).start()
            threading.Thread(target=self._log_watcher, daemon=True).start()
        return True

    def _maybe_reply_to_terminal_queries(self, *, fd: int, b: bytes) -> None:
        if not self._emulate_terminal:
            return
        self._term_query_buf = (self._term_query_buf + b)[-256:]
        if b"\x1b[5n" in self._term_query_buf:
            try:
                _write_all(fd, b"\x1b[0n")
            except Exception:
                traceback.print_exc()
            self._term_query_buf = self._term_query_buf.replace(b"\x1b[5n", b"")
        if b"\x1b[6n" in self._term_query_buf:
            try:
                _write_all(fd, b"\x1b[1;1R")
            except Exception:
                traceback.print_exc()
            self._term_query_buf = self._term_query_buf.replace(b"\x1b[6n", b"")
        if b"\x1b[c" in self._term_query_buf:
            try:
                _write_all(fd, b"\x1b[?1;2c")
            except Exception:
                traceback.print_exc()
            self._term_query_buf = self._term_query_buf.replace(b"\x1b[c", b"")
        if b"\x1b[>c" in self._term_query_buf:
            try:
                _write_all(fd, b"\x1b[>0;0;0c")
            except Exception:
                traceback.print_exc()
            self._term_query_buf = self._term_query_buf.replace(b"\x1b[>c", b"")
        if b"\x1b[?u" in self._term_query_buf:
            try:
                _write_all(fd, b"\x1b[?1u")
            except Exception:
                traceback.print_exc()
            self._term_query_buf = self._term_query_buf.replace(b"\x1b[?u", b"")
        if b"\x1b]10;?\x1b\\" in self._term_query_buf:
            try:
                _write_all(fd, b"\x1b]10;rgb:c0c0/c0c0/c0c0\x1b\\")
            except Exception:
                traceback.print_exc()
            self._term_query_buf = self._term_query_buf.replace(b"\x1b]10;?\x1b\\", b"")
        if b"\x1b]11;?\x1b\\" in self._term_query_buf:
            try:
                _write_all(fd, b"\x1b]11;rgb:0000/0000/0000\x1b\\")
            except Exception:
                traceback.print_exc()
            self._term_query_buf = self._term_query_buf.replace(b"\x1b]11;?\x1b\\", b"")

    def _pty_to_stdout(self) -> None:
        st = self.state
        if not st:
            return
        fd = st.pty_master_fd
        out_fd = sys.stdout.fileno()
        while not self._stop.is_set():
            try:
                b = os.read(fd, 4096)
                if not b:
                    break
                _write_all(out_fd, b)
                self._maybe_reply_to_terminal_queries(fd=fd, b=b)
                s = b.decode("utf-8", errors="replace")
                if s:
                    with self._lock:
                        st2 = self.state
                        if st2:
                            now_ts = _now()
                            _observe_shell_pre_exec_marker(st2, b, now_ts=now_ts)
                            visible = s.replace(SHELL_PRE_EXEC_MARKER, "")
                            st2.output_tail = (st2.output_tail + visible)[-st2.output_tail_max :]
                            _update_busy_from_pty_text(st2, visible, now_ts=now_ts)
                            cleaned = _strip_ansi(visible)
                            tail = st2.detach_trigger_tail
                            st2.detach_trigger_tail = (tail + cleaned)[-st2.detach_trigger_tail_max :]
                            if _maybe_detach_on_session_switch_trigger(st=st2, tail=tail, cleaned=cleaned, agent_backend=AGENT_BACKEND):
                                self._write_meta()
            except OSError:
                break

    def _startup_output_to_state(self, fd: int) -> None:
        try:
            while not self._stop.is_set():
                try:
                    b = os.read(fd, 4096)
                except OSError:
                    break
                if not b:
                    break
                s = b.decode("utf-8", errors="replace")
                if not s:
                    continue
                with self._lock:
                    st = self.state
                    if st and not st.shell_pre_exec_marker_seen and st.log_path is None:
                        st.output_tail = (st.output_tail + s)[-st.output_tail_max :]
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def _shell_startup_watchdog(self) -> None:
        if OWNER_TAG != "web" or SHELL_STARTUP_TIMEOUT_SECONDS <= 0:
            return
        deadline = _now() + SHELL_STARTUP_TIMEOUT_SECONDS
        while not self._stop.is_set():
            now_ts = _now()
            with self._lock:
                st = self.state
                if not st:
                    return
                if st.shell_pre_exec_marker_seen or st.log_path is not None:
                    return
                root_pid = int(st.codex_pid)
                start_ts = st.start_ts
                cwd = st.cwd
                output_tail = st.output_tail[-4000:]
            if now_ts >= deadline:
                _record_launch_attempt(
                    {
                        **_broker_launch_record(
                            stage="shell_startup",
                            error=f"{AGENT_BIN} shell startup did not reach agent exec within {SHELL_STARTUP_TIMEOUT_SECONDS:.1f}s",
                            cwd=cwd,
                            start_ts=start_ts,
                            agent_pid=root_pid,
                        ),
                        "pty_tail": output_tail,
                    }
                )
                with self._lock:
                    if self.state:
                        self.state.prelog_failure_recorded = True
                self._teardown_managed_process_group()
                return
            if not _process_group_alive(root_pid):
                return
            time.sleep(min(0.1, max(deadline - now_ts, 0.0)))

    def _stdin_to_pty(self) -> None:
        st = self.state
        if not st:
            return
        in_fd = sys.stdin.fileno()
        fd = st.pty_master_fd
        while not self._stop.is_set():
            try:
                b = os.read(in_fd, 4096)
                if not b:
                    with self._lock:
                        if self.state:
                            self.state.stdin_eof = True
                    self._stop.set()
                    break
                with self._lock:
                    st2 = self.state
                    if not st2:
                        continue
                    st2.last_local_input_ts = _now()
                    try:
                        _write_all(fd, b)
                    except OSError:
                        break
            except OSError:
                break

    def _log_watcher(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                st = self.state
                if not st or not st.log_path:
                    pass
                else:
                    log_path = st.log_path
                    off = st.log_off
            if not st or not st.log_path:
                time.sleep(0.25)
                continue

            objs, new_off = _read_jsonl_from_offset(log_path, off, max_bytes=256 * 1024)
            def maybe_drain_one_if_idle() -> None:
                fd: int | None = None
                kq: list[bytes] = []
                with self._lock:
                    st3 = self.state
                    if not st3:
                        return
                    if st3.busy or st3.turn_open or st3.pending_calls:
                        return
                    if not st3.key_queue:
                        return
                    fd = st3.pty_master_fd
                    if fd is None:
                        return
                    if st3.key_queue:
                        kq = st3.key_queue[:]
                        st3.key_queue.clear()
                for b in kq:
                    try:
                        _write_all(fd, b)
                    except Exception:
                        break

            def maybe_mark_idle() -> None:
                now_ts = _now()
                with self._lock:
                    st3 = self.state
                    if st3 and _should_clear_busy_state(st3, now_ts):
                        _mark_busy_state_idle(st3, now_ts)

            def maybe_clear_resume_delivery_mute() -> None:
                clear_meta = False
                with self._lock:
                    st3 = self.state
                    if st3 and st3.resume_session_id and (not st3.busy) and (not st3.turn_open) and (not st3.pending_calls):
                        st3.resume_session_id = None
                        clear_meta = True
                if clear_meta:
                    self._write_meta()

            with self._lock:
                st_check = self.state
                batch_still_current = bool(st_check and st_check.log_path is not None and _paths_match(st_check.log_path, log_path) and st_check.log_off == off)
            if not batch_still_current:
                continue

            if new_off == off:
                maybe_mark_idle()
                maybe_clear_resume_delivery_mute()
                maybe_drain_one_if_idle()
                time.sleep(0.25)
                continue

            with self._lock:
                st2 = self.state
                if not st2 or st2.log_path is None or (not _paths_match(st2.log_path, log_path)) or st2.log_off != off:
                    continue
                for obj in objs:
                    now_ts = _now()
                    token_update = _pi_token_update(obj)
                    if token_update is not None:
                        st2.token = token_update
                    if obj.get("type") == "event_msg":
                        p = obj.get("payload")
                        if not isinstance(p, dict):
                            raise ValueError("invalid rollout event_msg payload")
                        pt = p.get("type")
                        if pt == "token_count":
                            info = p.get("info")
                            if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
                                ctx = info.get("model_context_window")
                                last = info.get("last_token_usage")
                                if isinstance(ctx, int) and isinstance(last, dict):
                                    tt = last.get("total_tokens")
                                    if isinstance(tt, int):
                                        token_update = _pi_context_token_update(
                                            context_window=ctx,
                                            tokens_in_context=tt,
                                            as_of=obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None,
                                        )
                                        st2.token = token_update
                    _apply_rollout_obj_to_state(st2, obj, now_ts=now_ts)
                st2.log_off = new_off

            maybe_mark_idle()
            maybe_clear_resume_delivery_mute()
            maybe_drain_one_if_idle()

    def _write_meta(self) -> None:
        st = self.state
        if not st or not st.sock_path:
            return
        meta = {
            "session_id": st.session_id,
            "owner": OWNER_TAG if OWNER_TAG else None,
            "broker_pid": os.getpid(),
            "sessiond_pid": os.getpid(),
            "codex_pid": st.codex_pid,
            "cwd": st.cwd,
            "start_ts": st.start_ts,
            "log_path": str(st.log_path) if st.log_path else None,
            "ignored_rollout_paths": sorted(str(p) for p in st.ignored_rollout_paths),
            "sock_path": str(st.sock_path),
            "agent_backend": AGENT_BACKEND,
            "launch_id": (os.environ.get("CODEX_WEB_LAUNCH_ID") or "").strip() or None,
            "resume_session_id": st.resume_session_id,
            "model_provider": MODEL_PROVIDER_OVERRIDE or None,
            "preferred_auth_method": PREFERRED_AUTH_METHOD_OVERRIDE or None,
            "model": MODEL_OVERRIDE or None,
            "reasoning_effort": REASONING_EFFORT_OVERRIDE or None,
            "service_tier": SERVICE_TIER_OVERRIDE or None,
            "transport": (os.environ.get("CODEX_WEB_TRANSPORT") or "").strip() or None,
            "tmux_session": (os.environ.get("CODEX_WEB_TMUX_SESSION") or "").strip() or None,
            "tmux_window": (os.environ.get("CODEX_WEB_TMUX_WINDOW") or "").strip() or None,
            "spawn_nonce": (os.environ.get("CODEX_WEB_SPAWN_NONCE") or "").strip() or None,
            "control_protocol_version": 2,
            "control_capabilities": {"sync_send": True, "key_write_errors": True},
        }
        meta_path = st.sock_path.with_suffix(".json")
        SOCK_DIR.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        os.chmod(meta_path, 0o600)

    def _sock_server(self) -> None:
        st = self.state
        if not st or not st.sock_path:
            return
        SOCK_DIR.mkdir(parents=True, exist_ok=True)
        if st.sock_path.exists():
            st.sock_path.unlink()
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(str(st.sock_path))
        os.chmod(st.sock_path, 0o600)
        s.listen(20)
        s.settimeout(0.5)

        while not self._stop.is_set():
            try:
                conn, _ = s.accept()
            except socket.timeout:
                continue
            except Exception:
                sys.stderr.write(f"error: broker socket server crashed: {traceback.format_exc()}\n")
                try:
                    self._teardown_managed_process_group()
                except Exception:
                    traceback.print_exc()
                break
            threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()

        s.close()

    def _handle_conn(self, conn: socket.socket) -> None:
        def state_handler(_req: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            with self._lock:
                st = self.state
                if not st:
                    return {"error": "no state"}, None
                return {
                    "busy": st.busy,
                    "queue_len": 0,
                    "token": st.token,
                    "interrupted_idle": (not st.busy) and st.last_interrupted_idle_ts > 0.0,
                }, None

        def tail_handler(_req: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            with self._lock:
                st = self.state
                return {"tail": st.output_tail if st else ""}, None

        def send_handler(req: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            text = req.get("text")
            if not isinstance(text, str) or not text.strip():
                return {"error": "text required"}, None
            seq_raw = req.get("enter_seq")
            seq = _seq_bytes(seq_raw) if isinstance(seq_raw, str) else _encode_enter()
            sync_commit = bool(req.get("sync"))
            fd: int | None = None
            with self._lock:
                st = self.state
                if not st:
                    return {"error": "no state"}, None
                now_ts = _now()
                prev_busy = st.busy
                prev_turn_open = st.turn_open
                prev_turn_has_completion_candidate = st.turn_has_completion_candidate
                prev_last_interrupt_hint_ts = st.last_interrupt_hint_ts
                prev_last_interrupt_request_ts = st.last_interrupt_request_ts
                prev_last_interrupted_idle_ts = st.last_interrupted_idle_ts
                prev_last_turn_activity_ts = st.last_turn_activity_ts
                st.pending_calls.clear()
                st.busy = True
                st.turn_open = True
                st.turn_has_completion_candidate = False
                st.last_interrupt_hint_ts = 0.0
                st.last_interrupt_request_ts = 0.0
                st.last_interrupted_idle_ts = 0.0
                if now_ts > st.last_turn_activity_ts:
                    st.last_turn_activity_ts = now_ts
                fd = st.pty_master_fd
            def restore_state_after_inject_failure() -> None:
                with self._lock:
                    if self.state is st:
                        st.busy = prev_busy
                        st.turn_open = prev_turn_open
                        st.turn_has_completion_candidate = prev_turn_has_completion_candidate
                        st.last_interrupt_hint_ts = prev_last_interrupt_hint_ts
                        st.last_interrupt_request_ts = prev_last_interrupt_request_ts
                        st.last_interrupted_idle_ts = prev_last_interrupted_idle_ts
                        st.last_turn_activity_ts = prev_last_turn_activity_ts

            if sync_commit:
                if fd is None:
                    restore_state_after_inject_failure()
                    return {"error": "no pty", "commit_unknown": False}, None
                try:
                    _inject(fd, text=text, suffix=seq)
                except Exception as e:
                    restore_state_after_inject_failure()
                    return {"error": str(e), "commit_unknown": True}, None
                return {"queued": False, "queue_len": 0}, None
            def after_reply() -> None:
                if fd is None:
                    restore_state_after_inject_failure()
                    return
                try:
                    _inject(fd, text=text, suffix=seq)
                except Exception:
                    restore_state_after_inject_failure()
                    traceback.print_exc()
            return {"queued": False, "queue_len": 0}, after_reply

        def keys_handler(req: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            seq_raw = req.get("seq")
            if not isinstance(seq_raw, str) or not seq_raw:
                return {"error": "seq required"}, None
            b = _seq_bytes(seq_raw)
            mark_interrupt = req.get("interrupt") is True and b == b"\x1b"
            fd: int | None = None
            with self._lock:
                st = self.state
                if not st:
                    return {"error": "no state"}, None
                fd = st.pty_master_fd
                resp = {"ok": True, "queued": False, "n": len(b), "key_queue_len": len(st.key_queue)}
            wrote_keys = False
            if fd is not None:
                try:
                    _write_all(fd, b)
                    wrote_keys = True
                except Exception as e:
                    return {"error": str(e), "queued": False, "n": 0, "key_queue_len": 0, "commit_unknown": True}, None
            if mark_interrupt and wrote_keys:
                with self._lock:
                    if self.state is st:
                        _mark_explicit_interrupt_request(st, _now())
            return resp, None

        def shutdown_handler(_req: dict[str, Any]) -> tuple[dict[str, Any], Any]:
            return {"ok": True}, self._teardown_managed_process_group

        _handle_control_socket_connection(
            conn,
            handlers={
                "state": state_handler,
                "tail": tail_handler,
                "send": send_handler,
                "keys": keys_handler,
                "shutdown": shutdown_handler,
            },
            send_json_line=_send_socket_json_line,
            socket_peer_disconnected=_socket_peer_disconnected,
        )

    def _session_id_from_rollout_path(self, log_path: Path) -> str | None:
        # Codex stores rollout logs under date-based directories (e.g. ~/.codex/sessions/2026/01/22/rollout-...-<id>.jsonl),
        # so path components are not a stable session id. Extract the id from the filename.
        return _session_id_from_rollout_path(log_path)

    def _maybe_register_or_switch_rollout(self, *, log_path: Path) -> None:
        binding = _resolve_broker_log_binding(
            log_path=log_path,
            sessions_dir=self.sessions_dir,
            agent_backend=AGENT_BACKEND,
            session_id_from_rollout_path=self._session_id_from_rollout_path,
        )
        if binding is None:
            return
        seed = _seed_broker_log_state(log_path=binding.log_path, agent_backend=AGENT_BACKEND)

        with self._lock:
            st = self.state
            if not st:
                return
            result = _apply_broker_log_binding_to_state(st, binding=binding, seed=seed)
            if result is None:
                return

        if not result.have_sock:
            try:
                self._register_from_log(log_path=binding.log_path)
            except Exception:
                _dprint(f"broker: register_from_rollout failed: {traceback.format_exc()}")
                return
        elif result.previous_log_path is None or not _paths_match(result.previous_log_path, binding.log_path):
            self._write_meta()

    def run(self) -> int:
        rows, cols = _term_size()
        _require_proc()

        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        start_ts = _now()
        prelaunch_rollout_paths: set[Path] = set()
        if AGENT_BACKEND in ("pi", "cc"):
            prelaunch_rollout_paths = set(_iter_session_logs(self.sessions_dir, agent_backend=AGENT_BACKEND))
        headless = (OWNER_TAG == "web")
        local_terminal = (not self._emulate_terminal) and sys.stdin.isatty()

        startup_pipe_read: int | None = None
        pty_slave_fd: int | None = None
        try:
            if headless:
                master_fd, pty_slave_fd = pty.openpty()
                pty_slave_path = os.ttyname(pty_slave_fd)
                startup_pipe_read, startup_pipe_write = os.pipe()
                pid = os.fork()
                if pid == 0:
                    try:
                        _set_pdeathsig(signal.SIGHUP)
                        os.setsid()
                        os.close(startup_pipe_read)
                        os.close(master_fd)
                        devnull_fd = os.open(os.devnull, os.O_RDONLY)
                        try:
                            os.dup2(devnull_fd, 0)
                        finally:
                            if devnull_fd > 2:
                                os.close(devnull_fd)
                        os.dup2(startup_pipe_write, 1)
                        os.dup2(startup_pipe_write, 2)
                        if startup_pipe_write > 2:
                            os.close(startup_pipe_write)
                        if pty_slave_fd > 2:
                            os.close(pty_slave_fd)
                        term_raw = os.environ.get("TERM")
                        term = str(term_raw).strip() if term_raw is not None else ""
                        if not term:
                            term = "xterm-256color"
                        os.environ.setdefault("TERM", term)
                        os.environ["COLUMNS"] = str(cols)
                        os.environ["LINES"] = str(rows)
                        os.environ[BACKEND.home_env_var] = str(self.codex_home)
                        _exec_agent_via_login_shell(
                            cwd=self.cwd,
                            agent_args=self.codex_args,
                            pty_slave_path=pty_slave_path,
                        )
                    except Exception:
                        traceback.print_exc()
                        os._exit(127)
                os.close(startup_pipe_write)
            else:
                pid, master_fd = pty.fork()
                if pid == 0:
                    try:
                        _set_pdeathsig(signal.SIGHUP)
                        term_raw = os.environ.get("TERM")
                        term = str(term_raw).strip() if term_raw is not None else ""
                        if not term:
                            term = "xterm-256color"
                        os.environ.setdefault("TERM", term)
                        os.environ["COLUMNS"] = str(cols)
                        os.environ["LINES"] = str(rows)
                        os.environ[BACKEND.home_env_var] = str(self.codex_home)
                        if sys.stdin.isatty():
                            try:
                                fd = sys.stdin.fileno()
                                attrs = termios.tcgetattr(fd)
                                attrs[0] &= ~(termios.ICRNL | termios.INLCR | termios.IGNCR)
                                termios.tcsetattr(fd, termios.TCSANOW, attrs)
                            except (OSError, termios.error):
                                if DEBUG:
                                    traceback.print_exc()
                        os.environ[BACKEND.home_env_var] = str(self.codex_home)
                        _exec_agent(cwd=self.cwd, agent_args=self.codex_args)
                    except Exception:
                        traceback.print_exc()
                        os._exit(127)
        except Exception as e:
            _record_launch_attempt(
                _broker_launch_record(
                    stage="pty_fork",
                    error=f"pty setup failed before agent start: {type(e).__name__}: {e}",
                    cwd=self.cwd,
                    start_ts=start_ts,
                )
            )
            raise

        if local_terminal:
            try:
                fd = sys.stdin.fileno()
                self._stdin_termios = termios.tcgetattr(fd)
                tty.setraw(fd)
            except Exception:
                traceback.print_exc()
                self._stdin_termios = None

        try:
            _set_winsize(master_fd, rows, cols)
        except Exception:
            traceback.print_exc()

        st = State(
            codex_pid=pid,
            pty_master_fd=master_fd,
            cwd=self.cwd,
            start_ts=start_ts,
            codex_home=self.codex_home,
            sessions_dir=self.sessions_dir,
            busy=False,
            resume_session_id=self._resume_session_id,
        )
        declared_log_path = _session_log_path_from_args(args=self.codex_args, agent_backend=AGENT_BACKEND, sessions_dir=self.sessions_dir)
        st.declared_log_path = declared_log_path
        if AGENT_BACKEND in ("pi", "cc"):
            st.known_rollout_paths = set(prelaunch_rollout_paths)
        st.sock_path = SOCK_DIR / f"broker-{os.getpid()}.sock"
        if declared_log_path is not None:
            st.log_path = declared_log_path
            if declared_log_path.exists():
                try:
                    st.log_off = int(declared_log_path.stat().st_size)
                except Exception:
                    st.log_off = 0
            else:
                st.log_off = 0
        self.state = st
        if declared_log_path is not None and declared_log_path.exists():
            self._maybe_register_or_switch_rollout(log_path=declared_log_path)

        def _sigwinch(_signo: int, _frame: Any) -> None:
            try:
                r, c = _term_size()
                _set_winsize(master_fd, r, c)
            except Exception:
                traceback.print_exc()

        signal.signal(signal.SIGWINCH, _sigwinch)

        self._write_meta()
        threading.Thread(target=self._sock_server, daemon=True).start()
        threading.Thread(target=self._pty_to_stdout, daemon=True).start()
        if startup_pipe_read is not None:
            startup_fd = startup_pipe_read
            threading.Thread(target=lambda: self._startup_output_to_state(startup_fd), daemon=True).start()
        threading.Thread(target=self._shell_startup_watchdog, daemon=True).start()
        # Web-owned sessions launched inside tmux still have a real terminal and must
        # forward local pane input like a normal broker session.
        if local_terminal:
            threading.Thread(target=self._stdin_to_pty, daemon=True).start()
        threading.Thread(target=self._log_watcher, daemon=True).start()
        threading.Thread(target=self._discover_log_watcher, daemon=True).start()

        exit_code = 0
        try:
            while not self._stop.is_set():
                try:
                    wpid, status = os.waitpid(pid, os.WNOHANG)
                    if wpid == pid:
                        if os.WIFEXITED(status):
                            exit_code = int(os.WEXITSTATUS(status))
                        elif os.WIFSIGNALED(status):
                            exit_code = 128 + int(os.WTERMSIG(status))
                        break
                except ChildProcessError:
                    break
                time.sleep(0.1)
        finally:
            if self._stdin_termios is not None:
                try:
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, self._stdin_termios)
                except Exception:
                    traceback.print_exc()
                self._stdin_termios = None

        self._stop.set()
        try:
            os.close(master_fd)
        except Exception:
            traceback.print_exc()
        if pty_slave_fd is not None:
            try:
                os.close(pty_slave_fd)
            except Exception:
                traceback.print_exc()
        with self._lock:
            st2 = self.state
        if st2 and OWNER_TAG == "web" and st2.log_path is None and not st2.prelog_failure_recorded:
            _record_launch_attempt(
                {
                    **_broker_launch_record(
                        stage="agent_exit_before_log_bind",
                        error=f"{AGENT_BIN} exited with status {exit_code} before a session log was bound",
                        cwd=st2.cwd,
                        start_ts=st2.start_ts,
                        agent_pid=st2.codex_pid,
                        log_path=st2.log_path,
                        exit_code=exit_code,
                    ),
                    "agent_exit_status": exit_code,
                    "broker_exit_status": exit_code,
                    "pty_tail": st2.output_tail[-4000:],
                }
            )
        if st2 and st2.sock_path:
            try:
                st2.sock_path.unlink()
            except Exception:
                traceback.print_exc()
            try:
                st2.sock_path.with_suffix(".json").unlink()
            except Exception:
                traceback.print_exc()
        return exit_code


def main() -> None:
    _require_proc()
    ap = argparse.ArgumentParser(
        description="Foreground PTY broker for Codoxear CLI agents: preserves terminal UX and registers a control socket."
    )
    ap.add_argument("--cwd", default=os.getcwd(), help="Directory to run the agent in (default: current directory)")
    ap.add_argument("args", nargs=argparse.REMAINDER, help="Arguments after -- are passed to the selected agent CLI")
    ns = ap.parse_args()

    args = list(ns.args)
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        args = []

    b = Broker(cwd=_expand_cwd(str(ns.cwd)), codex_args=args)
    raise SystemExit(b.run())


if __name__ == "__main__":
    main()
