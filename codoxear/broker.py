#!/usr/bin/env python3
from __future__ import annotations

import argparse
import codecs
import json
import os
import pty
import pwd
import re
import signal
import socket
import sys
import termios
import threading
import time
import traceback
import tty
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codoxear.constants import CONTEXT_WINDOW_BASELINE_TOKENS
from codoxear import proc_fd_scan as _proc_fd_scan
from codoxear import pty_util as _pty_util
from codoxear.util import default_app_dir as _default_app_dir
from codoxear.util import find_session_log_for_session_id as _find_session_log_for_session_id
from codoxear.util import is_subagent_session_meta as _is_subagent_session_meta
from codoxear.util import read_session_meta_payload as _read_session_meta_payload
from codoxear.util import subagent_parent_thread_id as _subagent_parent_thread_id


APP_DIR = _default_app_dir()
SOCK_DIR = APP_DIR / "socks"
PROC_ROOT = Path("/proc")

CODEX_BIN = os.environ.get("CODEX_BIN", "codex")
OWNER_TAG = os.environ.get("CODEX_WEB_OWNER", "")
_CODEX_HOME_ENV = os.environ.get("CODEX_HOME")
if _CODEX_HOME_ENV is None or (not _CODEX_HOME_ENV.strip()):
    DEFAULT_CODEX_HOME = Path.home() / ".codex"
else:
    DEFAULT_CODEX_HOME = Path(_CODEX_HOME_ENV)
DEBUG = os.environ.get("CODEX_WEB_BROKER_DEBUG", "0") == "1"
FD_POLL_SECONDS_RAW = os.environ.get("CODEX_WEB_FD_POLL_SECONDS", "1.0")
_BUSY_QUIET_RAW = os.environ.get("CODEX_WEB_BUSY_QUIET_SECONDS")
if _BUSY_QUIET_RAW is None or (not _BUSY_QUIET_RAW.strip()):
    _BUSY_QUIET_RAW = "3.0"
BUSY_QUIET_SECONDS = max(float(_BUSY_QUIET_RAW), 0.0)

_BUSY_INTERRUPT_GRACE_RAW = os.environ.get("CODEX_WEB_BUSY_INTERRUPT_GRACE_SECONDS")
if _BUSY_INTERRUPT_GRACE_RAW is None or (not _BUSY_INTERRUPT_GRACE_RAW.strip()):
    _BUSY_INTERRUPT_GRACE_RAW = "3.0"
BUSY_INTERRUPT_GRACE_SECONDS = max(float(_BUSY_INTERRUPT_GRACE_RAW), 0.0)

INTERRUPT_HINT_TAIL_MAX = 4096

_SESSION_ID_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)
_ANSI_OSC_RE = re.compile("\x1B\\][^\x07]*(?:\x07|\x1B\\\\)")
_ANSI_CSI_RE = re.compile("\x1B(?:[@-Z\\-_]|\\[[0-?]*[ -/]*[@-~])")


def _dprint(msg: str) -> None:
    if not DEBUG:
        return
    sys.stderr.write(msg.rstrip("\n") + "\n")
    sys.stderr.flush()


def _now() -> float:
    return time.time()


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


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def _require_proc() -> None:
    if not sys.platform.startswith("linux"):
        sys.stderr.write("error: codoxear-broker requires linux (/proc, pty, termios).\n")
        raise SystemExit(2)
    if not (PROC_ROOT / "self" / "fd").is_dir():
        sys.stderr.write("error: codoxear-broker requires /proc (missing /proc/self/fd).\n")
        raise SystemExit(2)


def _proc_children_pids(proc_root: Path, pid: int) -> list[int]:
    return _proc_fd_scan._proc_children_pids(proc_root, pid)


def _proc_descendants(proc_root: Path, root_pid: int) -> set[int]:
    return _proc_fd_scan._proc_descendants(proc_root, root_pid)

def _proc_descendants_owned(proc_root: Path, root_pid: int, uid: int) -> set[int]:
    return _proc_fd_scan._proc_descendants_owned(proc_root, root_pid, uid)

def _proc_pid_uid(proc_root: Path, pid: int) -> int | None:
    return _proc_fd_scan._proc_pid_uid(proc_root, pid)


def _fd_is_writable(proc_root: Path, pid: int, fd: int) -> bool:
    return _proc_fd_scan._fd_is_writable(proc_root, pid, fd)


def _rollout_path_from_fd_link(link: str) -> Path | None:
    return _proc_fd_scan._rollout_path_from_fd_link(link)


def _iter_writable_rollout_paths(
    *,
    proc_root: Path,
    pid: int,
    sessions_dir: Path,
) -> list[Path]:
    return _proc_fd_scan._iter_writable_rollout_paths(proc_root=proc_root, pid=pid, sessions_dir=sessions_dir)


def _user_shell() -> str:
    sh = os.environ.get("SHELL")
    if isinstance(sh, str) and sh.strip():
        return sh.strip()
    try:
        return pwd.getpwuid(os.getuid()).pw_shell
    except Exception:
        return "/bin/zsh"


def _shell_argv_for_command(cmd: str) -> list[str]:
    shell = _user_shell()
    base = Path(shell).name
    if base not in ("zsh", "bash", "fish"):
        sys.stderr.write(f"error: unsupported login shell for PTY launch: {shell}\n")
        raise SystemExit(2)
    # -l: login (read profile); -i: interactive (read rc); -c: run command; command begins with exec to avoid wrapper processes.
    return [shell, "-l", "-i", "-c", cmd]


def _exec_codex(*, cwd: str, codex_args: list[str]) -> None:
    argv = [CODEX_BIN, *codex_args]
    os.chdir(cwd)
    os.execvpe(argv[0], argv, os.environ)

def _exec_codex_via_login_shell(*, cwd: str, codex_args: list[str]) -> None:
    q = shlex.quote
    argv = [CODEX_BIN, *codex_args]
    cmd = "exec " + " ".join(q(x) for x in argv)
    shell_argv = _shell_argv_for_command(cmd)
    os.chdir(cwd)
    os.execvpe(shell_argv[0], shell_argv, os.environ)


def _context_percent_remaining(*, tokens_in_context: int, context_window: int) -> int:
    if context_window <= CONTEXT_WINDOW_BASELINE_TOKENS:
        return 0
    effective = context_window - CONTEXT_WINDOW_BASELINE_TOKENS
    used = max(tokens_in_context - CONTEXT_WINDOW_BASELINE_TOKENS, 0)
    remaining = max(effective - used, 0)
    return int(round((remaining / effective) * 100.0))


def _enter_seq_bytes() -> bytes:
    return _seq_bytes(os.environ.get("CODEX_WEB_ENTER_SEQ", "\r"))


def _seq_bytes(raw: str) -> bytes:
    t = raw.strip().upper()
    if t in ("NONE", "EMPTY", "NOENTER", "NO_ENTER"):
        return b""
    if t in ("ESC", "ESCAPE"):
        return b"\x1b"
    if t in ("ENTER", "CR"):
        return b"\r"
    if t in ("LF",):
        return b"\n"
    if t in ("CRLF",):
        return b"\r\n"
    try:
        # Accept escape-like strings from env vars, e.g. "\\r", "\\n", "\\x0d".
        decoded = codecs.decode(raw.encode("utf-8"), "unicode_escape")
        b = decoded.encode("utf-8")
        return b if b else b"\r"
    except Exception:
        return b"\r"


def _encode_enter() -> bytes:
    b = _enter_seq_bytes()
    if DEBUG:
        _dprint(f"broker: enter_seq={b!r}")
    return b


def _inject(fd: int, *, text: str, suffix: bytes, delay_s: float = 0.2) -> None:
    os.write(fd, text.encode("utf-8"))
    if not suffix:
        return
    for _i in range(3):
        time.sleep(delay_s)
        os.write(fd, suffix)


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    _pty_util.set_winsize(fd, rows, cols)


def _term_size() -> tuple[int, int]:
    try:
        sz = os.get_terminal_size(sys.stdin.fileno())
        return int(sz.lines), int(sz.columns)
    except Exception:
        return 40, 120

def _paths_match(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except Exception:
        try:
            return a.absolute() == b.absolute()
        except Exception:
            return str(a) == str(b)


def _maybe_detach_on_new_session_hint(*, st: "State", tail: str, cleaned: str) -> bool:
    # Codex TUI prints this line when closing a thread on /new and similar flows.
    # The line can be split across PTY reads and may include ANSI escape sequences.
    phrase = "To continue this session, run "
    if not _hint_seen_in_new_text(tail=tail, cleaned=cleaned, phrase=phrase):
        return False
    for p in (st.log_path, st.last_rollout_path, st.last_detected_rollout_path):
        if p is not None:
            st.ignored_rollout_paths.add(p)
    st.log_path = None
    st.session_id = None
    st.log_off = 0
    st.last_rollout_path = None
    st.last_detected_rollout_path = None
    st.new_session_hint_tail = ""
    return True


def _read_jsonl_from_offset(path: Path, offset: int, max_bytes: int = 256 * 1024) -> tuple[list[dict[str, Any]], int]:
    try:
        with path.open("rb") as f:
            f.seek(offset)
            data = f.read(max_bytes)
            new_off = f.tell()
    except FileNotFoundError:
        return [], offset

    lines = data.splitlines()
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out, new_off


def _strip_ansi(text: str) -> str:
    return _ANSI_CSI_RE.sub("", _ANSI_OSC_RE.sub("", text))


def _hint_seen_in_new_text(*, tail: str, cleaned: str, phrase: str) -> bool:
    low_cleaned = cleaned.lower()
    low_phrase = phrase.lower()
    if low_phrase in low_cleaned:
        return True
    overlap = max(len(low_phrase) - 1, 0)
    if overlap <= 0:
        return False
    stitched = tail[-overlap:].lower() + low_cleaned
    pos = stitched.find(low_phrase)
    if pos < 0:
        return False
    return (pos + len(low_phrase)) > overlap


def _interrupt_hint_seen_in_new_text(*, tail: str, cleaned: str) -> bool:
    return _hint_seen_in_new_text(tail=tail, cleaned=cleaned, phrase="esc to interrupt")


def _compacting_hint_seen_in_new_text(*, tail: str, cleaned: str) -> bool:
    return (
        _hint_seen_in_new_text(tail=tail, cleaned=cleaned, phrase="compacting context")
        or _hint_seen_in_new_text(tail=tail, cleaned=cleaned, phrase="compacting conversation")
    )


def _update_busy_from_pty_text(st: "State", text: str, now_ts: float) -> None:
    cleaned = _strip_ansi(text)
    if not cleaned:
        return
    tail = st.interrupt_hint_tail
    st.interrupt_hint_tail = (st.interrupt_hint_tail + cleaned)[-st.interrupt_hint_tail_max :]
    if _interrupt_hint_seen_in_new_text(tail=tail, cleaned=cleaned):
        st.busy = True
        st.last_interrupt_hint_ts = now_ts
        if now_ts > st.last_turn_activity_ts:
            st.last_turn_activity_ts = now_ts
        return
    if _compacting_hint_seen_in_new_text(tail=tail, cleaned=cleaned):
        st.busy = True
        if now_ts > st.last_turn_activity_ts:
            st.last_turn_activity_ts = now_ts
        return


def _response_call_started(payload: dict[str, Any]) -> str | None:
    t = payload.get("type")
    if t not in ("function_call", "custom_tool_call"):
        return None
    call_id = payload.get("call_id")
    return call_id if isinstance(call_id, str) and call_id else None


def _response_call_finished(payload: dict[str, Any]) -> str | None:
    t = payload.get("type")
    if t not in ("function_call_output", "custom_tool_call_output"):
        return None
    call_id = payload.get("call_id")
    return call_id if isinstance(call_id, str) and call_id else None


def _should_clear_busy_state(st: "State", now_ts: float) -> bool:
    if not st.busy:
        return False
    if st.queue or st.key_queue:
        return False
    if st.pending_calls:
        return False
    if st.turn_open and (not st.turn_has_completion_candidate):
        return False
    if st.last_interrupt_hint_ts > 0.0 and (now_ts - st.last_interrupt_hint_ts) < BUSY_INTERRUPT_GRACE_SECONDS:
        return False
    if st.last_turn_activity_ts <= 0.0:
        return False
    return (now_ts - st.last_turn_activity_ts) >= BUSY_QUIET_SECONDS


def _reopen_turn_on_activity(st: "State") -> None:
    if st.turn_open:
        return
    st.turn_open = True
    st.turn_has_completion_candidate = False


def _apply_rollout_obj_to_state(st: "State", obj: dict[str, Any], now_ts: float) -> None:
    typ = obj.get("type")

    if typ == "event_msg":
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            raise ValueError("invalid rollout event_msg payload")
        ev_type = payload.get("type")
        if ev_type == "user_message":
            msg = payload.get("message")
            if isinstance(msg, str) and msg.strip():
                st.pending_calls.clear()
                st.busy = True
                st.turn_open = True
                st.turn_has_completion_candidate = False
                st.last_interrupt_hint_ts = 0.0
                st.last_turn_activity_ts = now_ts
            return
        if ev_type in ("turn_aborted", "thread_rolled_back"):
            st.pending_calls.clear()
            st.busy = False
            st.turn_open = False
            st.turn_has_completion_candidate = False
            st.last_interrupt_hint_ts = 0.0
            st.last_turn_activity_ts = 0.0
            return
        if ev_type == "task_complete":
            st.pending_calls.clear()
            st.busy = False
            st.turn_open = False
            st.turn_has_completion_candidate = False
            st.last_interrupt_hint_ts = 0.0
            st.last_turn_activity_ts = 0.0
            return
        if ev_type == "agent_message":
            msg = payload.get("message")
            if isinstance(msg, str) and msg.strip() and st.turn_open:
                st.turn_has_completion_candidate = True
            st.busy = True
            st.last_turn_activity_ts = now_ts
            return
        if ev_type == "agent_reasoning":
            _reopen_turn_on_activity(st)
            if st.turn_open:
                st.turn_has_completion_candidate = False
            st.busy = True
            st.last_turn_activity_ts = now_ts
            return
        if ev_type == "token_count" and st.busy:
            st.last_turn_activity_ts = now_ts
            return
        return

    if typ != "response_item":
        return
    payload = obj.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("invalid rollout response_item payload")

    started = _response_call_started(payload)
    if started is not None:
        st.pending_calls.add(started)
        _reopen_turn_on_activity(st)
        if st.turn_open:
            st.turn_has_completion_candidate = False
        st.busy = True
        st.last_turn_activity_ts = now_ts
        return

    finished = _response_call_finished(payload)
    if finished is not None:
        st.pending_calls.discard(finished)
        _reopen_turn_on_activity(st)
        if st.turn_open:
            st.turn_has_completion_candidate = False
        st.busy = True
        st.last_turn_activity_ts = now_ts
        return

    item_type = payload.get("type")
    role = payload.get("role")
    if item_type in (
        "reasoning",
        "function_call",
        "function_call_output",
        "custom_tool_call",
        "custom_tool_call_output",
        "web_search_call",
        "local_shell_call",
    ):
        _reopen_turn_on_activity(st)
        if st.turn_open:
            st.turn_has_completion_candidate = False
        st.busy = True
        st.last_turn_activity_ts = now_ts
        return
    if item_type == "message" and role == "assistant":
        content = payload.get("content")
        if not isinstance(content, list):
            raise ValueError("invalid assistant message content")
        has_text = any(
            isinstance(part, dict)
            and part.get("type") == "output_text"
            and isinstance(part.get("text"), str)
            and part.get("text")
            for part in content
        )
        if has_text and st.turn_open:
            st.turn_has_completion_candidate = True
        st.busy = True
        st.last_turn_activity_ts = now_ts
        return


@dataclass
class State:
    codex_pid: int
    pty_master_fd: int
    cwd: str
    start_ts: float
    codex_home: Path
    sessions_dir: Path
    log_path: Path | None = None
    session_id: str | None = None
    sock_path: Path | None = None
    busy: bool = False
    stdin_eof: bool = False
    queue: list[str] = field(default_factory=list)
    key_queue: list[bytes] = field(default_factory=list)
    output_tail: str = ""
    output_tail_max: int = 256 * 1024
    log_off: int = 0
    last_local_input_ts: float = 0.0
    last_turn_activity_ts: float = 0.0
    last_interrupt_hint_ts: float = 0.0
    pending_calls: set[str] = field(default_factory=set)
    turn_open: bool = False
    turn_has_completion_candidate: bool = False
    interrupt_hint_tail: str = ""
    interrupt_hint_tail_max: int = INTERRUPT_HINT_TAIL_MAX
    new_session_hint_tail: str = ""
    new_session_hint_tail_max: int = 8192
    token: dict[str, Any] | None = None
    last_rollout_path: Path | None = None
    last_detected_rollout_path: Path | None = None
    ignored_rollout_paths: set[Path] = field(default_factory=set)


class Broker:
    def __init__(self, *, cwd: str, codex_args: list[str]) -> None:
        self.cwd = cwd
        self.codex_args = codex_args
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.state: State | None = None
        self._emulate_terminal = (os.environ.get("CODEX_WEB_EMULATE_TERMINAL", "0") == "1") or (not sys.stdin.isatty())
        self._term_query_buf = b""
        self._stdin_termios: list[Any] | None = None

        self.codex_home = DEFAULT_CODEX_HOME
        self.sessions_dir = self.codex_home / "sessions"

    def _register_from_log(self, *, log_path: Path) -> bool:
        sid = self._session_id_from_rollout_path(log_path)
        if sid is None:
            raise RuntimeError(f"unable to determine session_id from rollout filename: {log_path}")
        if not sid:
            _dprint(f"broker: register_from_log: no session id: {log_path}")
            return False

        try:
            off = log_path.stat().st_size
        except Exception:
            off = 0

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
                os.write(fd, b"\x1b[0n")
            except Exception:
                traceback.print_exc()
            self._term_query_buf = self._term_query_buf.replace(b"\x1b[5n", b"")
        if b"\x1b[6n" in self._term_query_buf:
            try:
                os.write(fd, b"\x1b[1;1R")
            except Exception:
                traceback.print_exc()
            self._term_query_buf = self._term_query_buf.replace(b"\x1b[6n", b"")
        if b"\x1b[c" in self._term_query_buf:
            try:
                os.write(fd, b"\x1b[?1;2c")
            except Exception:
                traceback.print_exc()
            self._term_query_buf = self._term_query_buf.replace(b"\x1b[c", b"")
        if b"\x1b[>c" in self._term_query_buf:
            try:
                os.write(fd, b"\x1b[>0;0;0c")
            except Exception:
                traceback.print_exc()
            self._term_query_buf = self._term_query_buf.replace(b"\x1b[>c", b"")
        if b"\x1b[?u" in self._term_query_buf:
            try:
                os.write(fd, b"\x1b[?1u")
            except Exception:
                traceback.print_exc()
            self._term_query_buf = self._term_query_buf.replace(b"\x1b[?u", b"")
        if b"\x1b]10;?\x1b\\" in self._term_query_buf:
            try:
                os.write(fd, b"\x1b]10;rgb:c0c0/c0c0/c0c0\x1b\\")
            except Exception:
                traceback.print_exc()
            self._term_query_buf = self._term_query_buf.replace(b"\x1b]10;?\x1b\\", b"")
        if b"\x1b]11;?\x1b\\" in self._term_query_buf:
            try:
                os.write(fd, b"\x1b]11;rgb:0000/0000/0000\x1b\\")
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
                os.write(out_fd, b)
                self._maybe_reply_to_terminal_queries(fd=fd, b=b)
                s = b.decode("utf-8", errors="replace")
                if s:
                    with self._lock:
                        st2 = self.state
                        if st2:
                            st2.output_tail = (st2.output_tail + s)[-st2.output_tail_max :]
                            _update_busy_from_pty_text(st2, s, now_ts=_now())
                            cleaned = _strip_ansi(s)
                            tail = st2.new_session_hint_tail
                            st2.new_session_hint_tail = (tail + cleaned)[-st2.new_session_hint_tail_max :]
                            if _maybe_detach_on_new_session_hint(st=st2, tail=tail, cleaned=cleaned):
                                self._write_meta()
            except OSError:
                break

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
                        os.write(fd, b)
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
            def drain_queues(*, clear_busy: bool) -> None:
                q: list[str] = []
                kq: list[bytes] = []
                fd: int | None = None
                with self._lock:
                    st3 = self.state
                    if not st3:
                        return
                    if clear_busy:
                        st3.busy = False
                    if not st3.queue and not st3.key_queue:
                        return
                    kq = st3.key_queue[:]
                    st3.key_queue.clear()
                    q = st3.queue[:]
                    st3.queue.clear()
                    fd = st3.pty_master_fd
                if fd is None:
                    return
                for b in kq:
                    try:
                        os.write(fd, b)
                    except Exception:
                        break
                for msg in q:
                    try:
                        _inject(fd, text=msg, suffix=_encode_enter())
                    except Exception:
                        break

            def maybe_mark_idle() -> None:
                now_ts = _now()
                should_clear = False
                with self._lock:
                    st3 = self.state
                    if st3 and _should_clear_busy_state(st3, now_ts):
                        st3.busy = False
                        st3.turn_open = False
                        st3.turn_has_completion_candidate = False
                        st3.last_turn_activity_ts = 0.0
                        st3.last_interrupt_hint_ts = 0.0
                        should_clear = bool(st3.queue or st3.key_queue)
                if should_clear:
                    drain_queues(clear_busy=False)

            if new_off == off:
                maybe_mark_idle()
                time.sleep(0.25)
                continue

            with self._lock:
                st2 = self.state
                if st2:
                    st2.log_off = new_off

            for obj in objs:
                now_ts = _now()
                aborted_or_rolled_back = False
                if obj.get("type") == "event_msg":
                    p = obj.get("payload")
                    if not isinstance(p, dict):
                        raise ValueError("invalid rollout event_msg payload")
                    pt = p.get("type")
                    if pt in ("turn_aborted", "thread_rolled_back"):
                        aborted_or_rolled_back = True
                    if pt == "token_count":
                        info = p.get("info")
                        if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
                            ctx = info.get("model_context_window")
                            last = info.get("last_token_usage")
                            if isinstance(ctx, int) and isinstance(last, dict):
                                tt = last.get("total_tokens")
                                if isinstance(tt, int):
                                    token_update = {
                                        "context_window": ctx,
                                        "tokens_in_context": tt,
                                        "tokens_remaining": max(ctx - tt, 0),
                                        "percent_remaining": _context_percent_remaining(tokens_in_context=tt, context_window=ctx),
                                        "baseline_tokens": CONTEXT_WINDOW_BASELINE_TOKENS,
                                        "as_of": obj.get("timestamp") if isinstance(obj.get("timestamp"), str) else None,
                                    }
                                    with self._lock:
                                        if self.state:
                                            self.state.token = token_update
                with self._lock:
                    st3 = self.state
                    if st3:
                        _apply_rollout_obj_to_state(st3, obj, now_ts=now_ts)
                if aborted_or_rolled_back:
                    drain_queues(clear_busy=False)

            maybe_mark_idle()

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
            "sock_path": str(st.sock_path),
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
                break
            threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()

        s.close()

    def _handle_conn(self, conn: socket.socket) -> None:
        try:
            f = conn.makefile("rwb")
            line = f.readline()
            if not line:
                return
            req = json.loads(line.decode("utf-8"))
            cmd = req.get("cmd")
            if cmd == "state":
                with self._lock:
                    st = self.state
                    if not st:
                        resp = {"error": "no state"}
                    else:
                        resp = {"busy": st.busy, "queue_len": len(st.queue), "token": st.token}
                f.write((json.dumps(resp) + "\n").encode("utf-8"))
                f.flush()
                return

            if cmd == "tail":
                with self._lock:
                    st = self.state
                    resp = {"tail": st.output_tail if st else ""}
                f.write((json.dumps(resp) + "\n").encode("utf-8"))
                f.flush()
                return

            if cmd == "send":
                text = req.get("text")
                if not isinstance(text, str) or not text.strip():
                    resp = {"error": "text required"}
                else:
                    seq_raw = req.get("enter_seq")
                    seq = _seq_bytes(seq_raw) if isinstance(seq_raw, str) else _encode_enter()
                    fd: int | None = None
                    with self._lock:
                        st = self.state
                        if not st:
                            resp = {"error": "no state"}
                        else:
                            fd = st.pty_master_fd
                            resp = {"queued": False, "queue_len": len(st.queue)}
                    if fd is not None:
                        _inject(fd, text=text, suffix=seq)
                f.write((json.dumps(resp) + "\n").encode("utf-8"))
                f.flush()
                return

            if cmd == "keys":
                seq_raw = req.get("seq")
                if not isinstance(seq_raw, str) or not seq_raw:
                    resp = {"error": "seq required"}
                else:
                    b = _seq_bytes(seq_raw)
                    fd: int | None = None
                    with self._lock:
                        st = self.state
                        if not st:
                            resp = {"error": "no state"}
                        else:
                            fd = st.pty_master_fd
                            resp = {"ok": True, "queued": False, "n": len(b), "key_queue_len": len(st.key_queue)}
                    if fd is not None:
                        try:
                            os.write(fd, b)
                        except Exception:
                            traceback.print_exc()
                f.write((json.dumps(resp) + "\n").encode("utf-8"))
                f.flush()
                return

            if cmd == "shutdown":
                self._stop.set()
                with self._lock:
                    st = self.state
                if st:
                    try:
                        os.killpg(st.codex_pid, signal.SIGTERM)
                    except Exception:
                        traceback.print_exc()
                        try:
                            os.kill(st.codex_pid, signal.SIGTERM)
                        except Exception:
                            traceback.print_exc()
                f.write(b'{"ok":true}\n')
                f.flush()
                return

            f.write(b'{"error":"unknown cmd"}\n')
            f.flush()
        except Exception:
            try:
                conn.sendall((json.dumps({"error": "exception", "trace": traceback.format_exc()}) + "\n").encode("utf-8"))
            except Exception:
                traceback.print_exc()
        finally:
            try:
                conn.close()
            except Exception:
                traceback.print_exc()

    def _session_id_from_rollout_path(self, log_path: Path) -> str | None:
        # Codex stores rollout logs under date-based directories (e.g. ~/.codex/sessions/2026/01/22/rollout-...-<id>.jsonl),
        # so path components are not a stable session id. Extract the id from the filename.
        name = log_path.name
        m = _SESSION_ID_RE.findall(name)
        return m[-1] if m else None

    def _maybe_register_or_switch_rollout(self, *, log_path: Path) -> None:
        try:
            lp = log_path.resolve()
        except Exception:
            lp = log_path
        try:
            lp.resolve().relative_to(self.sessions_dir.resolve())
        except Exception:
            return
        if lp.name.startswith("rollout-") and lp.name.endswith(".jsonl"):
            pass
        else:
            return

        payload = _read_session_meta_payload(lp, timeout_s=1.5)
        if not payload:
            return
        if _is_subagent_session_meta(payload):
            parent = _subagent_parent_thread_id(payload)
            if not parent:
                return
            parent_log = _find_session_log_for_session_id(self.sessions_dir, parent)
            if not parent_log:
                return
            parent_payload = _read_session_meta_payload(parent_log, timeout_s=0.2)
            if not parent_payload:
                return
            if _is_subagent_session_meta(parent_payload):
                return
            lp = parent_log
            payload = parent_payload

        sid = payload.get("id")
        if not isinstance(sid, str) or not sid:
            sid = self._session_id_from_rollout_path(lp)
            if sid is None:
                raise RuntimeError(f"unable to determine session_id from rollout filename: {lp}")
        if not sid:
            return

        with self._lock:
            st = self.state
            if not st:
                return
            last = st.last_rollout_path
            if last is not None and _paths_match(last, lp):
                return
            st.last_rollout_path = lp
            have_sock = st.sock_path is not None
            prev_lp = st.log_path
            st.session_id = sid
            st.log_path = lp
            try:
                st.log_off = int(lp.stat().st_size)
            except Exception:
                st.log_off = 0

        if not have_sock:
            try:
                self._register_from_log(log_path=lp)
            except Exception:
                _dprint(f"broker: register_from_rollout failed: {traceback.format_exc()}")
                return
        elif prev_lp is None or not _paths_match(prev_lp, lp):
            self._write_meta()

    def _proc_fd_watcher(self) -> None:
        try:
            poll_s = float(FD_POLL_SECONDS_RAW)
        except Exception:
            sys.stderr.write(f"error: invalid CODEX_WEB_FD_POLL_SECONDS={FD_POLL_SECONDS_RAW!r}\n")
            os.kill(os.getpid(), signal.SIGTERM)
            return
        if not (poll_s > 0.0):
            sys.stderr.write(f"error: invalid CODEX_WEB_FD_POLL_SECONDS={FD_POLL_SECONDS_RAW!r}\n")
            os.kill(os.getpid(), signal.SIGTERM)
            return
        try:
            while not self._stop.is_set():
                with self._lock:
                    st = self.state
                    if not st:
                        return
                    root_pid = int(st.codex_pid)
                    sessions_dir = st.sessions_dir
                    last_detected = st.last_detected_rollout_path
                    ignored = set(st.ignored_rollout_paths)
                    need_clear = bool(
                        st.log_path is not None
                        or st.session_id is not None
                        or st.last_rollout_path is not None
                        or st.last_detected_rollout_path is not None
                    )

                if root_pid <= 0:
                    time.sleep(poll_s)
                    continue

                uid = _proc_pid_uid(PROC_ROOT, root_pid)
                if uid is None:
                    time.sleep(poll_s)
                    continue
                pids = _proc_descendants_owned(PROC_ROOT, root_pid, int(uid))
                candidates: list[tuple[int, int, Path]] = []
                had_any_rollout = False
                for pid in pids:
                    # Defensive: the descendant PID can be reused between enumeration and scanning.
                    pid_uid = _proc_pid_uid(PROC_ROOT, int(pid))
                    if pid_uid is None or int(pid_uid) != int(uid):
                        continue
                    try:
                        paths = _iter_writable_rollout_paths(proc_root=PROC_ROOT, pid=pid, sessions_dir=sessions_dir)
                    except PermissionError as e:
                        sys.stderr.write(f"error: permission denied scanning /proc/{pid}/fd for owned pid (uid={uid}): {e}\n")
                        sys.stderr.flush()
                        os.kill(os.getpid(), signal.SIGTERM)
                        return
                    for p in paths:
                        had_any_rollout = True
                        if p in ignored:
                            continue
                        try:
                            mtime_ns = int(p.stat().st_mtime_ns)
                        except Exception:
                            continue
                        candidates.append((mtime_ns, int(pid), p))

                if not candidates:
                    if need_clear:
                        with self._lock:
                            st3 = self.state
                            if not st3:
                                return
                            st3.log_path = None
                            st3.session_id = None
                            st3.log_off = 0
                            st3.last_rollout_path = None
                            st3.last_detected_rollout_path = None
                            if not had_any_rollout:
                                st3.ignored_rollout_paths.clear()
                        self._write_meta()
                    time.sleep(poll_s)
                    continue

                candidates.sort(key=lambda t: (t[0], t[1], str(t[2])))
                _mtime_ns, _pid, best_path = candidates[-1]

                if last_detected is not None and _paths_match(last_detected, best_path):
                    time.sleep(poll_s)
                    continue

                with self._lock:
                    st2 = self.state
                    if not st2:
                        return
                    st2.last_detected_rollout_path = best_path

                self._maybe_register_or_switch_rollout(log_path=best_path)
                with self._lock:
                    st4 = self.state
                    if st4 and st4.log_path is not None:
                        st4.ignored_rollout_paths.clear()
                time.sleep(poll_s)
        except Exception:
            sys.stderr.write(f"error: proc fd watcher crashed: {traceback.format_exc()}\n")
            os.kill(os.getpid(), signal.SIGTERM)

    def run(self) -> int:
        rows, cols = _term_size()
        _require_proc()

        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        start_ts = _now()
        headless = (OWNER_TAG == "web")

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
                os.environ["CODEX_HOME"] = str(self.codex_home)
                if sys.stdin.isatty():
                    try:
                        fd = sys.stdin.fileno()
                        attrs = termios.tcgetattr(fd)
                        attrs[0] &= ~(termios.ICRNL | termios.INLCR | termios.IGNCR)
                        termios.tcsetattr(fd, termios.TCSANOW, attrs)
                    except (OSError, termios.error):
                        if DEBUG:
                            traceback.print_exc()
                if headless:
                    _exec_codex_via_login_shell(cwd=self.cwd, codex_args=self.codex_args)
                else:
                    _exec_codex(cwd=self.cwd, codex_args=self.codex_args)
            except Exception:
                traceback.print_exc()
                os._exit(127)

        if (not headless) and (not self._emulate_terminal) and sys.stdin.isatty():
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
        )
        st.sock_path = SOCK_DIR / f"broker-{os.getpid()}.sock"
        self.state = st

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
        if not headless:
            threading.Thread(target=self._stdin_to_pty, daemon=True).start()
        threading.Thread(target=self._log_watcher, daemon=True).start()
        threading.Thread(target=self._proc_fd_watcher, daemon=True).start()

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
        with self._lock:
            st2 = self.state
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
        description="Foreground PTY broker for codex: preserves terminal UX and registers a control socket for Codoxear."
    )
    ap.add_argument("--cwd", default=os.getcwd(), help="Directory to run codex in (default: current directory)")
    ap.add_argument("args", nargs=argparse.REMAINDER, help="Arguments after -- are passed to codex")
    ns = ap.parse_args()

    args = list(ns.args)
    if args and args[0] == "--":
        args = args[1:]
    if not args:
        args = []

    b = Broker(cwd=str(ns.cwd), codex_args=args)
    raise SystemExit(b.run())


if __name__ == "__main__":
    main()
