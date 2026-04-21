#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from codoxear.agent_backend import get_agent_backend
from codoxear.agent_backend import normalize_agent_backend
from codoxear.constants import CONTEXT_WINDOW_BASELINE_TOKENS
from codoxear.pi_log import pi_assistant_text as _pi_assistant_text
from codoxear.pi_log import pi_assistant_is_final_turn_end as _pi_assistant_is_final_turn_end
from codoxear.pi_log import pi_assistant_thinking_count as _pi_assistant_thinking_count
from codoxear.pi_log import pi_assistant_tool_use_count as _pi_assistant_tool_use_count
from codoxear.pi_log import pi_message_role as _pi_message_role
from codoxear.pi_log import pi_token_update as _pi_token_update
from codoxear.pi_log import pi_user_text as _pi_user_text
from codoxear import pty_util as _pty_util
from codoxear.util import default_app_dir as _default_app_dir
from codoxear.util import find_new_session_log as _find_new_session_log
from codoxear.util import find_session_log_for_session_id as _find_session_log_for_session_id
from codoxear.util import is_subagent_session_meta as _is_subagent_session_meta
from codoxear.util import iter_session_logs as _iter_session_logs
from codoxear.util import proc_find_open_rollout_log as _proc_find_open_rollout_log
from codoxear.util import read_session_meta_payload as _read_session_meta_payload
from codoxear.util import _send_socket_json_line as _send_socket_json_line
from codoxear.util import _socket_peer_disconnected as _socket_peer_disconnected
from codoxear.util import subagent_parent_thread_id as _subagent_parent_thread_id


APP_DIR = _default_app_dir()
SOCK_DIR = APP_DIR / "socks"
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

INTERRUPT_HINT_TAIL_MAX = 4096
_BRACKETED_PASTE_START = b"\x1b[200~"
_BRACKETED_PASTE_END = b"\x1b[201~"

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


def _resume_session_id_from_args(args: list[str]) -> str | None:
    if AGENT_BACKEND == "pi":
        for idx, token in enumerate(args):
            if token != "--session":
                continue
            if (idx + 1) >= len(args):
                return None
            resume_id = str(args[idx + 1] or "").strip()
            if not resume_id:
                return None
            if resume_id.endswith(".jsonl"):
                try:
                    payload = _read_session_meta_payload(Path(resume_id), agent_backend="pi", timeout_s=0.0)
                except Exception:
                    return None
                if isinstance(payload, dict):
                    sid = payload.get("id")
                    if isinstance(sid, str) and sid:
                        return sid
                return None
            return resume_id
        return None
    for idx, token in enumerate(args):
        if token != "resume":
            continue
        if (idx + 1) >= len(args):
            return None
        resume_id = str(args[idx + 1] or "").strip()
        return resume_id or None
    return None


def _session_log_path_from_args(*, args: list[str], agent_backend: str, sessions_dir: Path) -> Path | None:
    if normalize_agent_backend(agent_backend) != "pi":
        return None
    for idx, token in enumerate(args):
        if token != "--session":
            continue
        if (idx + 1) >= len(args):
            return None
        raw = str(args[idx + 1] or "").strip()
        if (not raw) or (not raw.endswith(".jsonl")):
            return None
        path = Path(raw).expanduser()
        try:
            resolved = path.resolve()
        except Exception:
            resolved = path
        try:
            resolved.relative_to(sessions_dir.resolve())
        except Exception:
            return None
        return resolved
    return None


def _pi_session_dir_name(cwd: str) -> str:
    normalized = cwd.lstrip("/\\").replace("/", "-").replace("\\", "-").replace(":", "-")
    return f"--{normalized}--"


def _pi_session_dir_from_args(*, args: list[str], cwd: str, sessions_dir: Path) -> Path | None:
    for idx, token in enumerate(args):
        if token == "--no-session":
            return None
        if token != "--session-dir":
            continue
        if (idx + 1) >= len(args):
            return None
        raw = str(args[idx + 1] or "").strip()
        if not raw:
            return None
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path(cwd) / path).resolve()
        return path
    return sessions_dir / _pi_session_dir_name(cwd)


def _pi_new_session_log_path(*, cwd: str, sessions_dir: Path) -> Path:
    session_dir = sessions_dir / _pi_session_dir_name(cwd)
    session_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    filename = f"{timestamp.replace(':', '-').replace('.', '-')}_{uuid.uuid4()}.jsonl"
    return session_dir / filename


def _ensure_pi_session_arg(*, args: list[str], cwd: str, sessions_dir: Path) -> list[str]:
    if AGENT_BACKEND != "pi":
        return list(args)
    out = list(args)
    session_dir = _pi_session_dir_from_args(args=out, cwd=cwd, sessions_dir=sessions_dir)
    if session_dir is None:
        return out
    for token in out:
        if token == "--session":
            return out
    if session_dir == sessions_dir / _pi_session_dir_name(cwd):
        log_path = _pi_new_session_log_path(cwd=cwd, sessions_dir=sessions_dir)
    else:
        session_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        filename = f"{timestamp.replace(':', '-').replace('.', '-')}_{uuid.uuid4()}.jsonl"
        log_path = session_dir / filename
    out.extend(["--session", str(log_path)])
    return out


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



def _process_group_alive(root_pid: int) -> bool:
    if root_pid <= 0:
        return False
    try:
        os.killpg(root_pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


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


def _expand_cwd(cwd: str) -> str:
    if not isinstance(cwd, str) or not cwd.strip():
        raise ValueError("cwd must be a non-empty string")
    home = str(Path.home())
    s = cwd.strip().replace("${HOME}", home)
    s = re.sub(r"\$HOME(?![A-Za-z0-9_])", home, s)
    return os.path.expanduser(os.path.expandvars(s))


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


def _exec_agent(*, cwd: str, agent_args: list[str]) -> None:
    argv = [AGENT_BIN, *agent_args]
    os.chdir(cwd)
    os.execvpe(argv[0], argv, os.environ)

def _exec_agent_via_login_shell(*, cwd: str, agent_args: list[str]) -> None:
    q = shlex.quote
    argv = [AGENT_BIN, *agent_args]
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
    b = _pty_util.seq_bytes(raw)
    return b if b else b"\r"


def _encode_enter() -> bytes:
    b = _enter_seq_bytes()
    if DEBUG:
        _dprint(f"broker: enter_seq={b!r}")
    return b


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError("short write to PTY")
        view = view[n:]


def _inject(fd: int, *, text: str, suffix: bytes, delay_s: float = 0.05) -> None:
    payload = _BRACKETED_PASTE_START + text.encode("utf-8") + _BRACKETED_PASTE_END
    _write_all(fd, payload)
    if not suffix:
        return
    if delay_s > 0:
        time.sleep(delay_s)
    _write_all(fd, suffix)


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


def _close_turn_state(st: "State") -> None:
    st.pending_calls.clear()
    st.busy = False
    st.turn_open = False
    st.turn_has_completion_candidate = False
    st.last_interrupt_hint_ts = 0.0
    st.last_turn_activity_ts = 0.0


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
            _close_turn_state(st)
            return
        if ev_type == "task_complete":
            _close_turn_state(st)
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

    if typ == "message":
        user_text = _pi_user_text(obj)
        if isinstance(user_text, str) and user_text:
            st.pending_calls.clear()
            st.busy = True
            st.turn_open = True
            st.turn_has_completion_candidate = False
            st.last_interrupt_hint_ts = 0.0
            st.last_turn_activity_ts = now_ts
            return

        role = _pi_message_role(obj)
        has_text = bool(_pi_assistant_text(obj))
        thinking_count = _pi_assistant_thinking_count(obj)
        tool_count = _pi_assistant_tool_use_count(obj)
        is_tool_result = role == "toolResult"

        if has_text and role == "assistant" and _pi_assistant_is_final_turn_end(obj):
            _close_turn_state(st)
            return

        if is_tool_result or tool_count > 0 or thinking_count > 0:
            _reopen_turn_on_activity(st)
            if st.turn_open:
                st.turn_has_completion_candidate = False
            st.busy = True
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
    detach_trigger_tail: str = ""
    detach_trigger_tail_max: int = 8192
    token: dict[str, Any] | None = None
    last_rollout_path: Path | None = None
    last_detected_rollout_path: Path | None = None
    ignored_rollout_paths: set[Path] = field(default_factory=set)
    known_rollout_paths: set[Path] = field(default_factory=set)
    resume_session_id: str | None = None


class Broker:
    def __init__(self, *, cwd: str, codex_args: list[str]) -> None:
        self.cwd = cwd
        base_args = _ensure_pi_session_arg(args=codex_args, cwd=self.cwd, sessions_dir=BACKEND.sessions_dir())
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
                    need = (current_log_path is None) or (not current_log_path.exists())
                    root_pid = int(st.codex_pid)
                    sock_path = st.sock_path
                    known_paths = set(st.known_rollout_paths)
                    ignored_paths = set(st.ignored_rollout_paths)
                if root_pid > 0:
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
                    if AGENT_BACKEND == "pi":
                        claimed_paths = _claimed_log_paths_from_sock_meta(sock_dir=SOCK_DIR, exclude_sock=sock_path)
                        discovered = _find_new_session_log(
                            sessions_dir=self.sessions_dir,
                            agent_backend="pi",
                            cwd=self.cwd,
                            after_ts=0.0,
                            preexisting=known_paths,
                            exclude_paths=claimed_paths | ignored_paths,
                            timeout_s=0.0,
                        )
                        if discovered is not None:
                            _sid, lp = discovered
                            if lp.exists():
                                if current_log_path is None or (not _paths_match(lp, current_log_path)):
                                    self._maybe_register_or_switch_rollout(log_path=lp)
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
                            st2.output_tail = (st2.output_tail + s)[-st2.output_tail_max :]
                            _update_busy_from_pty_text(st2, s, now_ts=_now())
                            cleaned = _strip_ansi(s)
                            tail = st2.detach_trigger_tail
                            st2.detach_trigger_tail = (tail + cleaned)[-st2.detach_trigger_tail_max :]
                            if _maybe_detach_on_session_switch_trigger(st=st2, tail=tail, cleaned=cleaned, agent_backend=AGENT_BACKEND):
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
                        st3.busy = False
                        st3.turn_open = False
                        st3.turn_has_completion_candidate = False
                        st3.last_turn_activity_ts = 0.0
                        st3.last_interrupt_hint_ts = 0.0

            def maybe_clear_resume_delivery_mute() -> None:
                clear_meta = False
                with self._lock:
                    st3 = self.state
                    if st3 and st3.resume_session_id and (not st3.busy) and (not st3.turn_open) and (not st3.pending_calls):
                        st3.resume_session_id = None
                        clear_meta = True
                if clear_meta:
                    self._write_meta()

            if new_off == off:
                maybe_mark_idle()
                maybe_clear_resume_delivery_mute()
                maybe_drain_one_if_idle()
                time.sleep(0.25)
                continue

            with self._lock:
                st2 = self.state
                if st2:
                    st2.log_off = new_off

            for obj in objs:
                now_ts = _now()
                token_update = _pi_token_update(obj)
                if token_update is not None:
                    with self._lock:
                        if self.state:
                            self.state.token = token_update
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
            "sock_path": str(st.sock_path),
            "agent_backend": AGENT_BACKEND,
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
        f = None
        try:
            f = conn.makefile("rb")
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
                        resp = {"busy": st.busy, "queue_len": 0, "token": st.token}
                _send_socket_json_line(conn, resp)
                return

            if cmd == "tail":
                with self._lock:
                    st = self.state
                    resp = {"tail": st.output_tail if st else ""}
                _send_socket_json_line(conn, resp)
                return

            if cmd == "send":
                text = req.get("text")
                if not isinstance(text, str) or not text.strip():
                    resp = {"error": "text required"}
                    _send_socket_json_line(conn, resp)
                    return
                seq_raw = req.get("enter_seq")
                seq = _seq_bytes(seq_raw) if isinstance(seq_raw, str) else _encode_enter()
                fd: int | None = None
                with self._lock:
                    st = self.state
                    if not st:
                        resp = {"error": "no state"}
                    else:
                        now_ts = _now()
                        st.pending_calls.clear()
                        st.busy = True
                        st.turn_open = True
                        st.turn_has_completion_candidate = False
                        st.last_interrupt_hint_ts = 0.0
                        if now_ts > st.last_turn_activity_ts:
                            st.last_turn_activity_ts = now_ts
                        fd = st.pty_master_fd
                        resp = {"queued": False, "queue_len": 0}
                _send_socket_json_line(conn, resp)
                if fd is not None:
                    try:
                        _inject(fd, text=text, suffix=seq)
                    except Exception:
                        traceback.print_exc()
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
                            _write_all(fd, b)
                        except Exception:
                            traceback.print_exc()
                _send_socket_json_line(conn, resp)
                return

            if cmd == "shutdown":
                _send_socket_json_line(conn, {"ok": True})
                self._teardown_managed_process_group()
                return

            _send_socket_json_line(conn, {"error": "unknown cmd"})
        except Exception as exc:
            if _socket_peer_disconnected(exc):
                return
            try:
                _send_socket_json_line(conn, {"error": "exception", "trace": traceback.format_exc()})
            except Exception as send_exc:
                if not _socket_peer_disconnected(send_exc):
                    traceback.print_exc()
        finally:
            if f is not None:
                try:
                    f.close()
                except Exception as close_exc:
                    if not _socket_peer_disconnected(close_exc):
                        traceback.print_exc()
            try:
                conn.close()
            except Exception as close_exc:
                if not _socket_peer_disconnected(close_exc):
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
        if AGENT_BACKEND == "codex":
            if not (lp.name.startswith("rollout-") and lp.name.endswith(".jsonl")):
                return
        elif lp.suffix != ".jsonl":
            return

        payload = _read_session_meta_payload(lp, agent_backend=AGENT_BACKEND, timeout_s=1.5)
        if not payload:
            return
        if AGENT_BACKEND == "codex" and _is_subagent_session_meta(payload):
            parent = _subagent_parent_thread_id(payload)
            if not parent:
                return
            parent_log = _find_session_log_for_session_id(self.sessions_dir, parent, agent_backend=AGENT_BACKEND)
            if not parent_log:
                return
            parent_payload = _read_session_meta_payload(parent_log, agent_backend=AGENT_BACKEND, timeout_s=0.2)
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
            st.known_rollout_paths.add(lp)
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

    def run(self) -> int:
        rows, cols = _term_size()
        _require_proc()

        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        start_ts = _now()
        headless = (OWNER_TAG == "web")
        local_terminal = (not self._emulate_terminal) and sys.stdin.isatty()

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
                if headless:
                    os.environ[BACKEND.home_env_var] = str(self.codex_home)
                    _exec_agent_via_login_shell(cwd=self.cwd, agent_args=self.codex_args)
                else:
                    os.environ[BACKEND.home_env_var] = str(self.codex_home)
                    _exec_agent(cwd=self.cwd, agent_args=self.codex_args)
            except Exception:
                traceback.print_exc()
                os._exit(127)

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
        if AGENT_BACKEND == "pi":
            st.known_rollout_paths = set(_iter_session_logs(self.sessions_dir, agent_backend="pi"))
        st.sock_path = SOCK_DIR / f"broker-{os.getpid()}.sock"
        self.state = st
        declared_log_path = _session_log_path_from_args(args=self.codex_args, agent_backend=AGENT_BACKEND, sessions_dir=self.sessions_dir)
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
