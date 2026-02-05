#!/usr/bin/env python3
from __future__ import annotations

import argparse
import codecs
import fcntl
import json
import os
import pty
import pwd
import re
import signal
import socket
import struct
import subprocess
import sys
import termios
import threading
import time
import traceback
import tty
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from shutil import which
from typing import Any

from codoxear.util import find_session_log_for_session_id as _find_session_log_for_session_id
from codoxear.util import is_subagent_session_meta as _is_subagent_session_meta
from codoxear.util import read_session_meta_payload as _read_session_meta_payload
from codoxear.util import subagent_parent_thread_id as _subagent_parent_thread_id

def _default_app_dir() -> Path:
    base = Path.home() / ".local" / "share"
    new = base / "codoxear"
    old = base / "codex-web"
    if old.exists() and not new.exists():
        return old
    return new


APP_DIR = _default_app_dir()
SOCK_DIR = APP_DIR / "socks"
STRACE_DIR = APP_DIR / "strace"

CODEX_BIN = os.environ.get("CODEX_BIN", "codex")
STRACE_BIN = os.environ.get("CODEX_WEB_STRACE_BIN", "strace")
# This repo relies on strace for rollout log switching (/new). Do not add non-strace fallbacks.
STRACE_ENABLED = True
OWNER_TAG = os.environ.get("CODEX_WEB_OWNER", "")
DEFAULT_CODEX_HOME = Path(os.environ.get("CODEX_HOME") or str(Path.home() / ".codex"))
DEBUG = os.environ.get("CODEX_WEB_BROKER_DEBUG", "0") == "1"

CONTEXT_WINDOW_BASELINE_TOKENS = 12000

_ROLLOUT_PATH_RE = re.compile(r'"([^"]*rollout-[^"]*\.jsonl)"')
_SESSION_ID_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", re.I)

# Keep strace noise low but include all plausible file-open entry points.
_STRACE_TRACE = "trace=open,openat,openat2,creat,rename,renameat,renameat2"


def _dprint(msg: str) -> None:
    if not DEBUG:
        return
    try:
        sys.stderr.write(msg.rstrip("\n") + "\n")
        sys.stderr.flush()
    except Exception:
        pass


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


_STRACE_USABLE: bool | None = None


def _tracer_pid() -> int:
    if not sys.platform.startswith("linux"):
        return 0
    try:
        for line in Path("/proc/self/status").read_text("utf-8", errors="replace").splitlines():
            if line.startswith("TracerPid:"):
                return int(line.split(":", 1)[1].strip() or "0")
    except Exception:
        return 0
    return 0


def _strace_usable() -> bool:
    global _STRACE_USABLE
    if _STRACE_USABLE is not None:
        return _STRACE_USABLE
    if (not STRACE_ENABLED) or (which(STRACE_BIN) is None):
        _STRACE_USABLE = False
        return False
    try:
        proc = subprocess.run(
            [STRACE_BIN, "-qq", "-o", "/dev/null", "--", "/bin/true"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=1.5,
            check=False,
        )
        _STRACE_USABLE = proc.returncode == 0
        if (not _STRACE_USABLE) and DEBUG:
            err = (proc.stderr or b"").decode("utf-8", errors="replace").strip()
            if err:
                _dprint(f"broker: strace unusable (rc={proc.returncode}): {err}")
    except Exception as e:
        if DEBUG:
            _dprint(f"broker: strace unusable: {e}")
        _STRACE_USABLE = False
    return bool(_STRACE_USABLE)


def _require_strace() -> None:
    if not _strace_usable():
        tp = _tracer_pid()
        if tp:
            sys.stderr.write(
                f"error: strace is required but not usable: this process is already ptrace-traced (TracerPid={tp}).\n"
            )
        else:
            sys.stderr.write("error: strace is required but not usable. Check ptrace restrictions and strace install.\n")
        raise SystemExit(2)

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


def _exec_strace(*, trace_path: Path, cwd: str, codex_args: list[str]) -> None:
    argv = [
        STRACE_BIN,
        "-qq",
        "-f",
        "-s",
        "4096",
        "-e",
        _STRACE_TRACE,
        "-o",
        str(trace_path),
        "--",
        CODEX_BIN,
        *codex_args,
    ]
    os.chdir(cwd)
    os.execvpe(argv[0], argv, os.environ)

def _exec_strace_via_login_shell(*, trace_path: Path, cwd: str, codex_args: list[str]) -> None:
    q = shlex.quote
    argv = [
        STRACE_BIN,
        "-qq",
        "-f",
        "-s",
        "4096",
        "-e",
        _STRACE_TRACE,
        "-o",
        str(trace_path),
        "--",
        CODEX_BIN,
        *codex_args,
    ]
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


def _is_write_open_line(line: str) -> bool:
    # Fast path: symbolic flags.
    if ("O_WRONLY" in line) or ("O_RDWR" in line) or ("O_APPEND" in line):
        return True
    if "O_RDONLY" in line:
        return False
    if "O_CREAT" in line:
        # In practice Codex creates the rollout log with write flags; treat creat-like opens as write.
        return True

    # Fallback: numeric flags (e.g. 0x241). Determine access mode from low 2 bits.
    # O_RDONLY=0, O_WRONLY=1, O_RDWR=2.
    m = re.search(r",\s*(0x[0-9a-fA-F]+|\d+)(?:\s*,|\s*\))", line)
    if not m:
        return False
    try:
        v = int(m.group(1), 16 if m.group(1).lower().startswith("0x") else 10)
    except Exception:
        return False
    acc = v & 3
    return acc in (1, 2)


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


def _inject(fd: int, *, text: str, suffix: bytes, delay_s: float = 0.02) -> None:
    os.write(fd, text.encode("utf-8"))
    if suffix:
        time.sleep(delay_s)
        os.write(fd, suffix)


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    ws = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, ws)


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


def _read_session_id_from_log(log_path: Path) -> str | None:
    try:
        with log_path.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
        obj = json.loads(first) if first else {}
        if obj.get("type") != "session_meta":
            return None
        payload = obj.get("payload") or {}
        sid = payload.get("id")
        return sid if isinstance(sid, str) and sid else None
    except Exception:
        return None


def _is_subagent_session_log(log_path: Path) -> bool:
    try:
        with log_path.open("r", encoding="utf-8") as f:
            first = f.readline().strip()
        obj = json.loads(first) if first else {}
        if obj.get("type") != "session_meta":
            return False
        payload = obj.get("payload") or {}
        src = payload.get("source")
        return isinstance(src, dict) and ("subagent" in src)
    except Exception:
        return False


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
    token: dict[str, Any] | None = None
    trace_path: Path | None = None
    trace_pid: int | None = None
    last_rollout_path: Path | None = None


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
        sid = self._session_id_from_rollout_path(log_path) or _read_session_id_from_log(log_path)
        if not sid:
            _dprint(f"broker: register_from_log: no session id: {log_path}")
            return False

        try:
            off = log_path.stat().st_size
        except Exception:
            off = 0

        sock_path = SOCK_DIR / f"{sid}-{os.getpid()}.sock"
        with self._lock:
            st = self.state
            if not st:
                return False
            st.log_path = log_path
            st.session_id = sid
            st.sock_path = sock_path
            st.log_off = off

        _dprint(f"broker: registered session_id={sid} log_path={log_path} sock_path={sock_path}")
        self._write_meta()
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
                pass
            self._term_query_buf = self._term_query_buf.replace(b"\x1b[5n", b"")
        if b"\x1b[6n" in self._term_query_buf:
            try:
                os.write(fd, b"\x1b[1;1R")
            except Exception:
                pass
            self._term_query_buf = self._term_query_buf.replace(b"\x1b[6n", b"")
        if b"\x1b[c" in self._term_query_buf:
            try:
                os.write(fd, b"\x1b[?1;2c")
            except Exception:
                pass
            self._term_query_buf = self._term_query_buf.replace(b"\x1b[c", b"")
        if b"\x1b[>c" in self._term_query_buf:
            try:
                os.write(fd, b"\x1b[>0;0;0c")
            except Exception:
                pass
            self._term_query_buf = self._term_query_buf.replace(b"\x1b[>c", b"")
        if b"\x1b[?u" in self._term_query_buf:
            try:
                os.write(fd, b"\x1b[?1u")
            except Exception:
                pass
            self._term_query_buf = self._term_query_buf.replace(b"\x1b[?u", b"")
        if b"\x1b]10;?\x1b\\" in self._term_query_buf:
            try:
                os.write(fd, b"\x1b]10;rgb:c0c0/c0c0/c0c0\x1b\\")
            except Exception:
                pass
            self._term_query_buf = self._term_query_buf.replace(b"\x1b]10;?\x1b\\", b"")
        if b"\x1b]11;?\x1b\\" in self._term_query_buf:
            try:
                os.write(fd, b"\x1b]11;rgb:0000/0000/0000\x1b\\")
            except Exception:
                pass
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
                try:
                    s = b.decode("utf-8", errors="replace")
                except Exception:
                    s = ""
                if s:
                    with self._lock:
                        st2 = self.state
                        if st2:
                            st2.output_tail = (st2.output_tail + s)[-st2.output_tail_max :]
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
            if new_off == off:
                time.sleep(0.25)
                continue
            with self._lock:
                st2 = self.state
                if st2:
                    st2.log_off = new_off

            def flush_queues() -> None:
                q: list[str] = []
                kq: list[bytes] = []
                fd: int | None = None
                with self._lock:
                    st3 = self.state
                    if not st3:
                        return
                    if not st3.queue and not st3.key_queue and not st3.busy:
                        return
                    st3.busy = False
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

            for obj in objs:
                if obj.get("type") == "event_msg":
                    p = obj.get("payload") or {}
                    pt = p.get("type")
                    if pt == "user_message":
                        msg = p.get("message")
                        if isinstance(msg, str) and msg.strip():
                            with self._lock:
                                if self.state:
                                    self.state.busy = True
                        continue
                    if pt == "turn_aborted":
                        flush_queues()
                        continue
                    if pt == "agent_message":
                        msg = p.get("message")
                        if isinstance(msg, str) and msg.strip():
                            flush_queues()
                        continue
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
                        continue

                if obj.get("type") == "response_item":
                    p = obj.get("payload") or {}
                    if p.get("type") == "message" and p.get("role") == "assistant":
                        content = p.get("content") or []
                        for part in content:
                            if (
                                isinstance(part, dict)
                                and part.get("type") == "output_text"
                                and isinstance(part.get("text"), str)
                                and part.get("text")
                            ):
                                flush_queues()
                                break
                        continue

    def _write_meta(self) -> None:
        st = self.state
        if not st or not st.sock_path:
            return
        try:
            meta = {
                "session_id": st.session_id,
                "owner": OWNER_TAG if OWNER_TAG else None,
                "broker_pid": os.getpid(),
                "sessiond_pid": os.getpid(),
                "codex_pid": st.codex_pid,
                "trace_pid": st.trace_pid,
                "cwd": st.cwd,
                "start_ts": st.start_ts,
                "log_path": str(st.log_path) if st.log_path else None,
                "sock_path": str(st.sock_path),
                "trace_path": str(st.trace_path) if st.trace_path else None,
            }
            meta_path = st.sock_path.with_suffix(".json")
            SOCK_DIR.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            os.chmod(meta_path, 0o600)
        except Exception:
            pass

    def _sock_server(self) -> None:
        st = self.state
        if not st or not st.sock_path:
            return
        try:
            SOCK_DIR.mkdir(parents=True, exist_ok=True)
            if st.sock_path.exists():
                st.sock_path.unlink()
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.bind(str(st.sock_path))
            os.chmod(st.sock_path, 0o600)
            s.listen(20)
            s.settimeout(0.5)
        except Exception:
            _dprint(f"broker: sock_server init failed: {traceback.format_exc()}")
            return

        while not self._stop.is_set():
            try:
                conn, _ = s.accept()
            except socket.timeout:
                continue
            except Exception:
                break
            threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()

        try:
            s.close()
        except Exception:
            pass

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
                            recent_local = (_now() - st.last_local_input_ts) < 0.35
                            if st.busy or recent_local:
                                st.queue.append(text)
                                resp = {"queued": True, "queue_len": len(st.queue)}
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
                            recent_local = (_now() - st.last_local_input_ts) < 0.35
                            if st.busy or recent_local:
                                st.key_queue.append(b)
                                resp = {"ok": True, "queued": True, "n": len(b), "key_queue_len": len(st.key_queue)}
                            else:
                                fd = st.pty_master_fd
                                resp = {"ok": True, "queued": False, "n": len(b), "key_queue_len": len(st.key_queue)}
                    if fd is not None:
                        try:
                            os.write(fd, b)
                        except Exception:
                            pass
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
                        try:
                            os.kill(st.codex_pid, signal.SIGTERM)
                        except Exception:
                            pass
                f.write(b'{"ok":true}\n')
                f.flush()
                return

            f.write(b'{"error":"unknown cmd"}\n')
            f.flush()
        except Exception:
            try:
                conn.sendall((json.dumps({"error": "exception", "trace": traceback.format_exc()}) + "\n").encode("utf-8"))
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _session_id_from_rollout_path(self, log_path: Path) -> str | None:
        # Codex stores rollout logs under date-based directories (e.g. ~/.codex/sessions/2026/01/22/rollout-...-<id>.jsonl),
        # so path components are not a stable session id. Extract the id from the filename.
        try:
            name = log_path.name
        except Exception:
            name = str(log_path)
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
            sid = self._session_id_from_rollout_path(lp) or _read_session_id_from_log(lp)
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

    def _strace_watcher(self, *, trace_path: Path) -> None:
        off = 0
        last_stat = 0.0
        while not self._stop.is_set():
            try:
                if not trace_path.exists():
                    time.sleep(0.1)
                    continue
                try:
                    st = trace_path.stat()
                    if st.st_size < off:
                        off = 0
                except Exception:
                    pass
                with trace_path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(off)
                    data = f.read()
                    off = f.tell()
            except Exception:
                time.sleep(0.2)
                continue

            if not data:
                now = _now()
                if now - last_stat > 1.0:
                    last_stat = now
                time.sleep(0.2)
                continue

            for line in data.splitlines():
                if "rollout-" not in line or ".jsonl" not in line:
                    continue
                m = _ROLLOUT_PATH_RE.findall(line)
                if not m:
                    continue
                raw = line.strip()
                pid = None
                parts = raw.split(None, 1)
                if len(parts) == 2 and parts[0].isdigit():
                    try:
                        pid = int(parts[0])
                    except Exception:
                        pid = None
                    raw = parts[1].lstrip()

                is_open = (
                    (" open(" in raw)
                    or (" openat(" in raw)
                    or (" openat2(" in raw)
                    or (" creat(" in raw)
                    or raw.startswith("open(")
                    or raw.startswith("openat")
                    or raw.startswith("openat2")
                    or raw.startswith("creat")
                )
                is_rename = (
                    (" rename(" in raw)
                    or (" renameat(" in raw)
                    or (" renameat2(" in raw)
                    or raw.startswith("rename")
                    or raw.startswith("renameat")
                    or raw.startswith("renameat2")
                )

                if is_open:
                    # Only treat a rollout path as active if it is opened for writing/appending.
                    if not _is_write_open_line(raw):
                        continue
                    p = m[-1]
                    self._maybe_register_or_switch_rollout(log_path=Path(p))
                    continue

                if is_rename:
                    # rename() has (src, dst); treat dst as the candidate.
                    p = m[-1]
                    self._maybe_register_or_switch_rollout(log_path=Path(p))

    def run(self) -> int:
        rows, cols = _term_size()

        _require_strace()

        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        start_ts = _now()
        headless = (OWNER_TAG == "web")

        pid, master_fd = pty.fork()
        if pid == 0:
            try:
                _set_pdeathsig(signal.SIGHUP)
                try:
                    os.setpgid(0, 0)
                except Exception:
                    pass
                os.environ.setdefault("TERM", os.environ.get("TERM") or "xterm-256color")
                os.environ["COLUMNS"] = str(cols)
                os.environ["LINES"] = str(rows)
                os.environ["CODEX_HOME"] = str(self.codex_home)
                try:
                    fd = sys.stdin.fileno()
                    attrs = termios.tcgetattr(fd)
                    attrs[0] &= ~(termios.ICRNL | termios.INLCR | termios.IGNCR)
                    termios.tcsetattr(fd, termios.TCSANOW, attrs)
                except Exception:
                    pass
                trace_dir = STRACE_DIR / str(os.getppid())
                trace_dir.mkdir(parents=True, exist_ok=True)
                trace_path = trace_dir / "syscalls.log"
                if headless:
                    _exec_strace_via_login_shell(trace_path=trace_path, cwd=self.cwd, codex_args=self.codex_args)
                else:
                    _exec_strace(trace_path=trace_path, cwd=self.cwd, codex_args=self.codex_args)
            except Exception:
                traceback.print_exc()
                os._exit(127)

        if (not headless) and (not self._emulate_terminal) and sys.stdin.isatty():
            try:
                fd = sys.stdin.fileno()
                self._stdin_termios = termios.tcgetattr(fd)
                tty.setraw(fd)
            except Exception:
                self._stdin_termios = None

        try:
            _set_winsize(master_fd, rows, cols)
        except Exception:
            pass

        st = State(
            codex_pid=pid,
            pty_master_fd=master_fd,
            cwd=self.cwd,
            start_ts=start_ts,
            codex_home=self.codex_home,
            sessions_dir=self.sessions_dir,
            busy=False,
            trace_path=(STRACE_DIR / str(os.getpid()) / "syscalls.log"),
            trace_pid=pid,
        )
        st.sock_path = SOCK_DIR / f"broker-{os.getpid()}.sock"
        self.state = st

        def _sigwinch(_signo: int, _frame: Any) -> None:
            try:
                r, c = _term_size()
                _set_winsize(master_fd, r, c)
            except Exception:
                pass

        signal.signal(signal.SIGWINCH, _sigwinch)

        self._write_meta()
        threading.Thread(target=self._sock_server, daemon=True).start()
        threading.Thread(target=self._pty_to_stdout, daemon=True).start()
        if not headless:
            threading.Thread(target=self._stdin_to_pty, daemon=True).start()
        threading.Thread(target=self._log_watcher, daemon=True).start()
        # Start strace-based rollout detection (captures /new, /resume, and interactive selection).
        if st.trace_path is not None:
            threading.Thread(target=self._strace_watcher, kwargs={"trace_path": st.trace_path}, daemon=True).start()

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
                    pass
                self._stdin_termios = None

        self._stop.set()
        try:
            os.close(master_fd)
        except Exception:
            pass
        with self._lock:
            st2 = self.state
        if st2 and st2.sock_path:
            try:
                st2.sock_path.unlink()
            except Exception:
                pass
            try:
                st2.sock_path.with_suffix(".json").unlink()
            except Exception:
                pass
        return exit_code


def main() -> None:
    if not sys.platform.startswith("linux"):
        sys.stderr.write("error: codoxear-broker requires Linux (/proc, pty, termios)\n")
        raise SystemExit(2)
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
