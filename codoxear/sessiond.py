#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pty
import signal
import socket
import subprocess
import threading
import time
import traceback
import select
import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import pty_util as _pty_util
from .util import default_app_dir as _default_app_dir
from .util import now as _now
from .util import proc_find_open_rollout_log as _proc_find_open_rollout_log
from .util import read_jsonl_from_offset as _read_jsonl_from_offset_impl
from .util import read_session_meta_payload as _read_session_meta_payload
from .util import _send_socket_json_line as _send_socket_json_line
from .util import _socket_peer_disconnected as _socket_peer_disconnected


APP_DIR = _default_app_dir()
SOCK_DIR = APP_DIR / "socks"
SOCK_META_DIR = SOCK_DIR
ROOT_REPO_DIR = APP_DIR / "root-repo"
PENDING_DIR = APP_DIR / "pending"
PROC_ROOT = Path("/proc")

CODEX_BIN = os.environ.get("CODEX_BIN", "codex")
_CODEX_HOME_ENV = os.environ.get("CODEX_HOME")
if _CODEX_HOME_ENV is None or (not _CODEX_HOME_ENV.strip()):
    DEFAULT_CODEX_HOME = Path.home() / ".codex"
else:
    DEFAULT_CODEX_HOME = Path(_CODEX_HOME_ENV)
DEFAULT_ROWS = int(os.environ.get("CODEX_WEB_TTY_ROWS", "40"))
DEFAULT_COLS = int(os.environ.get("CODEX_WEB_TTY_COLS", "120"))
ENTER_SEQ = os.environ.get("CODEX_WEB_ENTER_SEQ", "\r")
OWNER_TAG = os.environ.get("CODEX_WEB_OWNER", "")
MODEL_OVERRIDE = os.environ.get("CODEX_WEB_MODEL", "").strip()
REASONING_EFFORT_OVERRIDE = os.environ.get("CODEX_WEB_REASONING_EFFORT", "").strip().lower()
_BRACKETED_PASTE_START = b"\x1b[200~"
_BRACKETED_PASTE_END = b"\x1b[201~"


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    _pty_util.set_winsize(fd, rows, cols)


def _encode_enter() -> bytes:
    return ENTER_SEQ.encode("utf-8")


def _seq_bytes(raw: str) -> bytes:
    return _pty_util.seq_bytes(raw)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        n = os.write(fd, view)
        if n <= 0:
            raise OSError("short write to PTY")
        view = view[n:]


def _inject(fd: int, *, text: str, suffix: bytes, delay_s: float = 0.05) -> None:
    _write_all(fd, _BRACKETED_PASTE_START + text.encode("utf-8") + _BRACKETED_PASTE_END)
    if not suffix:
        return
    if delay_s > 0:
        time.sleep(delay_s)
    _write_all(fd, suffix)


def _read_jsonl_from_offset(path: Path, offset: int, max_bytes: int = 256 * 1024) -> tuple[list[dict[str, Any]], int]:
    return _read_jsonl_from_offset_impl(path, offset, max_bytes=max_bytes)


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


@dataclass
class State:
    session_id: str | None
    codex_pid: int
    log_path: Path
    sock_path: Path
    pty_master_fd: int
    start_ts: float
    busy: bool = False
    output_tail: str = ""
    output_tail_max: int = 64 * 1024
    log_off: int = 0


class Sessiond:
    def __init__(self, cwd: str, codex_args: list[str]) -> None:
        self.cwd = cwd
        self.codex_args = codex_args
        self.rows = DEFAULT_ROWS
        self.cols = DEFAULT_COLS
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.state: State | None = None
        self.codex_home = DEFAULT_CODEX_HOME
        self.sessions_dir = self.codex_home / "sessions"

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

    def _pty_reader(self) -> None:
        st = self.state
        if not st:
            return
        fd = st.pty_master_fd
        while not self._stop.is_set():
            try:
                b = os.read(fd, 4096)
                if not b:
                    break
                # Minimal terminal-emulator responses for Codex TUI startup.
                if b"\x1b[6n" in b:
                    try:
                        _write_all(fd, b"\x1b[1;1R")
                    except Exception:
                        traceback.print_exc()
                # xterm "report terminal size" queries (some TUIs use these).
                if b"\x1b[18t" in b:
                    try:
                        _write_all(fd, f"\x1b[8;{self.rows};{self.cols}t".encode("ascii"))
                    except Exception:
                        traceback.print_exc()
                if b"\x1b[14t" in b:
                    try:
                        _write_all(fd, b"\x1b[4;0;0t")
                    except Exception:
                        traceback.print_exc()
                s = b.decode("utf-8", errors="replace")
                with self._lock:
                    st2 = self.state
                    if not st2:
                        continue
                    st2.output_tail = (st2.output_tail + s)[-st2.output_tail_max :]
            except OSError:
                break

    def _write_meta(self) -> None:
        with self._lock:
            st = self.state
        if not st:
            return
        meta = {
            "session_id": st.session_id,
            "owner": OWNER_TAG if OWNER_TAG else None,
            "broker_pid": os.getpid(),
            "sessiond_pid": os.getpid(),
            "codex_pid": st.codex_pid,
            "cwd": self.cwd,
            "start_ts": float(st.start_ts),
            "log_path": str(st.log_path),
            "sock_path": str(st.sock_path),
            "model": MODEL_OVERRIDE or None,
            "reasoning_effort": REASONING_EFFORT_OVERRIDE or None,
        }
        SOCK_META_DIR.mkdir(parents=True, exist_ok=True)
        meta_path = st.sock_path.with_suffix(".json")
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        os.chmod(meta_path, 0o600)

    def _log_watcher(self) -> None:
        st = self.state
        if not st:
            return
        while not self._stop.is_set():
            objs, off = _read_jsonl_from_offset(st.log_path, st.log_off, max_bytes=256 * 1024)
            if off == st.log_off:
                time.sleep(0.25)
                continue
            st.log_off = off

            saw_user = False
            saw_turn_end = False
            for obj in objs:
                if obj.get("type") == "event_msg":
                    p = obj.get("payload")
                    if not isinstance(p, dict):
                        raise ValueError("invalid rollout event_msg payload")
                    if p.get("type") == "user_message":
                        saw_user = True
                    if p.get("type") == "token_count":
                        info = p.get("info")
                        if isinstance(info, dict) and isinstance(info.get("total_token_usage"), dict):
                            saw_turn_end = True
                elif obj.get("type") == "response_item":
                    continue

            if saw_user:
                with self._lock:
                    if self.state:
                        self.state.busy = True

            if saw_turn_end:
                with self._lock:
                    if self.state:
                        self.state.busy = False

    def _sock_server(self) -> None:
        st = self.state
        if not st:
            return
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
                sys.stderr.write(f"error: sessiond socket server crashed: {traceback.format_exc()}\n")
                try:
                    self._teardown_managed_process_group()
                except Exception:
                    traceback.print_exc()
                break
            threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()

        try:
            s.close()
        except Exception:
            traceback.print_exc()

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
                        resp = {"busy": st.busy, "queue_len": 0}
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
                fd: int | None = None
                enter = _encode_enter()
                with self._lock:
                    st = self.state
                    if not st:
                        resp = {"error": "no state"}
                    else:
                        fd = st.pty_master_fd
                        resp = {"queued": False, "queue_len": 0}
                _send_socket_json_line(conn, resp)
                if fd is not None:
                    try:
                        _inject(fd, text=text, suffix=enter)
                    except Exception:
                        traceback.print_exc()
                return

            if cmd == "keys":
                seq = req.get("seq")
                if not isinstance(seq, str) or not seq:
                    resp = {"error": "seq required"}
                else:
                    b = _seq_bytes(seq)
                    with self._lock:
                        st = self.state
                        if not st:
                            resp = {"error": "no state"}
                        else:
                            try:
                                _write_all(st.pty_master_fd, b)
                                resp = {"ok": True}
                            except Exception as e:
                                resp = {"error": str(e)}
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

    def _ensure_root_repo(self) -> None:
        if (ROOT_REPO_DIR / ".git").exists():
            return
        ROOT_REPO_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "-q"], cwd=str(ROOT_REPO_DIR), check=True)
        (ROOT_REPO_DIR / ".codoxear-root").write_text("codoxear\n", encoding="utf-8")
        subprocess.run(["git", "add", ".codoxear-root"], cwd=str(ROOT_REPO_DIR), check=True)
        subprocess.run(
            ["git", "-c", "user.email=codoxear@local", "-c", "user.name=codoxear", "commit", "-qm", "init"],
            cwd=str(ROOT_REPO_DIR),
            check=True,
        )

    def _discover_log(self) -> None:
        deadline = _now() + 120.0
        while (not self._stop.is_set()) and (_now() < deadline):
            with self._lock:
                st0 = self.state
            pid = int(st0.codex_pid) if st0 else 0
            if pid > 0:
                lp = _proc_find_open_rollout_log(proc_root=PROC_ROOT, root_pid=pid, cwd=self.cwd)
            else:
                lp = None
            if lp and lp.exists():
                payload = _read_session_meta_payload(lp, timeout_s=0.0)
                sid = payload.get("id") if isinstance(payload, dict) else None
                if not (isinstance(sid, str) and sid):
                    time.sleep(0.25)
                    continue
                with self._lock:
                    st = self.state
                    if not st:
                        return
                    prev_lp = st.log_path
                    st.session_id = sid
                    st.log_path = lp
                    st.log_off = 0
                try:
                    if prev_lp.exists() and str(prev_lp).startswith(str(PENDING_DIR.resolve())):
                        prev_lp.unlink()
                except Exception:
                    traceback.print_exc()
                self._write_meta()
                return
            try:
                if pid > 0:
                    wpid, _status = os.waitpid(pid, os.WNOHANG)
                    if wpid == pid:
                        return
            except ChildProcessError:
                return
            except Exception:
                traceback.print_exc()
            time.sleep(0.25)

    def _start(self) -> State:
        SOCK_DIR.mkdir(parents=True, exist_ok=True)
        self._ensure_root_repo()
        start_ts = _now()

        pid, master_fd = pty.fork()
        if pid == 0:
            try:
                os.environ.setdefault("TERM", "xterm-256color")
                os.environ["COLUMNS"] = str(self.cols)
                os.environ["LINES"] = str(self.rows)
                os.chdir(str(ROOT_REPO_DIR))
                forced = [
                    "--no-alt-screen",
                    "-c",
                    "disable_response_storage=false",
                    "-c",
                    "disable_paste_burst=true",
                    "-C",
                    str(ROOT_REPO_DIR),
                ]
                if self.cwd:
                    forced += ["--add-dir", self.cwd]
                argv = [CODEX_BIN] + forced + self.codex_args
                os.execvp(argv[0], argv)
            except Exception:
                traceback.print_exc()
                os._exit(127)

        # Default PTY size from pty.fork() is 0x0; many TUIs assume non-zero dimensions.
        try:
            _set_winsize(master_fd, self.rows, self.cols)
        except Exception:
            traceback.print_exc()

        PENDING_DIR.mkdir(parents=True, exist_ok=True)
        pending_log = (PENDING_DIR / f"{os.getpid()}.jsonl").resolve()
        try:
            pending_log.touch(exist_ok=True)
            os.chmod(pending_log, 0o600)
        except Exception:
            traceback.print_exc()

        sock_path = SOCK_DIR / f"broker-{os.getpid()}.sock"
        st = State(
            session_id=None,
            codex_pid=pid,
            log_path=pending_log,
            sock_path=sock_path,
            pty_master_fd=master_fd,
            start_ts=float(start_ts),
            busy=False,
        )
        self.state = st
        return st

    def run(self) -> None:
        st = self._start()
        self._write_meta()
        print(
            json.dumps(
                {
                    "session_id": st.session_id,
                    "broker_pid": os.getpid(),
                    "codex_pid": st.codex_pid,
                    "log_path": str(st.log_path),
                    "sock_path": str(st.sock_path),
                }
            ),
            flush=True,
        )

        threading.Thread(target=self._pty_reader, daemon=True).start()
        threading.Thread(target=self._log_watcher, daemon=True).start()
        threading.Thread(target=self._sock_server, daemon=True).start()
        threading.Thread(
            target=self._discover_log,
            daemon=True,
        ).start()

        try:
            while not self._stop.is_set():
                time.sleep(0.5)
        finally:
            with self._lock:
                st2 = self.state
            if st2:
                try:
                    os.close(st2.pty_master_fd)
                except Exception:
                    traceback.print_exc()
                try:
                    st2.sock_path.unlink()
                except Exception:
                    traceback.print_exc()
                try:
                    st2.sock_path.with_suffix(".json").unlink()
                except Exception:
                    traceback.print_exc()
                try:
                    if st2.log_path.exists() and str(st2.log_path).startswith(str(PENDING_DIR.resolve())):
                        st2.log_path.unlink()
                except Exception:
                    traceback.print_exc()


def main() -> None:
    if not (sys.platform.startswith("linux") or sys.platform == "darwin"):
        sys.stderr.write(f"error: codoxear session helper requires Linux or macOS (unsupported: {sys.platform})\n")
        raise SystemExit(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwd", default=os.getcwd())
    ap.add_argument("args", nargs=argparse.REMAINDER)
    ns = ap.parse_args()

    args = list(ns.args)
    if args and args[0] == "--":
        args = args[1:]

    home = str(Path.home())
    cwd = str(ns.cwd).strip().replace("${HOME}", home)
    cwd = re.sub(r"\$HOME(?![A-Za-z0-9_])", home, cwd)
    cwd = os.path.expanduser(os.path.expandvars(cwd))
    sd = Sessiond(cwd=cwd, codex_args=args)
    sd.run()


if __name__ == "__main__":
    main()
