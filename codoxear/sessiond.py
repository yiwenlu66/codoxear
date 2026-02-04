#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import json
import os
import pty
import signal
import socket
import struct
import subprocess
import threading
import time
import traceback
import select
import sys
import termios
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .util import default_app_dir as _default_app_dir
from .util import find_new_session_log as _find_new_session_log
from .util import iter_session_logs as _iter_session_logs
from .util import now as _now
from .util import read_jsonl_from_offset as _read_jsonl_from_offset_impl


APP_DIR = _default_app_dir()
SOCK_DIR = APP_DIR / "socks"
SOCK_META_DIR = SOCK_DIR
ROOT_REPO_DIR = APP_DIR / "root-repo"

CODEX_BIN = os.environ.get("CODEX_BIN", "codex")
DEFAULT_CODEX_HOME = Path(os.environ.get("CODEX_HOME") or str(Path.home() / ".codex"))
DEFAULT_ROWS = int(os.environ.get("CODEX_WEB_TTY_ROWS", "40"))
DEFAULT_COLS = int(os.environ.get("CODEX_WEB_TTY_COLS", "120"))
ENTER_SEQ = os.environ.get("CODEX_WEB_ENTER_SEQ", "\r")


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    ws = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, ws)


def _encode_enter() -> bytes:
    return ENTER_SEQ.encode("utf-8")

def _read_jsonl_from_offset(path: Path, offset: int, max_bytes: int = 256 * 1024) -> tuple[list[dict[str, Any]], int]:
    return _read_jsonl_from_offset_impl(path, offset, max_bytes=max_bytes)


@dataclass
class State:
    session_id: str
    codex_pid: int
    log_path: Path
    sock_path: Path
    pty_master_fd: int
    busy: bool = False
    queue: list[str] = field(default_factory=list)
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
                        os.write(fd, b"\x1b[1;1R")
                    except Exception:
                        pass
                # xterm "report terminal size" queries (some TUIs use these).
                if b"\x1b[18t" in b:
                    try:
                        os.write(fd, f"\x1b[8;{self.rows};{self.cols}t".encode("ascii"))
                    except Exception:
                        pass
                if b"\x1b[14t" in b:
                    try:
                        os.write(fd, b"\x1b[4;0;0t")
                    except Exception:
                        pass
                s = b.decode("utf-8", errors="replace")
                with self._lock:
                    st2 = self.state
                    if not st2:
                        continue
                    st2.output_tail = (st2.output_tail + s)[-st2.output_tail_max :]
            except OSError:
                break

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
                    p = obj.get("payload") or {}
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
                    if not self.state:
                        continue
                    self.state.busy = False
                    q = self.state.queue[:]
                    self.state.queue.clear()
                    fd = self.state.pty_master_fd

                if q:
                    wrote_any = False
                    for msg in q:
                        try:
                            os.write(fd, msg.encode("utf-8") + _encode_enter())
                            wrote_any = True
                        except Exception:
                            break
                    if wrote_any:
                        with self._lock:
                            if self.state:
                                self.state.busy = True

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
                        resp = {"busy": st.busy, "queue_len": len(st.queue)}
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
                    with self._lock:
                        st = self.state
                        if not st:
                            resp = {"error": "no state"}
                        elif st.busy:
                            st.queue.append(text)
                            resp = {"queued": True, "queue_len": len(st.queue)}
                        else:
                            os.write(st.pty_master_fd, text.encode("utf-8") + _encode_enter())
                            st.busy = True
                            resp = {"queued": False, "queue_len": len(st.queue)}
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

    def launch(self) -> State:
        SOCK_DIR.mkdir(parents=True, exist_ok=True)
        pre = set(_iter_session_logs(self.sessions_dir))
        start_ts = _now()

        # Codex interactive mode expects to run in a Git repo. Use a small local repo as a stable root
        # and add the requested cwd as an additional writable directory.
        if not (ROOT_REPO_DIR / ".git").exists():
            ROOT_REPO_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-q"], cwd=str(ROOT_REPO_DIR), check=True)
            # Create an initial commit to satisfy tooling that assumes HEAD exists.
            (ROOT_REPO_DIR / ".codoxear-root").write_text("codoxear\n", encoding="utf-8")
            subprocess.run(["git", "add", ".codoxear-root"], cwd=str(ROOT_REPO_DIR), check=True)
            subprocess.run(
                ["git", "-c", "user.email=codoxear@local", "-c", "user.name=codoxear", "commit", "-qm", "init"],
                cwd=str(ROOT_REPO_DIR),
                check=True,
            )

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
            pass

        # Capture early output in case the child exits before producing a session log.
        early = b""
        early_deadline = _now() + 3.0
        exited = False
        while _now() < early_deadline:
            try:
                wpid, _status = os.waitpid(pid, os.WNOHANG)
                if wpid == pid:
                    exited = True
                    break
            except ChildProcessError:
                exited = True
                break
            r, _, _ = select.select([master_fd], [], [], 0.1)
            if master_fd in r:
                try:
                    chunk = os.read(master_fd, 4096)
                    if b"\x1b[6n" in chunk:
                        try:
                            os.write(master_fd, b"\x1b[1;1R")
                        except Exception:
                            pass
                    early += chunk
                except Exception:
                    break
        if exited:
            msg = early.decode("utf-8", errors="replace")
            sys.stderr.write(f"codex exited before session log was created. output={msg[-2000:]}\n")
            sys.stderr.flush()
            os._exit(1)

        found: tuple[str, Path] | None = None
        scan_deadline = _now() + 120.0
        while _now() < scan_deadline:
            found = _find_new_session_log(sessions_dir=self.sessions_dir, after_ts=start_ts, preexisting=pre, timeout_s=0.5)
            if found:
                break
            # Detect child exit while waiting.
            try:
                wpid, _status = os.waitpid(pid, os.WNOHANG)
                if wpid == pid:
                    tail = early.decode("utf-8", errors="replace")
                    sys.stderr.write(f"codex exited before session log was created. output={tail[-2000:]}\n")
                    sys.stderr.flush()
                    os._exit(1)
            except ChildProcessError:
                tail = early.decode("utf-8", errors="replace")
                sys.stderr.write(f"codex exited before session log was created. output={tail[-2000:]}\n")
                sys.stderr.flush()
                os._exit(1)
            # Keep sampling early output while waiting.
            r, _, _ = select.select([master_fd], [], [], 0.2)
            if master_fd in r:
                try:
                    chunk = os.read(master_fd, 4096)
                    if b"\x1b[6n" in chunk:
                        try:
                            os.write(master_fd, b"\x1b[1;1R")
                        except Exception:
                            pass
                    early += chunk
                except Exception:
                    pass

        if not found:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
            raise RuntimeError("failed to discover codex session log within timeout")
        session_id, log_path = found

        sock_path = SOCK_DIR / f"{session_id}.sock"
        st = State(
            session_id=session_id,
            codex_pid=pid,
            log_path=log_path,
            sock_path=sock_path,
            pty_master_fd=master_fd,
            busy=False,
        )
        self.state = st
        return st

    def run(self) -> None:
        st = self.launch()
        # Persist enough metadata for the web server to rediscover sessions after restart.
        try:
            meta = {
                "session_id": st.session_id,
                "codex_pid": st.codex_pid,
                "cwd": self.cwd,
                "start_ts": _now(),
                "log_path": str(st.log_path),
                "sock_path": str(st.sock_path),
            }
            SOCK_META_DIR.mkdir(parents=True, exist_ok=True)
            meta_path = SOCK_META_DIR / f"{st.session_id}.json"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            os.chmod(meta_path, 0o600)
        except Exception:
            pass
        # First line to stdout is the handshake for the parent.
        print(
            json.dumps(
                {
                    "session_id": st.session_id,
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
                    pass
                try:
                    st2.sock_path.unlink()
                except Exception:
                    pass


def main() -> None:
    if not sys.platform.startswith("linux"):
        sys.stderr.write("error: codoxear session helper requires Linux\n")
        raise SystemExit(2)
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwd", default=os.getcwd())
    ap.add_argument("args", nargs=argparse.REMAINDER)
    ns = ap.parse_args()

    args = list(ns.args)
    if args and args[0] == "--":
        args = args[1:]

    sd = Sessiond(cwd=str(ns.cwd), codex_args=args)
    sd.run()


if __name__ == "__main__":
    main()
