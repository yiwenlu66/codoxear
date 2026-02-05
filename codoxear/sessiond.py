#!/usr/bin/env python3
from __future__ import annotations

import argparse
import codecs
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
PENDING_DIR = APP_DIR / "pending"

CODEX_BIN = os.environ.get("CODEX_BIN", "codex")
DEFAULT_CODEX_HOME = Path(os.environ.get("CODEX_HOME") or str(Path.home() / ".codex"))
DEFAULT_ROWS = int(os.environ.get("CODEX_WEB_TTY_ROWS", "40"))
DEFAULT_COLS = int(os.environ.get("CODEX_WEB_TTY_COLS", "120"))
ENTER_SEQ = os.environ.get("CODEX_WEB_ENTER_SEQ", "\r")
OWNER_TAG = os.environ.get("CODEX_WEB_OWNER", "")


def _set_winsize(fd: int, rows: int, cols: int) -> None:
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    ws = struct.pack("HHHH", rows, cols, 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, ws)


def _encode_enter() -> bytes:
    return ENTER_SEQ.encode("utf-8")


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
        decoded = codecs.decode(raw.encode("utf-8"), "unicode_escape")
        b = decoded.encode("utf-8")
        return b
    except Exception:
        return raw.encode("utf-8", errors="replace")


def _read_jsonl_from_offset(path: Path, offset: int, max_bytes: int = 256 * 1024) -> tuple[list[dict[str, Any]], int]:
    return _read_jsonl_from_offset_impl(path, offset, max_bytes=max_bytes)


@dataclass
class State:
    session_id: str | None
    codex_pid: int
    log_path: Path
    sock_path: Path
    pty_master_fd: int
    start_ts: float
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

    def _write_meta(self) -> None:
        with self._lock:
            st = self.state
        if not st:
            return
        try:
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
            }
            SOCK_META_DIR.mkdir(parents=True, exist_ok=True)
            meta_path = st.sock_path.with_suffix(".json")
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            os.chmod(meta_path, 0o600)
        except Exception:
            pass

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
                                os.write(st.pty_master_fd, b)
                                resp = {"ok": True}
                            except Exception as e:
                                resp = {"error": str(e)}
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

    def _discover_log(self, *, preexisting: set[Path], after_ts: float) -> None:
        deadline = _now() + 120.0
        while (not self._stop.is_set()) and (_now() < deadline):
            found = _find_new_session_log(
                sessions_dir=self.sessions_dir,
                after_ts=after_ts,
                preexisting=preexisting,
                timeout_s=0.5,
            )
            if found:
                sid, lp = found
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
                    pass
                self._write_meta()
                return
            try:
                with self._lock:
                    st = self.state
                    pid = st.codex_pid if st else 0
                if pid > 0:
                    wpid, _status = os.waitpid(pid, os.WNOHANG)
                    if wpid == pid:
                        return
            except ChildProcessError:
                return
            except Exception:
                pass

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
            pass

        PENDING_DIR.mkdir(parents=True, exist_ok=True)
        pending_log = (PENDING_DIR / f"{os.getpid()}.jsonl").resolve()
        try:
            pending_log.touch(exist_ok=True)
            os.chmod(pending_log, 0o600)
        except Exception:
            pass

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
        pre = set(_iter_session_logs(self.sessions_dir))
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
            kwargs={"preexisting": pre, "after_ts": float(st.start_ts)},
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
                    pass
                try:
                    st2.sock_path.unlink()
                except Exception:
                    pass
                try:
                    st2.sock_path.with_suffix(".json").unlink()
                except Exception:
                    pass
                try:
                    if st2.log_path.exists() and str(st2.log_path).startswith(str(PENDING_DIR.resolve())):
                        st2.log_path.unlink()
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
