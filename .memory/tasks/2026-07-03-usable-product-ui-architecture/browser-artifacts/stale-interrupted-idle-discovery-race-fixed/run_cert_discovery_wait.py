#!/usr/bin/env python3
"""Docker certification harness: log-only stale interrupted_idle projection.

Drives the REAL codoxear server session-discovery / listing code paths against
a FAKE control socket (always reports busy=false, queue_len=0,
interrupted_idle=true) and a REAL rollout log written in Codex JSONL.

Phases:
  1. Interrupted non-final turn in the log. Expect /api/sessions busy=false
     (immediate-interrupt override applies).
  2. Append post-interrupt resumed activity (user_message starting a new turn)
     to the SAME log. The real log idle parser sees non-idle. Discriminator:
     does /api/sessions flip to busy=true despite the stale broker
     interrupted_idle input?
  3. Append a completion (task_complete) row so the log becomes idle again.
     Expect busy=false.

No product code is edited. This script only creates socket sidecars / log
files inside the container app dir and drives HTTP.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

ART = Path("/artifacts")
APP_DIR = Path(os.environ.get("HOME", "/home/tester")) / ".local" / "share" / "codoxear"
SOCK_DIR = APP_DIR / "socks"
LOG = APP_DIR / "rollout-broker-1.jsonl"
SIDECAR = SOCK_DIR / "broker-1.json"
SOCK = SOCK_DIR / "broker-1.sock"

PASSWORD = os.environ.get("CODEX_WEB_PASSWORD", "certpass")
PORT = int(os.environ.get("CODEX_WEB_PORT", "13790"))
HOST = "127.0.0.1"
BASE = f"http://{HOST}:{PORT}"
SESSION_ID = "broker-1"

snapshots: list[dict] = []
server_proc: subprocess.Popen | None = None
socket_stop = threading.Event()


# ---------------------------------------------------------------------------
# Fake control socket: always returns the STALE interrupted-idle broker state.
# ---------------------------------------------------------------------------
STALE_STATE = {"busy": False, "queue_len": 0, "interrupted_idle": True, "token": None}
state_call_count = {"n": 0}


def _handle_conn(conn: socket.socket) -> None:
    try:
        f = conn.makefile("rb")
        line = f.readline()
        if not line:
            return
        req = json.loads(line.decode("utf-8"))
        cmd = req.get("cmd") if isinstance(req, dict) else None
        if cmd == "state":
            state_call_count["n"] += 1
            resp = dict(STALE_STATE)
        elif cmd == "tail":
            resp = {"tail": ""}
        else:
            resp = {"ok": True}
        conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"fake-sock: {type(exc).__name__}: {exc}\n")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _socket_server() -> None:
    try:
        SOCK.unlink()
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(SOCK))
    srv.listen(16)
    srv.settimeout(0.3)
    while not socket_stop.is_set():
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        threading.Thread(target=_handle_conn, args=(conn,), daemon=True).start()
    try:
        srv.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Log builders (Codex JSONL, same shapes as tests/test_stale_interrupted_idle.py)
# ---------------------------------------------------------------------------
def write_phase1_interrupted_turn() -> int:
    rows = [
        {"type": "session_meta", "payload": {"id": SESSION_ID, "source": "cli"}},
        {"type": "event_msg", "payload": {"type": "user_message", "message": "first"}, "ts": 10.0},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "working"}],
            },
            "ts": 11.0,
        },
    ]
    with LOG.open("w", encoding="utf-8") as f:
        for o in rows:
            f.write(json.dumps(o) + "\n")
    return LOG.stat().st_size


def append_rows(rows: list[dict]) -> int:
    with LOG.open("a", encoding="utf-8") as f:
        for o in rows:
            f.write(json.dumps(o) + "\n")
    return LOG.stat().st_size


# ---------------------------------------------------------------------------
# HTTP client with cookie jar
# ---------------------------------------------------------------------------
def make_opener() -> urllib.request.OpenerDirector:
    cj = CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj)), cj


def http_get(opener, path: str) -> tuple[int, object]:
    try:
        with opener.open(BASE + path, timeout=5) as r:
            body = r.read().decode("utf-8", "replace")
            return r.status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def login(opener) -> bool:
    data = json.dumps({"password": PASSWORD}).encode("utf-8")
    req = urllib.request.Request(BASE + "/api/login", data=data, headers={"Content-Type": "application/json"})
    try:
        with opener.open(req, timeout=5) as r:
            return r.status == 200
    except urllib.error.HTTPError:
        return False


def fetch_sessions(opener) -> list[dict] | None:
    status, body = http_get(opener, "/api/sessions")
    if status != 200:
        return None
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and isinstance(obj.get("sessions"), list):
        return obj["sessions"]
    if isinstance(obj, list):
        return obj
    return None


def find_row(rows: list[dict]) -> dict | None:
    for r in rows:
        if str(r.get("session_id")) == SESSION_ID:
            return r
    return None


def poll_busy(opener, rounds: int, interval: float) -> list[bool | None]:
    out: list[bool | None] = []
    for _ in range(rounds):
        rows = fetch_sessions(opener)
        row = find_row(rows) if rows else None
        out.append(bool(row["busy"]) if row else None)
        time.sleep(interval)
    return out


def wait_for_port(timeout: float = 15.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((HOST, PORT), timeout=1):
                return True
        except OSError:
            time.sleep(0.3)
    return False


# ---------------------------------------------------------------------------
# In-process diagnostic: reproduce the unit-test condition vs the real
# listing order (prune re-baselines before update_meta_counters), using the
# real coordinators against the same log. Proves WHY the end-to-end result
# holds, without editing product code.
# ---------------------------------------------------------------------------
def in_process_diagnostic() -> dict:
    import threading as _t

    from codoxear.rollout_idle import _analyze_log_chunk, _compute_idle_from_log
    from codoxear.rollout_jsonl import _read_jsonl_records_from_offset
    from codoxear.session_log_runtime import SessionLogRuntimeCoordinator
    from codoxear.session_model import Session
    from codoxear.session_runtime import set_session_interrupted_idle

    def read_objs(path, offset, max_bytes=256 * 1024):
        recs, new_off = _read_jsonl_records_from_offset(path, offset, max_bytes=max_bytes)
        return [r.obj for r in recs], new_off

    def make_runtime(session):
        return SessionLogRuntimeCoordinator(
            lock=_t.Lock(),
            sessions=lambda: {SESSION_ID: session},
            analyze_log_chunk=_analyze_log_chunk,
            turn_context_run_settings=lambda _p: (None, None),
            compute_idle_from_log=_compute_idle_log_wrapper,
            read_jsonl_from_offset=read_objs,
            find_latest_token_update=lambda _p: None,
        )

    class _Wrap:
        # compute_idle_from_log signature is (path) -> bool|None
        def __call__(self, path):
            return _compute_idle_from_log(path)

    _compute_idle_log_wrapper = _Wrap()

    def fresh_session() -> Session:
        s = Session(
            session_id=SESSION_ID, thread_id=SESSION_ID, broker_pid=1, codex_pid=2,
            agent_backend="codex", owned=False, start_ts=100.0, cwd="/workspace",
            log_path=LOG, sock_path=SOCK, busy=False, queue_len=0,
        )
        s.meta_log_off = int(LOG.stat().st_size)
        set_session_interrupted_idle(s, True)
        return s

    result: dict = {}

    # Condition A (unit-test condition): interrupted_idle set at the OLD log
    # size, NO prune re-baselining. update_meta_counters alone sees the new
    # chunk and clears the override.
    size_before = LOG.stat().st_size
    snap_pre = {"log_size": size_before, "idle": _compute_idle_from_log(LOG)}

    # Use a separate copy of the log to avoid disturbing the server's log.
    import shutil, tempfile
    td = Path(tempfile.mkdtemp())
    logA = td / "a.jsonl"
    shutil.copy(LOG, logA)

    sA = Session(
        session_id="A", thread_id="A", broker_pid=1, codex_pid=2, agent_backend="codex",
        owned=False, start_ts=1.0, cwd="/tmp", log_path=logA, sock_path=SOCK,
        busy=False, queue_len=0,
    )
    sA.meta_log_off = int(logA.stat().st_size)
    set_session_interrupted_idle(sA, True)  # baseline at current size
    rtA = SessionLogRuntimeCoordinator(
        lock=_t.Lock(), sessions=lambda: {"A": sA}, analyze_log_chunk=_analyze_log_chunk,
        turn_context_run_settings=lambda _p: (None, None),
        compute_idle_from_log=lambda p: _compute_idle_from_log(p),
        read_jsonl_from_offset=read_objs, find_latest_token_update=lambda _p: None,
    )
    condA_pre = {"interrupted_idle": sA.interrupted_idle, "log_off_baseline": sA.interrupted_idle_log_off, "size": logA.stat().st_size}
    # append post-interrupt activity to logA
    with logA.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": "second"}, "ts": 20.0}) + "\n")
    rtA.update_meta_counters()
    condA = {
        "scenario": "update_meta_counters ALONE (baseline captured before append, no prune re-baseline)",
        "interrupted_idle_after": sA.interrupted_idle,
        "log_idle_after": _compute_idle_from_log(logA),
    }

    # Condition B (real listing order): prune re-baselines interrupted_idle to
    # the CURRENT log size (which now includes the appended activity) BEFORE
    # update_meta_counters runs. update_meta_counters then advances its read
    # cursor to that baseline and sees no post-baseline content.
    logB = td / "b.jsonl"
    shutil.copy(LOG, logB)
    # append post-interrupt activity FIRST (mimics activity present at poll time)
    with logB.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": "second"}, "ts": 20.0}) + "\n")
    sB = Session(
        session_id="B", thread_id="B", broker_pid=1, codex_pid=2, agent_backend="codex",
        owned=False, start_ts=1.0, cwd="/tmp", log_path=logB, sock_path=SOCK,
        busy=False, queue_len=0,
    )
    sB.meta_log_off = int(logB.stat().st_size) - 0  # pretend prior poll advanced meta_log_off to pre-append size
    # measure pre-append size by subtracting the appended line length approx is fragile;
    # instead set meta_log_off to the size BEFORE append by re-deriving:
    pre_size = logB.stat().st_size - len(json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": "second"}, "ts": 20.0}) + "\n")
    sB.meta_log_off = pre_size
    set_session_interrupted_idle(sB, False)
    # mimic prune reading stale socket after append: re-baseline to CURRENT size
    set_session_interrupted_idle(sB, True)
    rtB = SessionLogRuntimeCoordinator(
        lock=_t.Lock(), sessions=lambda: {"B": sB}, analyze_log_chunk=_analyze_log_chunk,
        turn_context_run_settings=lambda _p: (None, None),
        compute_idle_from_log=lambda p: _compute_idle_from_log(p),
        read_jsonl_from_offset=read_objs, find_latest_token_update=lambda _p: None,
    )
    rtB.update_meta_counters()
    condB = {
        "scenario": "real listing order: prune re-baselines to CURRENT log size, THEN update_meta_counters",
        "interrupted_idle_after": sB.interrupted_idle,
        "log_idle_after": _compute_idle_from_log(logB),
        "meta_log_off_after": sB.meta_log_off,
        "interrupted_idle_log_off": sB.interrupted_idle_log_off,
    }

    result["compute_idle_pre_append"] = snap_pre
    result["condition_A_update_only"] = condA
    result["condition_B_prune_then_update"] = condB
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    SOCK_DIR.mkdir(parents=True, exist_ok=True)

    sidecar = {
        "agent_backend": "codex",
        "session_id": SESSION_ID,
        "codex_pid": 999991,
        "broker_pid": 999992,
        "cwd": "/workspace",
        "log_path": str(LOG),
        "start_ts": 100.0,
        "owner": "terminal",
        "source": "cert",
    }
    SIDECAR.write_text(json.dumps(sidecar) + "\n", encoding="utf-8")

    size_p1 = write_phase1_interrupted_turn()

    # Start fake socket server.
    sock_thread = threading.Thread(target=_socket_server, daemon=True)
    sock_thread.start()
    time.sleep(0.2)
    if not SOCK.exists():
        print("FATAL: fake socket did not bind", file=sys.stderr)
        return 2

    # Start the real server.
    env = dict(os.environ)
    env["CODEX_WEB_PASSWORD"] = PASSWORD
    env["CODEX_WEB_PORT"] = str(PORT)
    env["CODEX_WEB_HOST"] = HOST
    env["CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS"] = "0.2"
    global server_proc
    log_out = open(ART / "server.stdout.log", "w", encoding="utf-8")
    log_err = open(ART / "server.stderr.log", "w", encoding="utf-8")
    server_proc = subprocess.Popen(
        [sys.executable, "-m", "codoxear.server"],
        cwd=str(ART),
        env=env,
        stdout=log_out,
        stderr=log_err,
    )

    try:
        if not wait_for_port(20.0):
            print("FATAL: server did not listen", file=sys.stderr)
            return 2
        time.sleep(0.5)

        opener, _cj = make_opener()
        if not login(opener):
            print("FATAL: login failed", file=sys.stderr)
            return 2

        # PHASE 1: interrupted turn, no post-interrupt activity.
        p1_busy = poll_busy(opener, rounds=3, interval=0.6)
        p1_rows = fetch_sessions(opener)
        snapshots.append({
            "phase": 1,
            "description": "interrupted non-final turn; stale socket interrupted_idle=true; no post-interrupt activity",
            "log_size": LOG.stat().st_size,
            "log_idle_verdict_expectation": "non-idle (non-final assistant response_item), but override masks -> busy=false",
            "busy_samples": p1_busy,
            "session_row": find_row(p1_rows) if p1_rows else None,
            "state_calls_so_far": state_call_count["n"],
        })

        baseline_size = LOG.stat().st_size

        # PHASE 2: append post-interrupt resumed activity (new user turn).
        size_p2 = append_rows([
            {"type": "event_msg", "payload": {"type": "user_message", "message": "second"}, "ts": 20.0},
        ])
        # Discovery-refresh race discriminator: wait beyond
        # CODEX_WEB_DISCOVER_MIN_INTERVAL_SECONDS so the first phase-2
        # /api/sessions poll runs discovery before update_meta_counters().
        time.sleep(0.35)
        p2_busy = poll_busy(opener, rounds=5, interval=0.7)
        p2_rows = fetch_sessions(opener)
        snapshots.append({
            "phase": 2,
            "description": "post-interrupt user_message appended to SAME log; log idle parser sees non-idle; socket still reports interrupted_idle=true",
            "log_size_before": baseline_size,
            "log_size_after": size_p2,
            "log_idle_verdict_expectation": "non-idle (open user turn) -> busy should be true IF override cleared",
            "busy_samples": p2_busy,
            "session_row": find_row(p2_rows) if p2_rows else None,
            "state_calls_so_far": state_call_count["n"],
        })

        # PHASE 3: append completion -> log idle.
        size_p3 = append_rows([
            {"type": "event_msg", "payload": {"type": "task_complete"}, "ts": 30.0},
        ])
        p3_busy = poll_busy(opener, rounds=5, interval=0.7)
        p3_rows = fetch_sessions(opener)
        snapshots.append({
            "phase": 3,
            "description": "task_complete appended -> log idle",
            "log_size_before": size_p2,
            "log_size_after": size_p3,
            "log_idle_verdict_expectation": "idle -> busy=false regardless of override",
            "busy_samples": p3_busy,
            "session_row": find_row(p3_rows) if p3_rows else None,
            "state_calls_so_far": state_call_count["n"],
        })

        diag = in_process_diagnostic()

        # Preserve log + sidecar snapshots.
        import shutil
        shutil.copy(LOG, ART / "rollout-broker-1.final.jsonl")
        shutil.copy(SIDECAR, ART / "broker-1.sidecar.json")
        (ART / "api-snapshots.json").write_text(json.dumps(snapshots, indent=2), encoding="utf-8")
        (ART / "in-process-diagnostic.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")

        # Decision.
        p2_last = p2_busy[-1] if p2_busy else None
        p1_last = p1_busy[-1] if p1_busy else None
        p3_last = p3_busy[-1] if p3_busy else None
        verdict = "FAIL" if (p2_last is False) else ("PASS" if (p2_last is True) else "INCONCLUSIVE")
        (ART / "VERDICT.txt").write_text(
            f"verdict={verdict}\nphase1_busy_last={p1_last}\nphase2_busy_last={p2_last}\nphase3_busy_last={p3_last}\n",
            encoding="utf-8",
        )

        print(json.dumps({
            "verdict": verdict,
            "phase1_busy": p1_busy,
            "phase2_busy": p2_busy,
            "phase3_busy": p3_busy,
            "diag": diag,
        }, indent=2))
        return 0
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        socket_stop.set()
        log_out.close()
        log_err.close()


if __name__ == "__main__":
    raise SystemExit(main())
