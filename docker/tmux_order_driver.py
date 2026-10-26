#!/usr/bin/env python3
"""Behavioural verifier: tmux window order mirrors the sidebar order.

Runs inside an isolated container (see `docker/tmux-order-verify.Dockerfile`)
with a real codoxear server and real tmux. Sessions are launched through the
public API with `create_in_tmux`, so the windows, the broker sidecars and the
listing rows are all produced by the production path; only the agent binary is a
stub (`PI_BIN`) that prints its working directory forever.

Verified contracts:
  1. codoxear windows end up in the same relative order as `/api/sessions`
     (which is what the browser renders).
  2. a tmux window codoxear does not own keeps its index through every reorder.
  3. the viewer keeps looking at the same pane content while its window's index
     changes (this is what the `-d` decision exists for).
  4. snoozing and un-snoozing a session — a real ordering change, not a prompt
     to re-sort — is followed by tmux.
  5. the order is stable afterwards (no oscillation).
"""

from __future__ import annotations

import http.cookiejar
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

PASSWORD = "tmux-order-verify-password"
PORT = 19555
BASE = f"http://127.0.0.1:{PORT}"
TMUX_SESSION = "codoxear"
FOREIGN_WINDOW = "user-shell"
WINDOW_FORMAT = "#{window_index}\t#{window_id}\t#{window_active}\t#{window_name}"
FAKE_AGENT = "/home/tester/fake-agent"
SERVER_LOG = "/home/tester/server.log"
WORKSPACES = ["/tmp/ws/alpha", "/tmp/ws/beta", "/tmp/ws/gamma"]
CONVERGE_TIMEOUT = 90.0


class VerificationError(RuntimeError):
    pass


def check(condition: object, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def log(message: str) -> None:
    print(message, flush=True)


# -- tmux ---------------------------------------------------------------


def tmux(*args: str) -> str:
    proc = subprocess.run(["tmux", *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise VerificationError(f"tmux {' '.join(args)} failed: {(proc.stderr or proc.stdout).strip()}")
    return proc.stdout


def tmux_soft(*args: str) -> str:
    proc = subprocess.run(["tmux", *args], capture_output=True, text=True, check=False)
    return proc.stdout


def windows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in tmux("list-windows", "-t", TMUX_SESSION, "-F", WINDOW_FORMAT).splitlines():
        if not line.strip():
            continue
        index, window_id, active, name = line.split("\t", 3)
        rows.append({"index": int(index), "id": window_id, "active": active == "1", "name": name})
    return sorted(rows, key=lambda row: int(row["index"]))


def codoxear_windows() -> list[dict[str, object]]:
    return [window for window in windows() if window["name"] != FOREIGN_WINDOW]


def codoxear_window_order() -> list[str]:
    return [str(window["name"]) for window in codoxear_windows()]


def foreign_index() -> int | None:
    for window in windows():
        if window["name"] == FOREIGN_WINDOW:
            return int(window["index"])
    return None


def current_window_id() -> str:
    listing = tmux("display-message", "-p", "-t", f"{TMUX_SESSION}:", "#{window_id}")
    return listing.strip()


def current_pane_marker() -> str | None:
    text = tmux("capture-pane", "-p", "-t", f"{TMUX_SESSION}:", "-S", "-40")
    markers = [line.strip() for line in text.splitlines() if line.strip().startswith("AGENT:")]
    return markers[-1] if markers else None


# -- HTTP ---------------------------------------------------------------


def make_client() -> tuple[urllib.request.OpenerDirector, http.cookiejar.CookieJar]:
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    return opener, jar


def call(opener: urllib.request.OpenerDirector, method: str, path: str, body: object | None = None) -> tuple[int, object]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    request = urllib.request.Request(BASE + path, data=data, method=method)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    try:
        with opener.open(request, timeout=30) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    try:
        return status, json.loads(raw.decode("utf-8"))
    except ValueError:
        return status, raw.decode("utf-8", "replace")


def sessions(opener: urllib.request.OpenerDirector) -> list[dict[str, object]]:
    status, body = call(opener, "GET", "/api/sessions")
    check(status == 200 and isinstance(body, dict), f"GET /api/sessions failed: {status} {body}")
    rows = body.get("sessions")
    check(isinstance(rows, list), "GET /api/sessions returned no sessions array")
    return [row for row in rows if isinstance(row, dict)]


def sidebar_order(rows: list[dict[str, object]]) -> list[str]:
    """The rendered order, reduced to the windows it names."""
    return [str(row["tmux_window"]) for row in rows if isinstance(row.get("tmux_window"), str) and row["tmux_window"]]


# -- setup --------------------------------------------------------------


def write_fake_agent() -> None:
    script = """#!/bin/sh
case "$1" in
  --help|-h|--version|-v) echo "fake agent"; exit 0 ;;
esac
while :; do printf 'AGENT:%s\\n' "$PWD"; sleep 2; done
"""
    with open(FAKE_AGENT, "w", encoding="utf-8") as handle:
        handle.write(script)
    os.chmod(FAKE_AGENT, 0o755)


def prepare_workspaces() -> None:
    for workspace in WORKSPACES:
        os.makedirs(workspace, exist_ok=True)


def start_server() -> subprocess.Popen[bytes]:
    env = dict(os.environ)
    env.update(
        {
            "CODEX_WEB_PASSWORD": PASSWORD,
            "CODEX_WEB_HOST": "127.0.0.1",
            "CODEX_WEB_PORT": str(PORT),
            "CODEX_WEB_DEFAULT_AGENT_BACKEND": "pi",
            "CODEX_WEB_TMUX_SESSION": TMUX_SESSION,
            "PI_BIN": FAKE_AGENT,
        }
    )
    log_handle = open(SERVER_LOG, "wb")
    process = subprocess.Popen(
        [sys.executable, "-m", "codoxear.server"],
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    return process


def wait_for_server() -> None:
    deadline = time.time() + 60.0
    opener, _ = make_client()
    while time.time() < deadline:
        try:
            status, _ = call(opener, "GET", "/")
        except OSError:
            status = 0
        if status in (200, 401):
            return
        time.sleep(0.25)
    raise VerificationError("server did not become reachable; see " + SERVER_LOG)


# -- the verification ---------------------------------------------------


def wait_for_convergence(opener: urllib.request.OpenerDirector, *, expected: list[str] | None = None, note: str = "") -> list[dict[str, object]]:
    deadline = time.time() + CONVERGE_TIMEOUT
    observed: tuple[list[str], list[str]] | None = None
    while time.time() < deadline:
        # Every GET is a listing generation, exactly like an open sidebar polling.
        rows = sessions(opener)
        wanted = sidebar_order(rows)
        actual = codoxear_window_order()
        observed = (actual, wanted)
        if actual == wanted and (expected is None or wanted == expected):
            return rows
        time.sleep(1.0)
    raise VerificationError(f"tmux order did not follow the sidebar {note}: tmux={observed[0]} sidebar={observed[1]} expected={expected}")


def launch_sessions(opener: urllib.request.OpenerDirector) -> None:
    for workspace in WORKSPACES:
        status, body = call(
            opener,
            "POST",
            "/api/sessions",
            {"cwd": workspace, "agent_backend": "pi", "create_in_tmux": True},
        )
        check(status == 200 and isinstance(body, dict) and body.get("tmux_window"), f"launching {workspace} in tmux failed: {status} {body}")
        log(f"launched {workspace} -> tmux window {body['tmux_window']}")


def wait_for_rows(opener: urllib.request.OpenerDirector, count: int) -> list[dict[str, object]]:
    deadline = time.time() + CONVERGE_TIMEOUT
    while time.time() < deadline:
        rows = sessions(opener)
        if len([row for row in rows if row.get("tmux_window")]) >= count:
            return rows
        time.sleep(0.5)
    raise VerificationError(f"expected {count} tmux-backed sessions, saw {sidebar_order(sessions(opener))}")


def main() -> int:
    prepare_workspaces()
    write_fake_agent()
    tmux_soft("kill-server")
    server = start_server()
    try:
        wait_for_server()
        opener, _ = make_client()
        status, body = call(opener, "POST", "/api/login", {"password": PASSWORD})
        check(status == 200, f"login failed: {status} {body}")

        # A window codoxear does not own, created before any session so it holds
        # index 0 for the whole run.
        tmux("new-session", "-d", "-s", TMUX_SESSION, "-n", FOREIGN_WINDOW, "sleep 3000")
        check(foreign_index() == 0, "foreign window did not start at index 0")

        launch_sessions(opener)
        rows = wait_for_rows(opener, len(WORKSPACES))
        expected = sidebar_order(rows)
        check(len(expected) == len(WORKSPACES), f"unexpected sidebar order: {expected}")

        rows = wait_for_convergence(opener, expected=expected, note="after launch")
        check(foreign_index() == 0, f"foreign window moved during the first convergence: {windows()}")
        order_after_launch = codoxear_window_order()
        log(f"PASS initial order: sidebar={expected} tmux={order_after_launch}")

        # A real ordering change: snooze the session the sidebar ranks first.
        top_session = str(rows[0]["session_id"])
        moving_window = str(rows[-1]["tmux_window"])
        target = next(window for window in codoxear_windows() if window["name"] == moving_window)
        tmux("select-window", "-t", f"{TMUX_SESSION}:{target['index']}")
        window_id_before = current_window_id()
        marker_before = current_pane_marker()
        check(marker_before is not None, "could not read the pane marker of the selected window")

        status, body = call(
            opener,
            "POST",
            f"/api/sessions/{urllib.parse.quote(top_session)}/edit",
            {"name": "", "priority_offset": 0, "snooze_until": int(time.time()) + 86400},
        )
        check(status == 200, f"snoozing the top session failed: {status} {body}")

        snoozed_rows = wait_for_convergence(opener, note="after snoozing the top session")
        check(
            bool(snoozed_rows[-1].get("snoozed")) and str(snoozed_rows[-1]["session_id"]) == top_session,
            f"snoozed session is not ranked last: {[(row.get('session_id'), row.get('snoozed')) for row in snoozed_rows]}",
        )
        check(foreign_index() == 0, f"foreign window moved while snoozing: {windows()}")
        check(current_window_id() == window_id_before, "the viewer's window changed during the reorder")
        check(current_pane_marker() == marker_before, "the viewer's pane content changed during the reorder")
        log(f"PASS snooze reorder: order={codoxear_window_order()} viewer kept {marker_before}")

        # A second real ordering change: un-snooze, which must restore the order.
        status, body = call(
            opener,
            "POST",
            f"/api/sessions/{urllib.parse.quote(top_session)}/edit",
            {"name": "", "priority_offset": 0, "snooze_until": None},
        )
        check(status == 200, f"un-snoozing the session failed: {status} {body}")
        wait_for_convergence(opener, note="after un-snoozing")
        check(codoxear_window_order() == order_after_launch, f"un-snooze did not restore the original order: {codoxear_window_order()}")
        check(len(windows()) == len(WORKSPACES) + 1, f"the reorder changed the window set: {windows()}")
        log(f"PASS restore order: order={codoxear_window_order()}")

        # Stability: keep generating listing generations and confirm nothing drifts.
        for _ in range(6):
            sessions(opener)
            time.sleep(2.0)
        check(codoxear_window_order() == order_after_launch, f"order drifted while idle: {codoxear_window_order()}")
        check(foreign_index() == 0, "foreign window moved while idle")
        log("PASS stable while idle")

        log("TMUX ORDER VERIFY: PASS")
        return 0
    except VerificationError as exc:
        log(f"TMUX ORDER VERIFY: FAIL — {exc}")
        log("--- tmux windows ---")
        log(tmux_soft("list-windows", "-t", TMUX_SESSION, "-F", WINDOW_FORMAT))
        log("--- server log tail ---")
        try:
            with open(SERVER_LOG, "r", encoding="utf-8", errors="replace") as handle:
                log("".join(handle.readlines()[-80:]))
        except OSError:
            pass
        return 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        tmux_soft("kill-server")


if __name__ == "__main__":
    sys.exit(main())
