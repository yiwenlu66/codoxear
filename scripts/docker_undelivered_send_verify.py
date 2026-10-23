#!/usr/bin/env python3
"""Drive scripts/docker_undelivered_send_verify.sh through HTTP and a real browser."""
import argparse
import atexit
import base64
import hashlib
import http.cookiejar
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--password", required=True)
ap.add_argument("--container", required=True)
ap.add_argument("--artifacts", required=True)
ap.add_argument("--commit", required=True)
ap.add_argument("--browser-session", required=True)
ap.add_argument("--mounts", required=True)
args = ap.parse_args()

ART = Path(args.artifacts)
BASE = f"http://127.0.0.1:{args.port}"
PROJ = "/home/tester/proj"
APP = "/home/tester/.local/share/codoxear"
BRIDGE_DIR = APP + "/pi-active-sessions"
# Broker default for CODEX_WEB_BUSY_QUIET_SECONDS; the send latch may settle no
# earlier than this after the last observed backend activity.
BUSY_QUIET_SECONDS = 3.0
# A fresh Pi session emits no rollout file until its first assistant row, so the
# only log-independent proof that it took the prompt is what its extension
# reports. This verifier therefore requires the acceptance receipt of the slow
# scenario. Historical commits (pre-receipt) can still be compared by setting
# CODOXEAR_UDV_REQUIRE_RECEIPT=0, which keeps the receipt measurements out of
# the pass/fail set but still reports them.
REQUIRE_ACCEPTANCE_RECEIPT = os.environ.get("CODOXEAR_UDV_REQUIRE_RECEIPT", "1") != "0"
T0 = time.time()
jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
report: dict = {
    "commit": args.commit,
    "base_url": BASE,
    "started_at": T0,
    "acceptance_receipt_required": REQUIRE_ACCEPTANCE_RECEIPT,
    "busy_quiet_seconds": BUSY_QUIET_SECONDS,
    "scenarios": {
        "1_gate_swallows_browser_send": {"pass": False},
        "2_server_publishes_undelivered_verdict": {"pass": False},
        "3_browser_restores_composer_and_toasts": {"pass": False},
        "4_second_send_delivers_without_new_verdict": {"pass": False},
        "1b_gate_produces_no_acceptance_receipt": {"pass": False},
        "5_slow_first_reply_of_fresh_session_is_not_undelivered": {"pass": False},
        "6_working_indicator_suppressed_receipt_alone_prevents_verdict": {"pass": False},
        "7_whitespace_prompt_still_matches_receipt_digest": {"pass": False},
    },
    "evidence": {"artifacts": {}},
}


def log(message: str) -> None:
    print(f"[{time.time() - T0:6.2f}s] {message}", flush=True)


def write_report() -> None:
    report["finished_at"] = time.time()
    report["wall_clock_seconds"] = round(time.time() - T0, 3)
    (ART / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")


atexit.register(write_report)


def http(method: str, path: str, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with opener.open(request, timeout=20) as response:
            raw = response.read()
            return response.status, json.loads(raw or b"null")
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            value = json.loads(raw)
        except Exception:
            value = raw.decode(errors="replace")
        return error.code, value
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        return 0, {"error": str(error)}


def dx(command: str, timeout: int = 30) -> str:
    completed = subprocess.run(
        ["sudo", "docker", "exec", args.container, "sh", "-lc", command],
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    return completed.stdout + completed.stderr


browser_log: list[dict] = []


def ab(*command: str, timeout: int = 40) -> str:
    env = {"AGENT_BROWSER_SESSION": args.browser_session, "PATH": str(Path("/usr/local/sbin")) + ":" + __import__("os").environ["PATH"], "HOME": __import__("os").environ["HOME"]}
    completed = subprocess.run(["agent-browser", *command], text=True, capture_output=True, timeout=timeout, env=env)
    output = completed.stdout + completed.stderr
    browser_log.append({"command": list(command), "returncode": completed.returncode, "output": output[-1000:]})
    (ART / "browser-commands.json").write_text(json.dumps(browser_log, indent=2))
    if completed.returncode:
        raise RuntimeError(f"agent-browser {' '.join(command)} failed: {output[-800:]}")
    return output


def ab_eval(script: str):
    raw = ab("eval", script, "--json")
    try:
        return json.loads(raw)["data"]["result"]
    except Exception as error:
        raise RuntimeError(f"could not decode agent-browser eval output: {raw[-800:]}") from error


def wait_for(predicate, timeout: float, step: float = 0.25, label: str = "condition"):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(step)
    report["evidence"].setdefault("timeouts", []).append({"label": label, "timeout": timeout, "last": last})
    return None


def session_row(session_id: str | None = None):
    status, data = http("GET", "/api/sessions")
    if status != 200 or not isinstance(data, dict):
        return None
    rows = data.get("sessions") or []
    if session_id is not None:
        return next((row for row in rows if row.get("session_id") == session_id), None)
    return next((row for row in rows if row.get("cwd") == PROJ and row.get("agent_backend") == "pi"), None)


def pane(target: str) -> str:
    return dx(f"tmux capture-pane -p -t {shlex.quote(target)} -S -80")


def docker_jsonl_files() -> list[str]:
    raw = dx("find /home/tester/.pi/agent/sessions -type f -name '*.jsonl' -print 2>/dev/null")
    return [line for line in raw.splitlines() if line.startswith("/")]


def pi_user_row(mark: str):
    for path in docker_jsonl_files():
        raw = dx(f"cat {shlex.quote(path)}")
        for line in raw.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            message = row.get("message") or {}
            if row.get("type") == "message" and message.get("role") == "user" and mark in json.dumps(message, ensure_ascii=False):
                return {"file": path, "row": row}
    return None


def journal_rows(session_id: str) -> list[dict]:
    raw = dx(f"cat {shlex.quote(APP + '/send_journal/' + session_id + '.jsonl')} 2>/dev/null")
    result = []
    for line in raw.splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return result


def dump_json(name: str, value) -> None:
    (ART / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> None:
    try:
        mounts = json.loads(Path(args.mounts).read_text())
    except Exception as error:
        fail(f"cannot read Docker mount evidence: {error}")
    expected_mount = [m for m in mounts if m.get("Destination") == "/home/tester"]
    isolation_ok = len(mounts) == 1 and len(expected_mount) == 1 and expected_mount[0].get("Type") == "bind"
    report["evidence"]["container_mounts"] = mounts
    report["evidence"]["runtime_isolation"] = {
        "pass": isolation_ok,
        "reason": "only the throwaway container HOME is bind-mounted; Codoxear runtime resolves under /home/tester/.local/share/codoxear",
    }
    if not isolation_ok:
        fail(f"unexpected container mounts: {mounts}")

    if not wait_for(lambda: http("GET", "/api/me")[0] == 401, 90, 0.5, "server startup"):
        fail("server never became available")
    status, login = http("POST", "/api/login", {"password": args.password})
    report["evidence"]["api_login"] = {"status": status, "response": login}
    if status != 200:
        fail(f"API login failed: {status} {login}")

    launch_started = time.time()
    status, launch = http("POST", "/api/sessions", {
        "cwd": PROJ,
        "agent_backend": "pi",
        "model_provider": "fake",
        "model": "fake-1",
        "create_in_tmux": True,
    })
    report["evidence"]["launch_response"] = {"status": status, "response": launch}
    if status != 200:
        fail(f"session launch failed: {status} {launch}")
    row = wait_for(session_row, 90, 0.25, "Pi session listed")
    if not row:
        fail("Pi session never appeared in GET /api/sessions")
    sid = row["session_id"]
    target = f"{row.get('tmux_session')}:{row.get('tmux_window')}"
    report["evidence"].update(session_id=sid, tmux_target=target, launch_to_listed_seconds=round(time.time() - launch_started, 3))
    dump_json("session-row-before-send.json", row)

    pane_samples = []
    gate_pane = None
    deadline = time.time() + 30
    while time.time() < deadline:
        text = pane(target)
        pane_samples.append({"ts": time.time(), "text": text})
        if "Press any key to continue" in text:
            gate_pane = text
            break
        time.sleep(0.5)
    (ART / "pane-samples-before-send.json").write_text(json.dumps(pane_samples, indent=2))
    (ART / "pane-before-send.txt").write_text(gate_pane or (pane_samples[-1]["text"] if pane_samples else ""))
    report["evidence"]["startup_gate"] = {
        "seen": gate_pane is not None,
        "pane_text": (gate_pane or "")[-3000:],
    }
    if gate_pane is None:
        fail("Pi startup deprecation gate was absent; refusing to test a substitute condition")

    # The browser, rather than the API client above, submits this first prompt.
    mark1 = f"UDV-swallowed-{int(time.time() * 1000)}"
    report["evidence"]["first_prompt"] = mark1
    ab("open", BASE + "/")
    ab("wait", "#pw")
    ab("fill", "#pw", args.password)
    ab("click", "#loginBtn")
    if not wait_for(lambda: ab_eval('Boolean(document.querySelector("#sessions"))'), 25, 0.25, "browser application shell"):
        fail("browser did not render the authenticated application shell")
    selector = f'#sessions .session[data-session-id="{sid}"]'
    if not wait_for(lambda: ab_eval(f'Boolean(document.querySelector({json.dumps(selector)}))'), 30, 0.25, "session card in browser"):
        fail("browser did not render the launched session card")
    ab("click", selector)
    if not wait_for(lambda: ab_eval('!document.querySelector("#msg").disabled'), 25, 0.25, "composer enabled"):
        fail("composer did not become enabled for selected session")
    # A selected session can retain a server draft. Clear it through the real
    # textarea before typing so only this marked prompt can be submitted.
    ab("click", "#msg")
    ab("press", "Control+a")
    ab("press", "Delete")
    if not wait_for(lambda: ab_eval('document.querySelector("#msg").value') == "", 5, 0.1, "composer cleared before send"):
        fail("could not clear composer before browser send")
    ab("type", "#msg", mark1)
    if ab_eval('document.querySelector("#msg").value') != mark1:
        fail("browser did not place exact first prompt in composer before submit")

    action_state = ab_eval('(() => { const msg = document.querySelector("#msg"); const send = document.querySelector("#sendBtn"); const r = send && send.getBoundingClientRect(); return { composer: msg ? msg.value : null, send_disabled: !send || send.disabled, send_visible: !!r && r.width > 0 && r.height > 0, toast: (document.querySelector("#toast") || {}).textContent || "" }; })()')
    send_started = time.time()
    send_action = "click #sendBtn"
    send_action_fallback_reason = None
    try:
        if action_state["send_disabled"] or not action_state["send_visible"]:
            raise RuntimeError("#sendBtn was disabled or not visible")
        ab("click", "#sendBtn")
    except RuntimeError as click_error:
        # Ctrl+Enter is the application's documented keyboard submission path;
        # it is used only when the physical Send control cannot be clicked.
        send_action = "Ctrl+Enter"
        send_action_fallback_reason = str(click_error)
        ab("click", "#msg")
        ab("press", "Control+Enter")
    ab("screenshot", "--full", str(ART / "browser-after-first-send.png"))

    post_action_samples = []
    def composer_cleared_after_send():
        state = ab_eval('(() => { const msg = document.querySelector("#msg"); const send = document.querySelector("#sendBtn"); const toast = document.querySelector("#toast"); return { composer: msg ? msg.value : null, send_disabled: !send || send.disabled, toast: toast ? toast.textContent : "" }; })()')
        post_action_samples.append({"elapsed_seconds": round(time.time() - send_started, 3), **state})
        return state if state.get("composer") == "" else None

    local_clear_state = wait_for(composer_cleared_after_send, 10, 0.1, "composer clear after browser send")
    report["evidence"]["browser_send_action"] = {
        "action": send_action,
        "fallback_reason": send_action_fallback_reason,
        "before_action": action_state,
        "post_action_samples": post_action_samples,
        "local_clear_state": local_clear_state,
    }
    if local_clear_state is None:
        last = post_action_samples[-1] if post_action_samples else {}
        fail(f"browser send did not fire: composer={last.get('composer')!r}, send_disabled={last.get('send_disabled')!r}, toast={last.get('toast')!r}")

    # The fsync'd submit row is the durable record of the browser submission;
    # capture it before waiting for the broker's quiet-window verdict.
    submit = wait_for(lambda: next((entry for entry in journal_rows(sid) if entry.get("type") == "submit" and entry.get("display_text") == mark1), None), 10, 0.1, "first send journal submit")
    journal_at_submit = journal_rows(sid)
    dump_json("first-send-journal.json", journal_at_submit)
    report["evidence"]["journal_after_browser_send"] = {"submit": submit, "rows": journal_at_submit}
    if submit is None:
        fail("browser send cleared the composer but produced no matching send-journal submit row")

    samples = []
    verdict_row = None
    restored = False
    toast_seen = None
    deadline = time.time() + 60
    while time.time() < deadline:
        browser_state = ab_eval('(() => { const msg = document.querySelector("#msg"); const toast = document.querySelector("#toast"); return { composer: msg ? msg.value : null, toast: toast ? toast.textContent : null }; })()')
        current = session_row(sid)
        verdict = current.get("send_verdict") if current else None
        sample = {"wall_ts": time.time(), "elapsed_seconds": round(time.time() - send_started, 3), "composer": browser_state.get("composer"), "toast": browser_state.get("toast"), "send_verdict": verdict}
        samples.append(sample)
        if isinstance(browser_state.get("toast"), str) and "did not reach the backend" in browser_state["toast"]:
            toast_seen = browser_state["toast"]
        if isinstance(verdict, dict) and verdict.get("verdict") == "undelivered":
            verdict_row = current
        if browser_state.get("composer") == mark1:
            restored = True
        if verdict_row and restored and toast_seen:
            break
        time.sleep(0.25)
    dump_json("browser-verdict-poll.json", samples)
    ab("screenshot", "--full", str(ART / "browser-after-verdict.png"))
    pane_after_first = pane(target)
    (ART / "pane-after-first-send.txt").write_text(pane_after_first)
    row_after = session_row(sid)
    dump_json("session-row-after-verdict.json", row_after)

    verdict = row_after.get("send_verdict") if row_after else None
    verdict_latency = None
    if isinstance(verdict, dict) and isinstance(verdict.get("ts"), (int, float)):
        verdict_latency = round(float(verdict["ts"]) - send_started, 3)
    report["evidence"].update(
        first_send_started_at=send_started,
        first_verdict_latency_seconds=verdict_latency,
        first_send_poll_samples=samples,
        first_verdict=verdict,
        composer_value_after_restore=ab_eval('document.querySelector("#msg") && document.querySelector("#msg").value'),
        toast_containing_backend_message=toast_seen,
        pane_after_first_send=pane_after_first[-3000:],
    )

    jsonl_paths = docker_jsonl_files()
    mark1_occurrences = []
    for path in jsonl_paths:
        raw = dx(f"grep -n -F -- {shlex.quote(mark1)} {shlex.quote(path)} 2>/dev/null || true")
        if raw.strip():
            mark1_occurrences.append({"file": path, "matches": raw})
    dump_json("first-prompt-pi-log-search.json", {"paths": jsonl_paths, "occurrences": mark1_occurrences})

    journal = journal_rows(sid)
    dump_json("first-send-journal.json", journal)
    submit = next((entry for entry in journal if entry.get("type") == "submit" and entry.get("display_text") == mark1), None)
    all_rows_status, all_sessions = http("GET", "/api/sessions")
    verdict_holders = [candidate.get("session_id") for candidate in (all_sessions.get("sessions") if isinstance(all_sessions, dict) else []) if candidate.get("send_verdict")]
    correlation_direct = bool(submit and isinstance(verdict, dict) and submit.get("id") == verdict.get("send_id"))
    correlation_fallback = bool(isinstance(verdict, dict) and verdict.get("send_id") and verdict_holders == [sid])
    report["evidence"]["send_id_correlation"] = {
        "journal_submit": submit,
        "verdict_send_id": verdict.get("send_id") if isinstance(verdict, dict) else None,
        "journal_submit_id_matches_verdict": correlation_direct,
        "fallback_used": not correlation_direct,
        "fallback_proof": "verdict id is non-empty and this is the only session with a verdict",
        "verdict_holders": verdict_holders,
    }

    scenario1 = not mark1_occurrences
    report["scenarios"]["1_gate_swallows_browser_send"] = {
        "pass": scenario1,
        "gate_seen": gate_pane is not None,
        "prompt_absent_from_all_pi_jsonl": not mark1_occurrences,
        "searched_jsonl_files": jsonl_paths,
        "pane_capture": "pane-before-send.txt",
    }
    scenario2 = bool(isinstance(verdict, dict) and verdict.get("verdict") == "undelivered" and isinstance(verdict.get("send_id"), str) and verdict.get("send_id") and isinstance(verdict.get("ts"), (int, float)) and (correlation_direct or correlation_fallback))
    report["scenarios"]["2_server_publishes_undelivered_verdict"] = {
        "pass": scenario2,
        "row_send_verdict": verdict,
        "verdict_latency_seconds": verdict_latency,
        "send_id_correlation": "journal-id" if correlation_direct else "fallback-only-verdict-holder",
    }
    scenario3 = bool(restored and toast_seen and ab_eval('document.querySelector("#msg").value') == mark1)
    report["scenarios"]["3_browser_restores_composer_and_toasts"] = {
        "pass": scenario3,
        "composer_value": ab_eval('document.querySelector("#msg").value'),
        "toast_seen": toast_seen,
        "screenshots": ["browser-after-first-send.png", "browser-after-verdict.png"],
    }

    # The swallowed prompt is the receipt mechanism's positive control: if the
    # gate ate the text before Pi's agent loop, no user message_start fires, so
    # no acceptance receipt may exist for it. Silence (no receipt) + verdict on
    # the swallowed side, receipt + no verdict on the accepted side, is what makes
    # the receipt the discriminating evidence rather than a coincidence.
    gate_broker_pid = (row_after or {}).get("broker_pid") or (row or {}).get("broker_pid")
    gate_bridge = container_snapshot(mark1, gate_broker_pid)
    gate_receipts = _receipt_candidates(gate_bridge["bridge"], mark1, send_started)
    dump_json("gate-bridge-artifacts.json", gate_bridge["bridge"])
    report["evidence"]["swallowed_prompt_receipt"] = {
        "broker_pid": gate_broker_pid,
        "artifact_names": sorted(gate_bridge["bridge"]),
        "receipt_for_prompt": gate_receipts,
        "expected": "none: the startup gate consumes the prompt before Pi's agent loop, so no user message_start can fire",
    }
    report["scenarios"]["1b_gate_produces_no_acceptance_receipt"] = {
        "pass": not gate_receipts,
        "receipt_candidates_for_swallowed_prompt": gate_receipts,
        "artifact_names": sorted(gate_bridge["bridge"]),
    }

    # Once the gate has consumed a key, Pi reaches its TUI. A direct second
    # send provides the negative control: it must create a Pi user row and not
    # overwrite the first verdict with the negative-control id.
    mark2 = f"UDV-delivered-{int(time.time() * 1000)}"
    send2_id = f"neg-{int(time.time() * 1000)}"
    second_started = time.time()
    status2, response2 = http("POST", f"/api/sessions/{sid}/send", {"text": mark2, "send_id": send2_id})
    landed2 = wait_for(lambda: pi_user_row(mark2), 45, 0.25, "negative-control Pi user row")
    time.sleep(12)
    row2 = session_row(sid)
    pane_after_second = pane(target)
    (ART / "pane-after-second-send.txt").write_text(pane_after_second)
    dump_json("session-row-after-negative-control.json", row2)
    verdict2 = row2.get("send_verdict") if row2 else None
    no_second_verdict = isinstance(verdict2, dict) and isinstance(verdict, dict) and verdict2.get("send_id") == verdict.get("send_id") and verdict2.get("send_id") != send2_id
    report["evidence"].update(
        negative_control={
            "started_at": second_started,
            "http_status": status2,
            "response": response2,
            "prompt": mark2,
            "send_id": send2_id,
            "pi_user_row": landed2,
            "row_after_wait": row2,
            "pane_after_second_send": pane_after_second[-3000:],
        }
    )
    report["scenarios"]["4_second_send_delivers_without_new_verdict"] = {
        "pass": bool(status2 == 200 and landed2 and no_second_verdict),
        "send_status": status2,
        "user_row_written": bool(landed2),
        "first_verdict": verdict,
        "verdict_after_second_send": verdict2,
        "negative_send_id": send2_id,
        "waited_seconds_after_second_send": round(time.time() - second_started, 3),
    }


SLOW_PROVIDER_PORT = 19694
SLOW_PROVIDER_DELAY_SECONDS = 6.0
SLOW_PROVIDER_SOURCE = r"""
import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
DELAY = float(sys.argv[2])
LOG = open(sys.argv[3], "a", buffering=1)
def note(event, **extra):
    LOG.write(json.dumps({"ts": time.time(), "event": event, **extra}) + "\n")
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def log_message(self, *a): pass
    def do_GET(self):
        body = b'{"ok":true}'
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        note("request", path=self.path, bytes=len(raw))
        time.sleep(DELAY)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream"); self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close"); self.end_headers()
        def chunk(delta, finish=None, usage=None, choices=True):
            obj = {"id": "chatcmpl-slow", "object": "chat.completion.chunk", "created": int(time.time()), "model": "slow-1",
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}] if choices else []}
            if usage: obj["usage"] = usage
            self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode()); self.wfile.flush()
        chunk({"role": "assistant", "content": ""})
        chunk({"content": "slow-reply-ok"})
        chunk({}, finish="stop")
        chunk({}, usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}, choices=False)
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
        note("response_sent")
        self.close_connection = True
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
"""


SNAPSHOT_MARKED = "=== CODOXEAR-MARKED"
SNAPSHOT_BRIDGE = "=== CODOXEAR-BRIDGE"
SNAPSHOT_BODIES = "=== CODOXEAR-BODIES"

# Keys whose value is the moment the backend accepted a prompt. Only these can
# date a receipt: a marker's session-start ``updatedAt`` already exists before
# the send and would otherwise be mistaken for acceptance.
_ACCEPT_TIME_KEYS = {
    "acceptedat", "acceptedts", "acceptedtime", "acceptedtimestamp",
    "receiptat", "receiptts", "receipttime", "acceptanceat", "acceptancets",
    "receivedat", "receivedts", "submittedat", "submittedts",
    "promptacceptedat", "promptacceptedts", "promptat", "promptts",
    "lastpromptat", "lastpromptts", "turnstartedat",
    # The Pi bridge receipt dates acceptance with the user message's own
    # event timestamp (epoch milliseconds), not with a file write time.
    "messagetimestamp", "messagetime", "messagets", "messagetimestampms",
    "promptmessagetimestamp",
}


def _parse_epoch(value) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        # Epoch milliseconds/microseconds are common in extension payloads.
        if number > 1e11:
            number = number / (1000.0 if number < 1e14 else 1e6)
        return number
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        return _parse_epoch(float(raw))
    except ValueError:
        pass
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _is_accept_time_key(trail: tuple[str, ...]) -> bool:
    """Whether this JSON path names the moment a prompt was accepted.

    A bare ``ts``/``at`` only counts inside an acceptance-shaped ancestor, so a
    marker's session-start ``updatedAt`` (which predates the send) is never
    mistaken for a receipt.
    """
    if not trail:
        return False
    leaf = re.sub(r"[^a-z0-9]", "", trail[-1].lower())
    if leaf in _ACCEPT_TIME_KEYS:
        return True
    if leaf not in {"ts", "at", "time", "timestamp", "when", "epoch", "millis", "ms"}:
        return False
    ancestors = " ".join(re.sub(r"[^a-z0-9]", "", part.lower()) for part in trail[:-1])
    return any(word in ancestors for word in ("accept", "receipt", "prompt", "submit", "received"))


def prompt_digests(text: str) -> dict[str, str]:
    """Digest token -> variant, for every plausible form of the submitted text.

    The broker injects the committed text plus Enter; the extension sees the
    submitted line. Digesting the trimmed/newline variants keeps the correlation
    independent of which side normalizes whitespace, and accepting truncated
    hex plus base64 forms keeps it independent of how the extension encodes the
    digest.
    """
    tokens: dict[str, str] = {}
    for variant in {text, text.strip(), text.rstrip(), text + "\n", text + "\r", text + "\r\n"}:
        if not variant:
            continue
        raw = hashlib.sha256(variant.encode()).digest()
        hexd = raw.hex()
        for token in (hexd, hexd[:32], hexd[:16], hashlib.sha1(variant.encode()).hexdigest()):
            tokens[token] = variant
        for encoded in (base64.b64encode(raw).decode("ascii"), base64.urlsafe_b64encode(raw).decode("ascii")):
            tokens[encoded] = variant
            tokens[encoded.rstrip("=")] = variant
    return tokens


def _digest_match(value: str, digests: dict[str, str]) -> tuple[str, str] | None:
    """Return (token, prompt variant) this artifact value digests, if any."""
    candidate = value.strip()
    for text in (candidate, candidate.lower()):
        if text in digests:
            return text, digests[text]
    return None


# Pi's TUI trims submitted text with JavaScript's String.prototype.trim()
# whitespace set, so a receipt names the normalized prompt rather than the raw
# composer bytes. The verifier measures that normalization explicitly instead of
# assuming the raw text and the ingested text are byte-identical.
_JS_TRIM_CHARS = "\t\n\v\f\r \u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"


def _stat_number(value) -> float:
    """Parse a numeric stat field, tolerating find's birth-time spelling.

    GNU find prints ``%B@`` as ``-1.-000000010`` on filesystems without a birth
    time, which is not a float literal; treat any non-positive result as unknown.
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = re.match(r"^\s*(-?\d+(?:\.\d+)?)", str(value or ""))
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


def _iter_json_leaves(node, trail: tuple[str, ...] = ()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _iter_json_leaves(value, trail + (str(key),))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _iter_json_leaves(value, trail + (str(index),))
    else:
        yield trail, node


def _receipt_candidates(artifacts: dict, prompt_text: str, since_ts: float) -> list[dict]:
    """Bridge artifacts that name this prompt and postdate the send.

    The Pi bridge's receipt contract (``writePromptReceipt`` in
    ``codoxear/pi_active_session_bridge.ts``) is one JSON object in
    ``<marker>.receipt`` which is ``broker-<broker_pid>.json.receipt`` on disk:
    ``{version: 1, pid: <Pi process pid>, textSha256: <sha256 hex of the ingested
    text>, messageTimestamp: <user message event time, epoch ms>}``. The broker
    accepts it only for the current idle-origin send, only from its own Pi pid,
    and only for the exact submitted text, then refuses to settle the send latch
    as undelivered while the receipt stands.

    The extraction stays tolerant of neighbouring field names because the
    receipt's *meaning* is what the verifier needs: an artifact that names this
    submission and dates its ingestion. Any digest/text match proves the naming;
    a receipt older than the send is a stale record of an earlier prompt (Pi's
    millisecond event clock is what rejects it in the broker too) and is dropped.
    """
    digests = prompt_digests(prompt_text)
    stripped_prompt = prompt_text.strip()
    candidates: list[dict] = []
    for name, artifact in artifacts.items():
        body = artifact.get("json")
        if not isinstance(body, dict):
            continue
        matches: list[dict] = []
        accepted: list[dict] = []
        length_match = False
        for trail, value in _iter_json_leaves(body):
            ts = _parse_epoch(value)
            if ts is not None and _is_accept_time_key(trail):
                accepted.append({"key": ".".join(trail), "ts": ts})
            key = re.sub(r"[^a-z0-9]", "", str(trail[-1]).lower()) if trail else ""
            if isinstance(value, str):
                matched = _digest_match(value, digests)
                if matched is not None:
                    matches.append({"kind": "digest", "key": ".".join(trail), "token": matched[0], "variant": matched[1]})
                elif stripped_prompt and stripped_prompt in value:
                    matches.append({"kind": "text", "key": ".".join(trail)})
            if key == "promptlength" and value == len(prompt_text):
                length_match = True
        if not matches:
            continue
        dated = max(accepted, key=lambda item: item["ts"]) if accepted else None
        if dated is not None:
            ts, source = dated["ts"], "payload:" + dated["key"]
        else:
            birth = _stat_number(artifact.get("birth"))
            mtime = _stat_number(artifact.get("mtime"))
            ts = mtime if mtime > 0.0 else birth
            source = "file:mtime" if mtime > 0.0 else "file:birth"
        if ts <= 0.0 or ts < since_ts - 1.0:
            continue
        candidates.append({
            "artifact": name,
            "match": matches[0],
            "matches": matches,
            "length_match": length_match,
            "ts": ts,
            "ts_source": source,
            "artifact_birth_ts": _stat_number(artifact.get("birth")),
            "artifact_mtime_ts": _stat_number(artifact.get("mtime")),
            "artifact_digest": matches[0].get("token") if matches[0]["kind"] == "digest" else None,
            "artifact_matched_text": matches[0].get("variant"),
            "artifact_pid": body.get("pid") if isinstance(body.get("pid"), int) and not isinstance(body.get("pid"), bool) else None,
            "artifact_session_file": body.get("sessionFile") if isinstance(body.get("sessionFile"), str) else None,
        })
    return candidates


def container_snapshot(mark: str, broker_pid: int | None = None) -> dict:
    """One docker exec: Pi rollouts, the marked prompt, and the bridge artifacts.

    The acceptance receipt is only meaningful if it can be read while no rollout
    for the session exists, so both observations must come from the same
    instant: a single exec, one captured_at stamp. The container shares the host
    clock, which is why payload timestamps are directly comparable to the
    verifier's send time.
    """
    selected = f"broker-{int(broker_pid)}*" if isinstance(broker_pid, int) and broker_pid > 0 else "broker-*"
    raw = dx(
        "d=/home/tester/.pi/agent/sessions; "
        "find $d -type f -name '*.jsonl' -printf '%p\\t%B@\\t%T@\\t%s\\n' 2>/dev/null; "
        f"echo '{SNAPSHOT_MARKED}'; "
        f"grep -rlF -- {shlex.quote(mark)} $d 2>/dev/null; true; "
        f"echo '{SNAPSHOT_BRIDGE}'; "
        f"b={shlex.quote(BRIDGE_DIR)}; "
        f"find \"$b\" -maxdepth 1 -type f -name {selected} -printf '%f\\t%B@\\t%T@\\t%s\\n' 2>/dev/null; true; "
        f"echo '{SNAPSHOT_BODIES}'; "
        f"for f in \"$b\"/{selected}; do [ -f \"$f\" ] || continue; "
        "printf '@@ %s\\n' \"$(basename \"$f\")\"; cat \"$f\" 2>/dev/null; printf '\\n'; done 2>/dev/null; true",
        timeout=30,
    )
    captured_at = time.time()
    before_marked, _, rest = raw.partition(SNAPSHOT_MARKED)
    marked_part, _, rest = rest.partition(SNAPSHOT_BRIDGE)
    bridge_files_part, _, bodies_part = rest.partition(SNAPSHOT_BODIES)

    files: dict[str, dict] = {}
    for line in before_marked.splitlines():
        cols = line.split("\t")
        if len(cols) == 4 and cols[0].startswith("/"):
            birth = _stat_number(cols[1])
            mtime = _stat_number(cols[2])
            files[cols[0]] = {"birth": birth if birth > 0.0 else mtime, "mtime": mtime, "size": int(cols[3] or 0)}

    bridge: dict[str, dict] = {}
    for line in bridge_files_part.splitlines():
        cols = line.split("\t")
        if len(cols) == 4 and cols[0].startswith("broker-"):
            birth = _stat_number(cols[1])
            mtime = _stat_number(cols[2])
            bridge[cols[0]] = {"birth": birth if birth > 0.0 else mtime, "mtime": mtime, "size": int(cols[3] or 0), "json": None}
    for chunk in ("\n" + bodies_part).split("\n@@ ")[1:]:
        name, _, payload = chunk.partition("\n")
        name = name.strip()
        if not name:
            continue
        entry = bridge.setdefault(name, {"birth": 0.0, "mtime": 0.0, "size": 0, "json": None})
        try:
            parsed = json.loads(payload.strip() or "null")
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            entry["json"] = parsed

    return {
        "files": files,
        "marked": [line for line in marked_part.splitlines() if line.startswith("/")],
        "bridge": bridge,
        "captured_at": captured_at,
    }


NO_WORKING_INDICATOR_EXTENSION = r"""
// Test-only control extension: Pi renders no working indicator, so the broker's
// PTY "Working..." heuristic cannot supply acceptance evidence. Production code
// is untouched; the broker's receipt path is the only acceptance evidence left.
export default function (pi) {
	const hide = (_event, ctx) => {
		try {
			ctx.ui.setWorkingVisible(false);
		} catch {}
		try {
			ctx.ui.setWorkingMessage("busy");
		} catch {}
	};
	pi.on("session_start", hide);
	pi.on("session_switch", hide);
}
"""


def prepare_fresh_send() -> None:
    """One-time container setup shared by the fresh-session scenarios."""
    if http("GET", "/api/me")[0] != 200:
        status, login = http("POST", "/api/login", {"password": args.password})
        if status != 200:
            fail(f"fresh scenarios: API login failed: {status} {login}")
    # A delayed OpenAI-compatible provider inside the container: the first
    # assistant row (and therefore the rollout file) arrives seconds after the
    # prompt was accepted.
    dx("cat > /home/tester/slow_provider.py <<'PYEOF'\n" + SLOW_PROVIDER_SOURCE + "\nPYEOF")
    dx(f"setsid nohup python3 /home/tester/slow_provider.py {SLOW_PROVIDER_PORT} {SLOW_PROVIDER_DELAY_SECONDS} /home/tester/slow_provider.log "
       ">/home/tester/slow_provider.out 2>&1 < /dev/null &")
    probe = wait_for(lambda: '"ok"' in dx(f"python3 -c \"import urllib.request;print(urllib.request.urlopen('http://127.0.0.1:{SLOW_PROVIDER_PORT}/v1/health',timeout=2).read())\" 2>&1"), 15, 0.5, "slow provider up")
    if not probe:
        fail("slow provider did not start: " + dx("cat /home/tester/slow_provider.out"))
    # The legacy tool file that seeds Pi's startup deprecation gate is global to
    # ~/.pi/agent; every fresh-session scenario needs a gate-free TUI.
    dx("rm -f /home/tester/.pi/agent/tools/legacy-custom-tool.js")
    report["evidence"]["fresh_scenarios_setup"] = {
        "slow_provider_port": SLOW_PROVIDER_PORT,
        "slow_provider_delay_seconds": SLOW_PROVIDER_DELAY_SECONDS,
        "busy_quiet_seconds": BUSY_QUIET_SECONDS,
    }


def fresh_send_scenario(*, key: str, cwd: str, prompt_kind: str, suppress_working: bool) -> None:
    """First prompt of a FRESH Pi session whose first assistant reply takes > 3 s.

    Desired behaviour (asserted): the backend's own acceptance receipt names
    this prompt and is dated inside the quiet window while Pi has written no
    rollout at all; the broker never publishes an `undelivered` verdict; the
    composer keeps no restored draft; and the prompt is a user row in the Pi
    jsonl. Timings are reported either way.

    The receipt is the point: for a fresh session the rollout appears seconds
    after acceptance, so a verifier that only checks "no verdict + the prompt
    eventually lands" cannot tell acceptance evidence from luck. Requiring a
    prompt-correlated receipt timestamped before the first rollout (and before
    the quiet-window deadline) is the log-independent observation the fix
    depends on.

    ``suppress_working`` installs the test-only extension that stops Pi rendering
    its working indicator, which removes the broker's PTY fallback: any verdict
    suppression then has to come from the receipt. ``prompt_kind="whitespace"``
    submits a prompt with leading/trailing spaces and a trailing newline so the
    receipt digest is exercised against Pi's own input normalization.
    """
    slug = key.split("_", 1)[0]
    result: dict = {"pass": False}
    report["scenarios"][key] = result
    evidence: dict = {
        "slow_provider_delay_seconds": SLOW_PROVIDER_DELAY_SECONDS,
        "prompt_kind": prompt_kind,
        "suppress_working_indicator": suppress_working,
        "cwd": cwd,
    }
    report["evidence"][key] = evidence
    if slug == "5":
        report["evidence"]["fresh_session_first_send"] = evidence

    dx(f"mkdir -p {shlex.quote(cwd)}")
    extension_path = None
    if suppress_working:
        dx("mkdir -p /home/tester/.pi/agent/extensions")
        dx(f"cat > /home/tester/.pi/agent/extensions/udv-no-working-indicator.ts <<'TSEOF'\n{NO_WORKING_INDICATOR_EXTENSION.strip()}\nTSEOF")
        extension_path = "/home/tester/.pi/agent/extensions/udv-no-working-indicator.ts"
    evidence["working_indicator_suppression_extension"] = extension_path

    proj2 = cwd
    known = {r.get("session_id") for r in (http("GET", "/api/sessions")[1] or {}).get("sessions", [])}
    status, launch = http("POST", "/api/sessions", {
        "cwd": proj2, "agent_backend": "pi", "model_provider": "slow", "model": "slow-1", "create_in_tmux": True,
    })
    evidence["launch_response"] = {"status": status, "response": launch}
    if status != 200:
        fail(f"{key}: session launch failed: {status} {launch}")

    def fresh_row():
        status, data = http("GET", "/api/sessions")
        if status != 200:
            return None
        return next((r for r in data.get("sessions", []) if r.get("cwd") == proj2 and r.get("session_id") not in known), None)

    row = wait_for(fresh_row, 90, 0.25, f"{key} session listed")
    if not row:
        fail(f"{key}: session never appeared")
    sid5 = row["session_id"]
    target5 = f"{row.get('tmux_session')}:{row.get('tmux_window')}"
    evidence.update(session_id=sid5, tmux_target=target5)

    # Wait until Pi's TUI is up and no gate is present.
    def tui_ready():
        text = pane(target5)
        return text if text.strip() and "Press any key to continue" not in text else None
    wait_for(tui_ready, 40, 0.5, f"{key} TUI ready")
    time.sleep(4)
    pane_before = pane(target5)
    (ART / f"scenario{slug}-pane-before-send.txt").write_text(pane_before)
    evidence["gate_absent"] = "Press any key to continue" not in pane_before
    if not evidence["gate_absent"]:
        fail(f"{key}: deprecation gate still present; not a clean fresh-session test")
    row_before = session_row(sid5)
    dump_json(f"scenario{slug}-session-row-before-send.json", row_before)
    evidence["row_before_send"] = {k: row_before.get(k) for k in ("log_path", "thread_id", "pid", "broker_pid", "busy", "send_verdict", "agent_backend")} if row_before else None

    prompt_core = f"UDV-fresh-{prompt_kind}-{int(time.time() * 1000)}"
    mark5 = f"  {prompt_core}  " if prompt_kind == "whitespace" else prompt_core
    draft_text = mark5 + "\n" if prompt_kind == "whitespace" else mark5
    evidence["prompt"] = mark5
    evidence["submitted_text"] = draft_text
    evidence["prompt_core"] = prompt_core
    broker_pid = (row_before or {}).get("broker_pid")
    evidence["broker_pid"] = broker_pid
    baseline = container_snapshot(prompt_core, broker_pid)
    evidence["jsonl_files_before_send"] = baseline["files"]
    evidence["bridge_artifacts_before_send"] = {
        name: {"birth": item["birth"], "mtime": item["mtime"], "size": item["size"], "body": item["json"]}
        for name, item in baseline["bridge"].items()
    }
    baseline_marker = ((baseline["bridge"].get(f"broker-{broker_pid}.json") or {}).get("json") or {})
    if isinstance(baseline_marker.get("sessionFile"), str):
        # Pi allocates the session file path at session creation but writes no
        # bytes until its first assistant row; the marker therefore names a file
        # that does not exist yet, which is exactly the window under test.
        evidence["planned_session_file_before_send"] = baseline_marker["sessionFile"]
        evidence["planned_session_file_existed_before_send"] = baseline_marker["sessionFile"] in baseline["files"]

    # Real browser: select the fresh session, type, click Send.
    ab("open", BASE + "/")
    if wait_for(lambda: ab_eval('Boolean(document.querySelector("#pw"))') or ab_eval('Boolean(document.querySelector("#sessions"))'), 20, 0.25, f"{key} browser landing"):
        if ab_eval('Boolean(document.querySelector("#pw"))'):
            ab("fill", "#pw", args.password)
            ab("click", "#loginBtn")
    selector = f'#sessions .session[data-session-id="{sid5}"]'
    if not wait_for(lambda: ab_eval(f'Boolean(document.querySelector({json.dumps(selector)}))'), 30, 0.25, f"{key} session card"):
        fail(f"{key}: browser did not render the fresh session card")
    ab("click", selector)
    if not wait_for(lambda: ab_eval('!document.querySelector("#msg").disabled'), 25, 0.25, f"{key} composer enabled"):
        fail(f"{key}: composer did not become enabled")
    ab("click", "#msg")
    ab("press", "Control+a")
    ab("press", "Delete")
    if not wait_for(lambda: ab_eval('document.querySelector("#msg").value') == "", 5, 0.1, f"{key} composer cleared"):
        fail(f"{key}: could not clear composer")
    ab("type", "#msg", mark5)
    if prompt_kind == "whitespace":
        # Trailing newline through the documented newline key, so the committed
        # text carries outer whitespace exactly as a user paste would.
        ab("press", "Shift+Enter")
    typed_value = ab_eval('document.querySelector("#msg").value')
    evidence["composer_value_before_send"] = typed_value
    if typed_value != draft_text:
        fail(f"{key}: composer holds {typed_value!r}, expected {draft_text!r}")

    send_started = time.time()
    ab("click", "#sendBtn")
    evidence["send_started_at"] = send_started

    timeline = []
    pane_samples: list[dict] = []
    next_pane_probe = 0.0
    first = {"jsonl_file_seen": None, "verdict_seen": None, "user_row_seen": None, "composer_restored": None, "toast_seen": None,
             "slow_request_seen": None, "slow_response_sent": None, "receipt_seen": None}
    verdict_payload = None
    send_id = None
    composer = None
    deadline = send_started + 25
    while time.time() < deadline:
        rel = lambda: round(time.time() - send_started, 3)
        current = session_row(sid5)
        verdict = current.get("send_verdict") if current else None
        t_row = rel()
        snap = container_snapshot(prompt_core, broker_pid)
        t_fs = rel()
        browser_state = ab_eval('(() => { const m = document.querySelector("#msg"); const t = document.querySelector("#toast"); return { composer: m ? m.value : null, toast: t ? t.textContent : null }; })()')
        t_br = rel()
        composer = browser_state.get("composer")
        new_files = [p for p in snap["files"] if p not in baseline["files"]]
        marker_json = (snap["bridge"].get(f"broker-{broker_pid}.json") or {}).get("json") or {}
        planned = marker_json.get("sessionFile") if isinstance(marker_json.get("sessionFile"), str) else None
        receipts = _receipt_candidates(snap["bridge"], draft_text, send_started)
        if t_fs >= next_pane_probe:
            pane_text = pane(target5)
            pane_samples.append({
                "t": t_fs,
                "has_working": "Working" in pane_text,
                "has_interrupt_hint": "esc to interrupt" in pane_text,
                "has_compacting_hint": "compacting context" in pane_text or "compacting conversation" in pane_text,
                "busy": current.get("busy") if current else None,
            })
            next_pane_probe = t_fs + 1.0
        timeline.append({"t_row": t_row, "t_fs": t_fs, "t_browser": t_br, "log_path": current.get("log_path") if current else None,
                         "busy": current.get("busy") if current else None, "send_verdict": verdict, "new_jsonl_files": new_files,
                         "marked_files": snap["marked"], "composer": composer, "toast": browser_state.get("toast"),
                         "receipts": [{"artifact": item["artifact"], "match": item["match"], "ts": item["ts"],
                                       "ts_source": item["ts_source"]} for item in receipts]})
        if receipts and first["receipt_seen"] is None:
            best = max(receipts, key=lambda item: item["ts"])
            first["receipt_seen"] = {
                **best,
                "sample_t": t_fs,
                "send_relative_ts": round(best["ts"] - send_started, 3),
                "log_path_at_seen": current.get("log_path") if current else None,
                "busy_at_seen": current.get("busy") if current else None,
                "pi_pid_at_seen": current.get("pid") if current else None,
                "send_verdict_at_seen": verdict,
                "planned_session_file_at_seen": planned,
                "planned_session_file_existed_at_seen": bool(planned and planned in snap["files"]),
                "new_jsonl_files_at_seen": new_files,
            }
        if new_files and first["jsonl_file_seen"] is None:
            first["jsonl_file_seen"] = {
                "sample_t": t_fs,
                "files": {p: snap["files"][p] for p in new_files},
                "earliest_birth_ts": min(snap["files"][p]["birth"] for p in new_files),
            }
        if verdict and first["verdict_seen"] is None:
            first["verdict_seen"] = {"sample_t": t_row, "verdict_ts_relative": round(float(verdict["ts"]) - send_started, 3) if isinstance(verdict.get("ts"), (int, float)) else None}
            verdict_payload = verdict
        if snap["marked"] and first["user_row_seen"] is None:
            first["user_row_seen"] = {"sample_t": t_fs, "files": snap["marked"]}
        if composer == draft_text and first["composer_restored"] is None:
            first["composer_restored"] = {"sample_t": t_br}
        toast = browser_state.get("toast")
        if isinstance(toast, str) and "did not reach the backend" in toast and first["toast_seen"] is None:
            first["toast_seen"] = {"sample_t": t_br, "text": toast}
        time.sleep(0.1)
    dump_json(f"scenario{slug}-timeline.json", timeline)
    dump_json(f"scenario{slug}-pane-samples.json", pane_samples)
    ab("screenshot", "--full", str(ART / f"scenario{slug}-browser-after-20s.png"))

    slow_log = [json.loads(l) for l in dx("cat /home/tester/slow_provider.log 2>/dev/null").splitlines() if l.startswith("{")]
    for entry in slow_log:
        entry["relative_to_send"] = round(entry["ts"] - send_started, 3)
        if entry["ts"] < send_started - 1.0:
            continue
        if entry["event"] == "request" and first["slow_request_seen"] is None:
            first["slow_request_seen"] = entry["relative_to_send"]
        if entry["event"] == "response_sent" and first["slow_response_sent"] is None:
            first["slow_response_sent"] = entry["relative_to_send"]

    journal = journal_rows(sid5)
    submit = next((e for e in journal if e.get("type") == "submit" and prompt_core in str(e.get("display_text") or "")), None)
    send_id = submit.get("id") if submit else None
    final_row = session_row(sid5)
    dump_json(f"scenario{slug}-session-row-final.json", final_row)
    final_snap = container_snapshot(prompt_core, broker_pid)
    user_row = pi_user_row(prompt_core)
    pane_after = pane(target5)
    (ART / f"scenario{slug}-pane-after.txt").write_text(pane_after)
    log_grep = dx("grep -rniE 'send_verdict|undelivered' /home/tester/server.log /home/tester/.local/share/codoxear --include='*.log' --include='server.log' 2>/dev/null | grep -v '/send_journal/' | head -50")
    tmux_log_grep = dx(f"tmux capture-pane -p -t {shlex.quote(target5)} -S -2000 | grep -niE 'send_verdict|undelivered' | head -20")
    final_composer = ab_eval('document.querySelector("#msg").value')

    # The browser's client send_id (verdict.send_id, "s1-...") differs from the
    # server journal id, so correlate by session: this session is fresh, had no
    # verdict before the send, and receives exactly one send.
    verdict_for_send = bool(verdict_payload) and not (row_before or {}).get("send_verdict")
    session_dir_marker = proj2.replace("/", "-")
    jsonl_for_session_at_send = [p for p in baseline["files"] if f"-{session_dir_marker.strip('-')}--" in p]
    evidence.update(
        send_id=send_id, journal_rows=journal, first_events=first, verdict_payload=verdict_payload,
        final_composer_value=final_composer, jsonl_files_after=final_snap["files"], jsonl_files_containing_prompt=final_snap["marked"],
        bridge_artifacts_after=final_snap["bridge"],
        pi_user_row=user_row, slow_provider_log=slow_log, server_and_broker_log_matches=log_grep, pane_log_matches=tmux_log_grep,
        pane_activity_samples=pane_samples, pane_after=pane_after[-2500:],
    )
    receipt = first["receipt_seen"]
    receipt_ts = receipt.get("ts") if receipt else None
    rollout_birth = (first["jsonl_file_seen"] or {}).get("earliest_birth_ts")
    deadline_ts = send_started + BUSY_QUIET_SECONDS
    # Identity: the receipt's artifact must name the rollout that later appears,
    # so the receipt cannot be a leftover from another session's prompt.
    rollout_path = None
    if first["jsonl_file_seen"]:
        rollout_path = next(iter(first["jsonl_file_seen"]["files"]))
    elif final_row is not None and final_row.get("log_path"):
        rollout_path = final_row["log_path"]
    final_marker = ((final_snap["bridge"].get(f"broker-{broker_pid}.json") or {}).get("json") or {})
    receipt_session_file = (
        final_marker.get("sessionFile")
        or (receipt or {}).get("artifact_session_file")
        or (receipt or {}).get("planned_session_file_at_seen")
    )
    committed_text = submit.get("text") if isinstance(submit, dict) and isinstance(submit.get("text"), str) else None
    committed_digest = hashlib.sha256(committed_text.encode("utf-8")).hexdigest() if committed_text is not None else None
    normalized_text = committed_text.strip(_JS_TRIM_CHARS) if committed_text is not None else None
    normalized_digest = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest() if normalized_text else None
    receipt_digest = (receipt or {}).get("artifact_digest")
    receipt_asserts = {
        "acceptance_receipt_observed": bool(receipt),
        "receipt_names_this_prompt": bool(receipt and receipt.get("match")),
        # The receipt must be dated inside the quiet window: at the moment the
        # latch would settle, acceptance evidence already existed.
        "receipt_ts_inside_quiet_window": bool(
            receipt_ts is not None and 0.0 <= float(receipt_ts) - send_started < BUSY_QUIET_SECONDS
        ),
        # ...and it must precede the rollout, i.e. it is evidence that existed
        # while Pi had written no session file at all.
        "receipt_precedes_first_rollout": bool(
            receipt_ts is not None and rollout_birth is not None and float(receipt_ts) < float(rollout_birth)
        ),
        "quiet_window_deadline_precedes_first_rollout": bool(
            rollout_birth is not None and deadline_ts < float(rollout_birth)
        ),
        "receipt_observed_while_no_log_bound": bool(
            receipt
            and receipt.get("log_path_at_seen") is None
            and receipt.get("planned_session_file_existed_at_seen") is False
        ),
        "receipt_artifact_names_session_rollout": bool(
            receipt_session_file and rollout_path and Path(str(receipt_session_file)).name == Path(str(rollout_path)).name
        ),
        # The receipt must come from the session's own Pi process, which is what
        # makes it evidence about this backend rather than a stray file.
        "receipt_pid_is_backend_process": bool(
            receipt and receipt.get("artifact_pid") is not None and receipt.get("artifact_pid") == receipt.get("pi_pid_at_seen")
        ),
        # The latch must still be open at the deadline (no rollout, no log). The
        # receipt is what holds it, so `busy` stays true across the whole
        # rollout-free stretch instead of publishing a verdict at the deadline.
        "latch_held_through_quiet_window": bool(
            rollout_birth is not None
            and [
                sample for sample in timeline
                if BUSY_QUIET_SECONDS <= sample["t_row"] < float(rollout_birth) - send_started
            ]
            and all(
                sample.get("busy") is True
                for sample in timeline
                if BUSY_QUIET_SECONDS <= sample["t_row"] < float(rollout_birth) - send_started
            )
        ),
    }
    if prompt_kind == "whitespace":
        # The receipt is only usable when its digest names the text the broker
        # armed the latch with. Pi's TUI normalizes submitted input, so a prompt
        # carrying outer whitespace is where that correlation can silently break.
        receipt_asserts["prompt_carried_outer_whitespace"] = bool(committed_text and committed_text != normalized_text)
        receipt_asserts["receipt_digest_matches_normalized_text"] = bool(
            normalized_digest and receipt_digest and normalized_digest == receipt_digest
        )
    control_asserts: dict = {}
    if suppress_working:
        # The control is only worth anything if the PTY really carried no
        # activity evidence: no working indicator text and no other broker hint.
        in_window = [s for s in pane_samples if rollout_birth is None or s["t"] < float(rollout_birth) - send_started]
        control_asserts = {
            "working_indicator_suppressed": bool(pane_samples) and not any(s["has_working"] for s in pane_samples),
            "no_pty_activity_hint_before_rollout": bool(in_window) and not any(
                s["has_working"] or s["has_interrupt_hint"] or s["has_compacting_hint"] for s in in_window
            ),
        }
    summary = {
        "send_started_at": send_started,
        "prompt": draft_text,
        "row_log_path_at_send_time": row_before.get("log_path") if row_before else None,
        "pi_jsonl_for_this_session_existed_at_send_time": bool(jsonl_for_session_at_send),
        "jsonl_files_before_send_count": len(baseline["files"]),
        "new_jsonl_file_first_seen_s": (first["jsonl_file_seen"] or {}).get("sample_t"),
        "first_rollout_birth_relative_s": round(float(rollout_birth) - send_started, 3) if rollout_birth else None,
        "quiet_window_deadline_s": round(deadline_ts - send_started, 3),
        "acceptance_receipt_first_seen_s": (receipt or {}).get("sample_t"),
        "acceptance_receipt_ts_relative_s": (receipt or {}).get("send_relative_ts"),
        "acceptance_receipt_ts_source": (receipt or {}).get("ts_source"),
        "acceptance_receipt_artifact": (receipt or {}).get("artifact"),
        "acceptance_receipt_match": (receipt or {}).get("match"),
        "receipt_only_window_seconds": (
            round(float(rollout_birth) - float(receipt["ts"]), 3) if receipt and rollout_birth else None
        ),
        "committed_text": committed_text,
        "committed_text_digest": committed_digest,
        "committed_text_normalized": normalized_text,
        "committed_text_normalized_digest": normalized_digest,
        "receipt_digest": receipt_digest,
        "verdict_first_seen_in_row_s": (first["verdict_seen"] or {}).get("sample_t"),
        "verdict_ts_relative_s": (first["verdict_seen"] or {}).get("verdict_ts_relative"),
        "user_row_first_seen_s": (first["user_row_seen"] or {}).get("sample_t"),
        "slow_provider_request_s": first["slow_request_seen"],
        "slow_provider_response_s": first["slow_response_sent"],
        "composer_restored_at_s": (first["composer_restored"] or {}).get("sample_t"),
        "final_composer_value": final_composer,
        "verdict_payload": verdict_payload,
    }
    asserts = {
        "no_verdict_published_for_send": not verdict_for_send,
        "composer_not_restored": final_composer != draft_text and first["composer_restored"] is None,
        "user_row_in_pi_jsonl": bool(user_row),
        "slow_enough_to_exercise_latch": bool(rollout_birth is not None and deadline_ts < float(rollout_birth)),
    }
    if REQUIRE_ACCEPTANCE_RECEIPT:
        asserts.update(receipt_asserts)
    asserts.update(control_asserts)
    result.update(
        timings=summary,
        asserts=asserts,
        receipt_asserts=receipt_asserts,
        control_asserts=control_asserts,
        receipt_assertions_required=REQUIRE_ACCEPTANCE_RECEIPT,
    )
    result["pass"] = all(asserts.values())


FRESH_SCENARIOS = (
    # The product path: Pi renders its working indicator, the rollout is late.
    dict(key="5_slow_first_reply_of_fresh_session_is_not_undelivered", cwd="/home/tester/proj2", prompt_kind="plain", suppress_working=False),
    # Decisive control: no working indicator at all, so only the receipt can
    # keep the latch from settling inside the rollout-free window.
    dict(key="6_working_indicator_suppressed_receipt_alone_prevents_verdict", cwd="/home/tester/proj3", prompt_kind="plain", suppress_working=True),
    # Same control with the prompt Pi has to normalize (outer whitespace plus a
    # trailing newline), which is where a raw-text digest stops matching.
    dict(key="7_whitespace_prompt_still_matches_receipt_digest", cwd="/home/tester/proj4", prompt_kind="whitespace", suppress_working=True),
)


def run_fresh_scenarios() -> None:
    try:
        prepare_fresh_send()
    except Exception as error:
        report["fatal_error_fresh_setup"] = str(error)
        log(f"FAIL fresh setup: {error}")
        return
    for spec in FRESH_SCENARIOS:
        try:
            fresh_send_scenario(**spec)
        except Exception as error:
            report[f"fatal_error_{spec['key']}"] = str(error)
            log(f"FAIL {spec['key']}: {error}")


try:
    main()
except Exception as error:
    report["fatal_error"] = str(error)
    log(f"FAIL: {error}")
try:
    run_fresh_scenarios()
finally:
    write_report()
    atexit.unregister(write_report)
    try:
        ab("close")
    except Exception:
        pass

flags = {name: value.get("pass", False) for name, value in report["scenarios"].items()}
print(json.dumps(flags, indent=2))
fatal = [name for name in report if name.startswith("fatal_error")]
sys.exit(0 if all(flags.values()) and not fatal else 1)
