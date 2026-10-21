#!/usr/bin/env python3
"""Drive scripts/docker_undelivered_send_verify.sh through HTTP and a real browser."""
import argparse
import atexit
import http.cookiejar
import json
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
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
T0 = time.time()
jar = http.cookiejar.CookieJar()
opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
report: dict = {
    "commit": args.commit,
    "base_url": BASE,
    "started_at": T0,
    "scenarios": {
        "1_gate_swallows_browser_send": {"pass": False},
        "2_server_publishes_undelivered_verdict": {"pass": False},
        "3_browser_restores_composer_and_toasts": {"pass": False},
        "4_second_send_delivers_without_new_verdict": {"pass": False},
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


try:
    main()
except Exception as error:
    report["fatal_error"] = str(error)
    log(f"FAIL: {error}")
finally:
    write_report()
    atexit.unregister(write_report)
    try:
        ab("close")
    except Exception:
        pass

flags = {name: value.get("pass", False) for name, value in report["scenarios"].items()}
print(json.dumps(flags, indent=2))
sys.exit(0 if all(flags.values()) and "fatal_error" not in report else 1)
