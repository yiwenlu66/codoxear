"""Container-only behavioral driver. No credentials or external model traffic."""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.cookiejar
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import threading
import time
import urllib.request
from urllib.parse import quote

args = argparse.ArgumentParser()
args.add_argument("--port", type=int, required=True)
PORT = args.parse_args().port
HOME = Path.home()
PROOF = HOME / "proof"
PROOF.mkdir(exist_ok=True)
report = {}


def save(name, obj):
    (PROOF / name).write_text(json.dumps(obj, indent=2))


def until(fn, timeout=45):
    end = time.monotonic() + timeout
    last = None
    while time.monotonic() < end:
        try:
            last = fn()
            if last:
                return last
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(0.15)
    raise AssertionError(f"Timed out: {fn}; last={last!r}")


class Provider(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        last = str([m for m in payload["messages"] if m["role"] == "user"][-1]["content"])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            self.wfile.write(b'data: {"id":"test","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n')
            self.wfile.flush()
            if "SLOW" in last:
                time.sleep(25)
            for data in [
                {"id": "test", "choices": [{"index": 0, "delta": {"content": "LOCAL RESPONSE: " + last}}]},
                {"id": "test", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40}},
            ]:
                self.wfile.write(("data: " + json.dumps(data) + "\n\n").encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def tmux(name, command):
    subprocess.run(["tmux", "new-session", "-d", "-s", name, "-x", "100", "-y", "30", command], check=True)


jar = http.cookiejar.CookieJar()
client = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def api(path, data=None):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}" + path,
                                 data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with client.open(req, timeout=8) as response:
        return json.load(response)


def logs(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def has_message(path, role, text):
    return any(row.get("message", {}).get("role") == role and text in json.dumps(row["message"]) for row in logs(path))


try:
    config = HOME / ".pi/agent"
    config.mkdir(parents=True, exist_ok=True)
    (HOME / "external-work").mkdir(exist_ok=True)
    (HOME / "child-work").mkdir(exist_ok=True)
    (HOME / "broker-work").mkdir(exist_ok=True)
    (config / "models.json").write_text(json.dumps({"providers": {"local": {
        "baseUrl": "http://127.0.0.1:19722/v1", "apiKey": "fake-key", "api": "openai-completions",
        "models": [{"id": "local-test", "reasoning": False, "contextWindow": 100000, "maxTokens": 4096, "input": ["text"]}],
    }}}))
    (config / "settings.json").write_text(json.dumps({"defaultProvider": "local", "defaultModel": "local-test", "retry": {"enabled": False}, "quietStartup": True, "defaultProjectTrust": "always"}))
    provider = ThreadingHTTPServer(("127.0.0.1", 19722), Provider)
    threading.Thread(target=provider.serve_forever, daemon=True).start()
    subprocess.run(["codoxear-pi-extension", "install"], check=True)
    installed = json.loads(subprocess.check_output(["codoxear-pi-extension", "status", "--json"]))
    assert installed["status"] == "ok", installed
    report["installer"] = installed
    pi = "pi --approve --no-skills --no-prompt-templates --no-context-files"
    tmux("external", f"cd {HOME}/external-work; exec {pi}")
    tmux("child", f"cd {HOME}/child-work; export PI_SUBAGENT_CHILD=1; exec {pi}")
    server_log = open(PROOF / "server.log", "w")
    server = subprocess.Popen(["python3", "-m", "codoxear.server"], stdout=server_log, stderr=subprocess.STDOUT)
    until(lambda: api("/api/login", {"password": "external-test-password"}))
    listing = until(lambda: (v := api("/api/sessions")) and len(v["external"]) == 1 and v)
    external = listing["external"][0]
    sid = external["session_id"]
    log = external["log_path"]
    pid = int(sid.split("-")[2])
    assert listing["sessions"] == []
    tail = api(f"/api/sessions/{sid}/messages/tail")
    assert tail["transcript_state"] == "bound", tail
    assert not any(c["name"] in {"model", "new", "compact", "resume"} for c in external["slash_commands"])
    report["initial"] = listing
    assert api(f"/api/sessions/{sid}/send", {"text": "API real user message"})["ok"]
    until(lambda: has_message(log, "assistant", "LOCAL RESPONSE"))
    assert has_message(log, "user", "API real user message")
    assert not any(row.get("message", {}).get("customType") for row in logs(log))
    tail = api(f"/api/sessions/{sid}/messages/tail")
    assert any(e.get("role") == "user" and "API real user" in e.get("text", "") for e in tail["events"]), tail
    report["real_user_and_turn"] = tail
    (HOME / "external-work/proof.txt").write_text("external cwd file\n")
    assert api(f"/api/sessions/{sid}/file/read?path=proof.txt")["text"] == "external cwd file\n"
    assert api(f"/api/sessions/{sid}/search?q=API%20real%20user")["matches"]
    anchor = tail["events"][-1]["history_cursor"]
    assert api(f"/api/sessions/{sid}/messages/neighbor?direction=previous&cursor={quote(anchor)}")["neighbor"]
    assert sid not in json.dumps(api("/api/sessions/snapshot"))
    assert api(f"/api/sessions/{sid}/diagnostics")["external"] is True
    report["file_search_neighbor_and_snapshot_isolation"] = True
    with client.open(f"http://127.0.0.1:{PORT}/api/sessions/{sid}/live?cursor={quote(tail['live_cursor'])}", timeout=10) as live:
        assert live.headers["Content-Type"] == "text/event-stream"
        api(f"/api/sessions/{sid}/send", {"text": "SSE real time message"})
        received = []
        for _ in range(100):
            line = live.readline().decode()
            if line.startswith("data:"):
                event = json.loads(line[5:])
                received.extend(event.get("events", []))
                if any("SSE real time" in event.get("text", "") for event in received):
                    break
        assert any(event.get("role") == "user" and "SSE real time" in event.get("text", "") for event in received)
        report["sse_live_delta"] = received
    until(lambda: not api("/api/sessions")["external"][0]["busy"])
    api(f"/api/sessions/{sid}/send", {"text": "SLOW interrupt target"})
    until(lambda: api("/api/sessions")["external"][0]["busy"])
    steering = api(f"/api/sessions/{sid}/send", {"text": "STEER busy message"})
    assert steering["delivery"] == "steer", steering
    api(f"/api/sessions/{sid}/interrupt", {})
    until(lambda: not api("/api/sessions")["external"][0]["busy"])
    until(lambda: any(row.get("message", {}).get("stopReason") == "aborted" for row in logs(log)))
    report["retained_context_abort_and_idle"] = True
    # Pi abort restores unconsumed steering to the terminal editor. Verify
    # actual busy steering in a separate turn allowed to consume the user row.
    api(f"/api/sessions/{sid}/send", {"text": "SLOW steering target"})
    until(lambda: api("/api/sessions")["external"][0]["busy"])
    api(f"/api/sessions/{sid}/send", {"text": "STEER consumed real user"})
    until(lambda: has_message(log, "user", "STEER consumed real user"), timeout=60)
    until(lambda: not api("/api/sessions")["external"][0]["busy"])
    report["busy_steer_user_row"] = True
    # Native /new through the terminal updates the same process identity.
    subprocess.run(["tmux", "send-keys", "-t", "external", "C-u", "/new", "Enter"], check=True)
    switched = until(lambda: (v := api("/api/sessions")["external"][0]) and v["thread_id"] != external["thread_id"] and v)
    assert switched["session_id"] == sid
    report["new_rebind"] = switched
    # Broker path + globally installed loader: one marker/socket, no external row.
    tmux("broker", f"cd {HOME}/broker-work; export CODEX_WEB_AGENT_BACKEND=pi; exec codoxear-broker -- {pi.removeprefix('pi ')}")
    broker_listing = until(lambda: (v := api("/api/sessions")) and len(v["sessions"]) == 1 and v)
    assert len(broker_listing["external"]) == 1, broker_listing
    broker = broker_listing["sessions"][0]
    caps = until(lambda: list((HOME / ".local/share/codoxear/pi-active-sessions").glob("*.caps")))
    marker = Path(str(caps[0]).removesuffix(".caps"))
    until(lambda: marker.exists())  # Pi session_start, not just broker sidecar creation.
    api(f"/api/sessions/{broker['session_id']}/send", {"text": "BROKER still works"})
    broker = until(lambda: (row := api("/api/sessions")["sessions"][0]) and row.get("log_path") and row)
    until(lambda: has_message(broker["log_path"], "user", "BROKER still works"))
    assert len(caps) == 1
    assert json.loads(caps[0].read_text())["bridgeVersion"] == 4
    report["broker_and_child_exclusion"] = broker_listing
    assert len(list((HOME / ".local/share/codoxear/external").glob("pi-*.json"))) == 1
    # A nested Pi with inherited broker marker env must use its own external id.
    parent = json.loads(marker.read_text())
    tmux("nested", f"cd {HOME}/child-work; export CODEX_WEB_PI_ACTIVE_SESSION_FILE={shlex.quote(str(marker))} CODEX_WEB_PI_BRIDGE_PID={parent['pid']}; exec {pi}")
    nested_listing = until(lambda: (v := api("/api/sessions")) and len(v["external"]) == 2 and v)
    assert json.loads(marker.read_text())["pid"] == parent["pid"]
    nested = next(row for row in nested_listing["external"] if row["session_id"] != sid)
    api(f"/api/sessions/{nested['session_id']}/delete", {})
    os.kill(int(nested["session_id"].split("-")[2]), 0)  # Dismiss did not terminate it.
    report["nested_inheritance_and_dismiss"] = True
    # Expose a live transcript to the browser phase.
    api(f"/api/sessions/{sid}/send", {"text": "UI ready after new"})
    until(lambda: has_message(switched["log_path"], "assistant", "UI ready after new"))
    (PROOF / "ready").touch()
    until(lambda: (PROOF / "finish").exists(), timeout=180)
    report["browser_user_row"] = has_message(switched["log_path"], "user", "BROWSER real user message")
    # Crash path leaves stale files; pid+start validation must remove the row.
    os.kill(pid, signal.SIGKILL)
    until(lambda: not api("/api/sessions")["external"])
    report["crash_removed"] = True
    save("report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    server.terminate()
    server.wait(timeout=10)
except Exception:
    for session in ("external", "child", "broker", "nested"):
        proc = subprocess.run(["tmux", "capture-pane", "-pt", session], capture_output=True, text=True)
        (PROOF / f"{session}-terminal.txt").write_text(proc.stdout + proc.stderr)
    raise
