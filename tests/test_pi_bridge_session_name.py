from __future__ import annotations

import json
import socket
import stat
import subprocess
import tempfile
import textwrap
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "codoxear" / "pi_active_session_bridge.ts"


def _bundle(dest: Path) -> None:
    build = subprocess.run(
        [
            "npx",
            "esbuild",
            str(BRIDGE),
            "--bundle",
            "--platform=node",
            "--format=cjs",
            f"--outfile={dest}",
            "--log-level=error",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if build.returncode:
        raise AssertionError(build.stderr or build.stdout)


def _request(sock_path: Path, payload: dict) -> dict:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    try:
        sock.connect(str(sock_path))
        sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        buf = b""
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.split(b"\n", 1)[0].decode("utf-8"))
    finally:
        sock.close()


def test_bridge_exposes_session_name_control_over_its_socket() -> None:
    with tempfile.TemporaryDirectory() as td:
        temp_dir = Path(td)
        marker = temp_dir / "active.json"
        bundle = temp_dir / "bridge.cjs"
        load_state = temp_dir / "load.json"
        _bundle(bundle)

        exercise = temp_dir / "exercise.cjs"
        exercise.write_text(
            textwrap.dedent(
                """
                const fs = require("node:fs");
                const bridgeModule = require(process.argv[2]);
                const bridge = bridgeModule.default || bridgeModule;
                const marker = process.argv[3];
                const loadStatePath = process.argv[4];
                let sessionName = "";
                let callsAtLoad = 0;
                let requests = 0;
                const pi = {
                  on() {},
                  registerCommand() {},
                  getCommands() { return []; },
                  getThinkingLevel() { return "off"; },
                  setThinkingLevel() {},
                  setSessionName(name) {
                    requests += 1;
                    sessionName = name;
                  },
                  getSessionName() { return sessionName || undefined; },
                };
                bridge(pi);
                fs.writeFileSync(loadStatePath, JSON.stringify({ callsAtLoad }));
                process.stdout.write("READY\\n");
                process.stdin.resume();
                process.stdin.on("end", () => process.exit(0));
                process.stdin.on("data", () => {});
                """
            ),
            encoding="utf-8",
        )

        env = {
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(temp_dir),
            "CODEX_WEB_PI_ACTIVE_SESSION_FILE": str(marker),
        }
        proc = subprocess.Popen(
            ["node", str(exercise), str(bundle), str(marker), str(load_state)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        try:
            deadline = time.monotonic() + 30.0
            while True:
                line = proc.stdout.readline().decode("utf-8").strip()
                if line == "READY":
                    break
                if line or proc.poll() is not None:
                    raise AssertionError(f"bridge did not start: {line} {proc.stderr.read().decode('utf-8')}")
                if time.monotonic() > deadline:
                    raise AssertionError("bridge did not start in time")

            # Load-time work is passive: nothing calls a runtime action.
            assert json.loads(load_state.read_text(encoding="utf-8")) == {"callsAtLoad": 0}

            caps = json.loads(Path(f"{marker}.caps").read_text(encoding="utf-8"))
            assert caps["bridgeVersion"] == 3
            assert "session-name" in caps["features"]
            assert caps["pid"] == proc.pid

            sock_path = Path(f"{marker}.bridge.sock")
            assert sock_path.exists()
            # The mode is tightened in the listening callback, so allow the
            # event loop a moment rather than asserting on the bind default.
            mode_deadline = time.monotonic() + 5.0
            while stat.S_IMODE(sock_path.stat().st_mode) != 0o600:
                assert time.monotonic() < mode_deadline, "bridge socket mode was never tightened"
                time.sleep(0.01)

            assert _request(sock_path, {"op": "get_session_name"}) == {"ok": True, "pid": proc.pid, "name": ""}

            applied = _request(sock_path, {"op": "set_session_name", "name": "  Auth\nrefactor "})
            assert applied == {"ok": True, "pid": proc.pid, "name": "Auth refactor"}
            assert _request(sock_path, {"op": "get_session_name"})["name"] == "Auth refactor"

            cleared = _request(sock_path, {"op": "set_session_name", "name": ""})
            assert cleared == {"ok": True, "pid": proc.pid, "name": ""}

            unknown = _request(sock_path, {"op": "nope"})
            assert unknown["ok"] is False
            assert unknown["pid"] == proc.pid
        finally:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            for stream in (proc.stdout, proc.stderr):
                try:
                    stream.close()
                except Exception:
                    pass
