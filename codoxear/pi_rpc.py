from __future__ import annotations

import json
import os
import queue
import signal
import secrets
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


PI_BIN = os.environ.get("PI_BIN", "pi")


class PiRpcClient:
    def __init__(
        self,
        *,
        proc: Any | None = None,
        cwd: str | None = None,
        session_path: Path | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._stdin_lock = threading.Lock()
        self._pending: dict[str, queue.Queue[dict[str, Any]]] = {}
        self._events: list[dict[str, Any]] = []
        self._stderr_lock = threading.Lock()
        self._stderr_lines: list[str] = []
        self._stderr_bytes = 0
        self._stderr_max_bytes = 16 * 1024
        self._closed = False
        self._proc = proc if proc is not None else self._spawn(cwd=cwd, session_path=session_path)
        self._reader = threading.Thread(target=self._reader_loop, name="pi-rpc-reader", daemon=True)
        self._reader.start()
        stderr = getattr(self._proc, "stderr", None)
        self._stderr_reader = None
        if stderr is not None:
            self._stderr_reader = threading.Thread(target=self._stderr_loop, name="pi-rpc-stderr", daemon=True)
            self._stderr_reader.start()

    @property
    def pid(self) -> int:
        pid = getattr(self._proc, "pid", 0)
        return int(pid) if isinstance(pid, int) else 0

    def _spawn(self, *, cwd: str | None, session_path: Path | None) -> subprocess.Popen[str]:
        argv = [PI_BIN, "--mode", "rpc"]
        if session_path is not None:
            argv.extend(["--session", str(session_path)])
        return subprocess.Popen(
            argv,
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

    def _append_stderr_line(self, line: str) -> None:
        if not isinstance(line, str) or not line:
            return
        with self._stderr_lock:
            self._stderr_lines.append(line)
            self._stderr_bytes += len(line.encode("utf-8", errors="replace"))
            while self._stderr_lines and self._stderr_bytes > self._stderr_max_bytes:
                removed = self._stderr_lines.pop(0)
                self._stderr_bytes -= len(removed.encode("utf-8", errors="replace"))

    def _stderr_loop(self) -> None:
        stderr = getattr(self._proc, "stderr", None)
        if stderr is None:
            return
        while True:
            try:
                line = stderr.readline()
            except Exception:
                break
            if not line:
                break
            self._append_stderr_line(line)

    def _terminate_managed_process_group(self, *, wait_seconds: float = 1.0) -> bool:
        pid = self.pid
        if pid <= 0:
            return True
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        deadline = time.monotonic() + max(wait_seconds, 0.0)
        while time.monotonic() < deadline:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                return False
            time.sleep(0.05)
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            try:
                os.killpg(pid, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                return False
            time.sleep(0.05)
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            return False
        return False

    def _reader_loop(self) -> None:
        stdout = getattr(self._proc, "stdout", None)
        if stdout is None:
            return
        while True:
            try:
                line = stdout.readline()
            except Exception:
                break
            if not line:
                break
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "response" and isinstance(payload.get("id"), str):
                rid = payload["id"]
                with self._lock:
                    pending = self._pending.get(rid)
                if pending is not None:
                    pending.put(payload)
                    continue
            with self._lock:
                self._events.append(payload)

    def _write_jsonl(self, payload: dict[str, Any]) -> None:
        stdin = getattr(self._proc, "stdin", None)
        if stdin is None:
            raise RuntimeError("pi rpc stdin is unavailable")
        with self._stdin_lock:
            stdin.write(json.dumps(payload) + "\n")
            stdin.flush()

    def send_command(self, command_type: str, *, payload: dict[str, Any] | None = None, request_id: str | None = None, timeout_s: float = 5.0) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("pi rpc client is closed")
        if not isinstance(command_type, str) or not command_type:
            raise ValueError("command_type required")
        rid = request_id or f"pi-{secrets.token_hex(8)}"
        body = {
            "id": rid,
            "type": command_type,
        }
        if payload:
            body.update(payload)
        waitq: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self._lock:
            self._pending[rid] = waitq
        try:
            self._write_jsonl(body)
            try:
                resp = waitq.get(timeout=timeout_s)
            except queue.Empty as exc:
                raise TimeoutError(f"pi rpc {command_type} timed out") from exc
        finally:
            with self._lock:
                self._pending.pop(rid, None)
        success = resp.get("success")
        if isinstance(success, bool):
            ok = success
        else:
            ok = bool(resp.get("ok", False))
        if not ok:
            raise RuntimeError(str(resp.get("error") or f"pi rpc {command_type} failed"))
        result = resp.get("data")
        if not isinstance(result, dict):
            result = resp.get("result")
        if result is None:
            return {}
        if not isinstance(result, dict):
            raise RuntimeError(f"pi rpc {command_type} returned invalid result")
        return result

    def prompt(self, text: str) -> dict[str, Any]:
        return self.send_command("prompt", payload={"message": text})

    def abort(self, turn_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] | None = None
        if isinstance(turn_id, str) and turn_id:
            payload = {"turn_id": turn_id}
        return self.send_command("abort", payload=payload)

    def get_state(self) -> dict[str, Any]:
        return self.send_command("get_state")

    def send_ui_response(
        self,
        request_id: str,
        *,
        value: Any | None = None,
        confirmed: bool | None = None,
        cancelled: bool = False,
    ) -> None:
        if self._closed:
            raise RuntimeError("pi rpc client is closed")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id required")

        payload: dict[str, Any] = {
            "type": "extension_ui_response",
            "id": request_id,
        }
        if cancelled:
            payload["cancelled"] = True
        elif confirmed is not None:
            payload["confirmed"] = confirmed
        else:
            payload["value"] = value
        self._write_jsonl(payload)

    def drain_events(self) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
        return events

    def drain_stderr_lines(self) -> list[str]:
        with self._stderr_lock:
            lines = list(self._stderr_lines)
            self._stderr_lines.clear()
            self._stderr_bytes = 0
        return lines

    def stderr_tail(self) -> str:
        with self._stderr_lock:
            return "".join(self._stderr_lines)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        stdout = getattr(self._proc, "stdout", None)
        if stdout is not None:
            try:
                stdout.close()
            except Exception:
                pass
        stdin = getattr(self._proc, "stdin", None)
        if stdin is not None:
            try:
                stdin.close()
            except Exception:
                pass
        stderr = getattr(self._proc, "stderr", None)
        if stderr is not None:
            try:
                stderr.close()
            except Exception:
                pass
        if not self._terminate_managed_process_group(wait_seconds=1.0):
            try:
                self._proc.terminate()
            except Exception:
                pass
        wait = getattr(self._proc, "wait", None)
        if callable(wait):
            try:
                wait(timeout=1.0)
            except Exception:
                pass
