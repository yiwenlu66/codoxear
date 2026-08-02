from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


CODEX_LIVE_COMMANDS = (
    {"name": "model", "description": "Select model for this conversation"},
    {"name": "effort", "description": "Set reasoning effort for this conversation"},
)
_PROBE_THREAD_ID = "00000000-0000-0000-0000-000000000000"


def codex_live_control_compatible_args(args: list[str]) -> bool:
    """Reject TUI-only provider layers that a sibling app-server can't reproduce."""
    return not any(token in {"--remote", "--oss", "--local-provider", "--profile", "-p"} for token in args)


def codex_app_server_config_args(args: list[str]) -> list[str]:
    """Copy only config-layer CLI flags that the sibling app-server accepts."""
    out: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in {"-c", "--config", "--enable", "--disable"} and index + 1 < len(args):
            out.extend([token, args[index + 1]])
            index += 2
            continue
        if token == "--strict-config":
            out.append(token)
        index += 1
    return out


class CodexLiveControlError(RuntimeError):
    pass


@dataclass
class CodexAppServerProcess:
    process: subprocess.Popen[bytes]
    socket_path: Path
    stderr_file: Any


def _connect_unix_websocket(socket_path: Path, *, timeout_s: float) -> Any:
    from websockets.sync.client import unix_connect

    return unix_connect(
        path=str(socket_path),
        uri="ws://localhost/",
        open_timeout=timeout_s,
        close_timeout=min(timeout_s, 0.5),
        compression=None,
    )


def _recv_response(websocket: Any, request_id: int, *, timeout_s: float) -> dict[str, Any]:
    deadline = time.monotonic() + max(timeout_s, 0.0)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Codex app-server request {request_id} timed out")
        raw = websocket.recv(timeout=remaining)
        if not isinstance(raw, str):
            continue
        message = json.loads(raw)
        if not isinstance(message, dict) or message.get("id") != request_id:
            continue
        return message


def _initialized_websocket(
    socket_path: Path,
    *,
    timeout_s: float,
    connect: Callable[..., Any] = _connect_unix_websocket,
) -> Any:
    websocket = connect(socket_path, timeout_s=timeout_s)
    websocket.send(
        json.dumps(
            {
                "method": "initialize",
                "id": 1,
                "params": {
                    "clientInfo": {
                        "name": "codoxear",
                        "title": "Codoxear",
                        "version": "0.1.0",
                    },
                    "capabilities": {"experimentalApi": True},
                },
            }
        )
    )
    response = _recv_response(websocket, 1, timeout_s=timeout_s)
    if isinstance(response.get("error"), dict):
        websocket.close()
        raise CodexLiveControlError(f"Codex app-server initialize failed: {response['error'].get('message') or response['error']}")
    websocket.send(json.dumps({"method": "initialized", "params": {}}))
    return websocket


def codex_live_settings_supported(
    socket_path: Path,
    *,
    timeout_s: float = 1.0,
    connect: Callable[..., Any] = _connect_unix_websocket,
) -> bool:
    websocket = _initialized_websocket(socket_path, timeout_s=timeout_s, connect=connect)
    try:
        websocket.send(
            json.dumps(
                {
                    "method": "thread/settings/update",
                    "id": 2,
                    "params": {"threadId": _PROBE_THREAD_ID, "model": "codoxear-capability-probe"},
                }
            )
        )
        response = _recv_response(websocket, 2, timeout_s=timeout_s)
    finally:
        websocket.close()
    error = response.get("error")
    if not isinstance(error, dict):
        return True
    if error.get("code") == -32601:
        return False
    message = str(error.get("message") or "").lower()
    return "method not found" not in message and "requires experimentalapi capability" not in message


def update_codex_thread_settings(
    socket_path: Path,
    *,
    thread_id: str,
    model: str | None = None,
    effort: str | None = None,
    timeout_s: float = 2.0,
    connect: Callable[..., Any] = _connect_unix_websocket,
) -> dict[str, Any]:
    thread_id = str(thread_id or "").strip()
    model = str(model).strip() if model is not None else None
    effort = str(effort).strip().lower() if effort is not None else None
    if not thread_id:
        raise ValueError("Codex thread is not bound yet")
    if bool(model) == bool(effort):
        raise ValueError("exactly one of model or effort is required")

    params: dict[str, Any] = {"threadId": thread_id}
    if model:
        params["model"] = model
    if effort:
        params["effort"] = effort
    websocket = _initialized_websocket(socket_path, timeout_s=timeout_s, connect=connect)
    try:
        websocket.send(json.dumps({"method": "thread/settings/update", "id": 2, "params": params}))
        response = _recv_response(websocket, 2, timeout_s=timeout_s)
    finally:
        websocket.close()
    error = response.get("error")
    if isinstance(error, dict):
        raise CodexLiveControlError(str(error.get("message") or error))
    if response.get("result") != {}:
        raise CodexLiveControlError("Codex app-server returned an invalid settings acknowledgement")
    return {"ok": True, "model": model, "effort": effort}


def _stderr_text(stderr_file: Any) -> str:
    try:
        stderr_file.flush()
        stderr_file.seek(0)
        raw = stderr_file.read()
    except Exception:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")[-4000:].strip()
    return str(raw or "")[-4000:].strip()


def stop_codex_app_server(server: CodexAppServerProcess | None, *, wait_seconds: float = 1.0) -> None:
    if server is None:
        return
    process = server.process
    if process.poll() is None:
        try:
            os.killpg(int(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=max(wait_seconds, 0.0))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(int(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
    try:
        server.stderr_file.close()
    except Exception:
        pass
    try:
        server.socket_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def start_codex_app_server(
    *,
    agent_bin: str,
    cwd: str,
    codex_home: Path,
    socket_path: Path,
    shell_argv_for_command: Callable[[str], list[str]],
    config_args: list[str] | None = None,
    timeout_s: float = 5.0,
    environ: dict[str, str] | None = None,
    preexec_fn: Callable[[], None] | None = None,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
    probe: Callable[..., bool] = codex_live_settings_supported,
) -> tuple[CodexAppServerProcess | None, str | None]:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass
    stderr_file = tempfile.TemporaryFile(mode="w+b")
    env = dict(os.environ if environ is None else environ)
    env["CODEX_HOME"] = str(codex_home)
    command = shlex.join([agent_bin, *(config_args or []), "app-server", "--listen", f"unix://{socket_path}"])
    try:
        process = popen(
            shell_argv_for_command(f"exec {command}"),
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=stderr_file,
            start_new_session=True,
            preexec_fn=preexec_fn,
        )
    except Exception as exc:
        stderr_file.close()
        return None, f"failed to start Codex app-server: {exc}"

    server = CodexAppServerProcess(process=process, socket_path=socket_path, stderr_file=stderr_file)
    deadline = time.monotonic() + max(timeout_s, 0.0)
    last_probe_error: BaseException | None = None
    while time.monotonic() <= deadline:
        returncode = process.poll()
        if returncode is not None:
            detail = _stderr_text(stderr_file)
            stop_codex_app_server(server)
            suffix = f": {detail}" if detail else ""
            return None, f"Codex app-server exited with status {returncode}{suffix}"
        if socket_path.exists():
            try:
                if probe(socket_path, timeout_s=min(1.0, max(deadline - time.monotonic(), 0.1))):
                    return server, None
                stop_codex_app_server(server)
                return None, "Codex app-server does not support thread/settings/update"
            except Exception as exc:
                last_probe_error = exc
        time.sleep(0.05)

    detail = _stderr_text(stderr_file)
    stop_codex_app_server(server)
    if detail:
        return None, f"Codex app-server did not become ready: {detail}"
    if last_probe_error is not None:
        return None, f"Codex app-server did not become ready: {last_probe_error}"
    return None, "Codex app-server did not become ready before timeout"
