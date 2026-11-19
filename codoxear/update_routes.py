"""Update check / apply / settings routes.

``check`` is a read of the deployed checkout's position relative to its remote;
``settings`` persists the auto-update preference; ``apply`` starts the operator's
update command as a runner detached from this request (see
``_spawn_update_process``), because the command normally deploys and restarts
this very server: the HTTP response must be already delivered, and the outcome
is reported through ``update_status.json`` rather than through the response.
"""
from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import update_check


JsonResponse = Callable[[Any, int, dict[str, Any]], None]
ReadBody = Callable[[Any], bytes]
ReadJsonBody = Callable[..., dict[str, Any]]

CHECK_PATH = "/api/update/check"
SETTINGS_PATH = "/api/update/settings"
APPLY_PATH = "/api/update/apply"

# A status left at "running" longer than this is treated as lost: the runner is
# reported only by itself, so a runner killed mid-update (a server restart is
# exactly the expected shape of a successful update) would otherwise block
# every later attempt forever.
UPDATE_STUCK_SECONDS = 300.0
UPDATE_TIMEOUT_SECONDS = 600.0
# The response is written after the runner starts; this delay keeps a restart
# triggered immediately by the update command from cutting it off mid-flight.
RESPONSE_FLUSH_SECONDS = 2.0
TIMEOUT_EXIT_CODE = -1
LAUNCH_FAILED_EXIT_CODE = -2


@dataclass(frozen=True)
class UpdateRouteDeps:
    require_auth: Callable[[Any], bool]
    json_response: JsonResponse
    read_body: ReadBody
    read_json_body: ReadJsonBody
    app_dir: Path
    working_dir: Path
    update_command: str


def handle_update_get_route(handler: Any, *, path: str, deps: UpdateRouteDeps) -> bool:
    if path != CHECK_PATH:
        return False
    if not deps.require_auth(handler):
        handler._unauthorized()
        return True
    result = update_check.check_for_updates(deps.working_dir)
    settings = update_check.read_update_settings(deps.app_dir)
    status = update_check.read_update_status(deps.app_dir)
    result["update_command_configured"] = bool(deps.update_command)
    result["auto_update"] = bool(settings.get("auto_update", False))
    if status:
        result["last_status"] = status
    deps.json_response(handler, 200, result)
    return True


def handle_update_post_route(handler: Any, *, path: str, deps: UpdateRouteDeps) -> bool:
    if path == SETTINGS_PATH:
        return _handle_settings_post(handler, deps=deps)
    if path == APPLY_PATH:
        return _handle_apply_post(handler, deps=deps)
    return False


def _handle_settings_post(handler: Any, *, deps: UpdateRouteDeps) -> bool:
    if not deps.require_auth(handler):
        handler._unauthorized()
        return True
    obj = deps.read_json_body(handler)
    settings = update_check.read_update_settings(deps.app_dir)
    if "auto_update" in obj:
        settings["auto_update"] = bool(obj["auto_update"])
    update_check.write_update_settings(deps.app_dir, settings)
    deps.json_response(handler, 200, {"ok": True, "auto_update": settings.get("auto_update", False)})
    return True


def _handle_apply_post(handler: Any, *, deps: UpdateRouteDeps) -> bool:
    if not deps.require_auth(handler):
        handler._unauthorized()
        return True
    # HTTP/1.1 keep-alive: an unconsumed body is parsed as the next request line
    # on this connection, so the body is consumed even though this route
    # takes no fields.
    deps.read_body(handler)
    if not deps.update_command:
        deps.json_response(handler, 400, {"error": "no update command configured"})
        return True
    status = update_check.read_update_status(deps.app_dir)
    if status.get("state") == "running" and time.time() - _status_started_ts(status) < UPDATE_STUCK_SECONDS:
        deps.json_response(handler, 409, {"error": "update already in progress"})
        return True

    target_commit = update_check.check_for_updates(deps.working_dir).get("latest_commit", "unknown")
    _spawn_update_process(
        update_command=deps.update_command,
        app_dir=deps.app_dir,
        target_commit=target_commit,
        working_dir=deps.working_dir,
    )
    deps.json_response(handler, 202, {"started": True, "target_commit": target_commit})
    return True


def _status_started_ts(status: dict[str, Any]) -> float:
    """A missing or non-numeric ``started_ts`` reads as the epoch, i.e. stale."""
    value = status.get("started_ts")
    return float(value) if isinstance(value, (int, float)) else 0.0


# Standalone runner: it is started with `python -c` and must survive both this
# request and the restart its own command triggers, so it carries no imports
# from this package (a checkout that changes underneath it must not matter).
# argv: update_command, cwd, log_path, status_path, target_commit,
#       flush_seconds, timeout_seconds, timeout_exit_code, launch_failed_exit_code
_CHILD_SCRIPT = '''"""Detached update runner started by POST /api/update/apply."""
import json
import os
import subprocess
import sys
import time

(
    update_command,
    cwd,
    log_path,
    status_path,
    target_commit,
    flush_seconds,
    timeout_seconds,
    timeout_exit_code,
    launch_failed_exit_code,
) = sys.argv[1:10]

time.sleep(float(flush_seconds))

exit_code = 0
with open(log_path, "w") as log:
    try:
        exit_code = subprocess.run(
            update_command,
            shell=True,
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=float(timeout_seconds),
        ).returncode
    except subprocess.TimeoutExpired:
        log.write("\\nUpdate command killed after %s seconds\\n" % timeout_seconds)
        exit_code = int(timeout_exit_code)
    except Exception as exc:
        log.write("\\nUpdate command failed to run: %r\\n" % (exc,))
        exit_code = int(launch_failed_exit_code)

status = {
    "state": "done" if exit_code == 0 else "failed",
    "exit_code": exit_code,
    "target_commit": target_commit,
    "finished_ts": time.time(),
}
tmp = status_path + ".tmp"
with open(tmp, "w") as handle:
    json.dump(status, handle, indent=2)
    handle.write("\\n")
os.replace(tmp, status_path)
'''


def _spawn_update_process(
    *,
    update_command: str,
    app_dir: Path,
    target_commit: str,
    working_dir: Path,
) -> None:
    """Record the attempt, then start the runner that will record its outcome."""
    update_check.write_update_status(app_dir, {
        "state": "running",
        "target_commit": target_commit,
        "started_ts": time.time(),
    })
    subprocess.Popen(
        [
            sys.executable,
            "-c",
            _CHILD_SCRIPT,
            update_command,
            str(working_dir),
            str(update_check.update_log_path(app_dir)),
            str(update_check.update_status_path(app_dir)),
            target_commit,
            str(RESPONSE_FLUSH_SECONDS),
            str(UPDATE_TIMEOUT_SECONDS),
            str(TIMEOUT_EXIT_CODE),
            str(LAUNCH_FAILED_EXIT_CODE),
        ],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
