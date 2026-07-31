from __future__ import annotations

import os
import signal
import time
from typing import Callable


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The PID exists but is owned by another user.
        return True
    except Exception:
        return False
    return True


def process_group_alive(root_pid: int) -> bool:
    if root_pid <= 0:
        return False
    try:
        os.killpg(root_pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False
    return True


def terminate_process_group(
    root_pid: int,
    *,
    process_group_alive: Callable[[int], bool],
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    wait_seconds: float = 1.0,
) -> bool:
    if not process_group_alive(root_pid):
        return True
    try:
        os.killpg(root_pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    deadline = now() + max(wait_seconds, 0.0)
    while process_group_alive(root_pid):
        if now() >= deadline:
            break
        sleep(0.05)
    if not process_group_alive(root_pid):
        return True
    try:
        os.killpg(root_pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    deadline = now() + 0.2
    while process_group_alive(root_pid):
        if now() >= deadline:
            break
        sleep(0.05)
    return not process_group_alive(root_pid)


def terminate_process(
    pid: int,
    *,
    pid_alive: Callable[[int], bool],
    now: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    wait_seconds: float = 1.0,
) -> bool:
    if not pid_alive(pid):
        return True
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    deadline = now() + max(wait_seconds, 0.0)
    while pid_alive(pid):
        if now() >= deadline:
            break
        sleep(0.05)
    if not pid_alive(pid):
        return True
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    deadline = now() + 0.2
    while pid_alive(pid):
        if now() >= deadline:
            break
        sleep(0.05)
    return not pid_alive(pid)
