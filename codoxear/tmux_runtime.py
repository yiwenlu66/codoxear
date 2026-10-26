from __future__ import annotations

import re
import subprocess
import threading
import time
from typing import Any, Callable


def _clean_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


class TmuxCommandError(RuntimeError):
    """A tmux invocation that codoxear depends on for ordering failed."""

    def __init__(self, command: str, detail: str) -> None:
        super().__init__(f"tmux {command} failed: {detail}")
        self.command = command
        self.detail = detail


# Window name goes last so it can contain tabs without shifting the fields the
# order mirror parses; the index is what defines the displayed order. The
# `@codoxear_nonce` user option is the mirror's rename-proof identity for a
# window (its launch name stops matching as soon as the mirror titles it).
TMUX_WINDOW_FORMAT = "#{window_index}\t#{window_id}\t#{window_active}\t#{@codoxear_nonce}\t#{window_name}"

_NONCE_RE = re.compile(r"^[0-9a-f]{6,64}$")


def parse_window_nonce(raw: str) -> str | None:
    """A valid stamp value, or None.

    A tmux that cannot expand user options returns the literal format string, so
    only hex-shaped values are accepted: anything else means "no stamp", and an
    un-stamped window is never renamed (its name is the only way back to it).
    """
    value = raw.strip()
    return value if _NONCE_RE.match(value) else None


def parse_window_rows(text: str | None) -> list[dict[str, Any]]:
    """Parse `list-windows -F TMUX_WINDOW_FORMAT` output into window records.

    Malformed lines are dropped rather than guessed at: a window codoxear
    cannot read is a window the order mirror must not move.
    """
    rows: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t", 4)
        if len(parts) != 5:
            continue
        index_raw, window_id, active_raw, nonce_raw, name = parts
        try:
            index = int(index_raw)
        except ValueError:
            continue
        window_id = window_id.strip()
        if not window_id:
            continue
        rows.append(
            {
                "index": index,
                "id": window_id,
                "active": active_raw.strip() == "1",
                "nonce": parse_window_nonce(nonce_raw),
                "name": name,
            }
        )
    rows.sort(key=lambda row: row["index"])
    return rows


def tmux_list_windows(
    tmux_bin: str,
    *,
    tmux_session_name: str,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[dict[str, Any]]:
    """List one tmux session's windows, ordered by index."""
    proc = run(
        [tmux_bin, "list-windows", "-t", tmux_session_name, "-F", TMUX_WINDOW_FORMAT],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or f"exit status {proc.returncode}").strip()
        raise TmuxCommandError("list-windows", detail)
    return parse_window_rows(proc.stdout)


def tmux_swap_windows(
    tmux_bin: str,
    *,
    src_window_id: str,
    dst_window_id: str,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Swap two windows' positions by id.

    No `-d` flag: its meaning for `swap-window` changed between tmux releases.
    Measured, `swap-window -d` keeps the viewer on the same *window* when one of
    the swapped windows is current on 3.8, but on 3.5a it leaves the viewer on
    the index (showing another window's pane), same as no flag at all — and on
    3.8 `-d` on two unrelated windows moves the viewer to the swapped source.
    Since no flag is correct on every version, the mirror never passes one and
    instead restores the viewer's window explicitly after the pass
    (`tmux_select_window`, driven by `tmux_order.reconcile_windows`).
    """
    argv = [tmux_bin, "swap-window", "-s", src_window_id, "-t", dst_window_id]
    proc = run(argv, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or f"exit status {proc.returncode}").strip()
        raise TmuxCommandError("swap-window", detail)


def tmux_select_window(
    tmux_bin: str,
    *,
    window_id: str,
    tmux_session_name: str,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Make `window_id` the session's current window again.

    Swapping the current window's position leaves the session pointing at
    whatever window landed in its old index; this puts the viewer back on the
    pane it was reading.
    """
    proc = run(
        [tmux_bin, "select-window", "-t", f"{tmux_session_name}:{window_id}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or f"exit status {proc.returncode}").strip()
        raise TmuxCommandError("select-window", detail)


def tmux_available(
    *,
    ttl_seconds: float,
    cache_lock: threading.Lock,
    get_cache: Callable[[], tuple[float, bool] | None],
    set_cache: Callable[[tuple[float, bool]], None],
    which: Callable[[str], str | None],
    now: Callable[[], float] = time.time,
) -> bool:
    now_ts = now()
    ttl = max(0.0, float(ttl_seconds))
    with cache_lock:
        cache = get_cache()
        if cache is not None:
            cached_at, cached = cache
            if ttl > 0 and (now_ts - cached_at) < ttl:
                return bool(cached)
    available = which("tmux") is not None
    with cache_lock:
        set_cache((now_ts, bool(available)))
    return bool(available)


def tmux_pane_snapshot(
    tmux_bin: str,
    *,
    tmux_session_name: str,
    pane_id: str | None = None,
    window: str | None = None,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    target = _clean_optional_text(pane_id)
    if target is None and _clean_optional_text(window) is not None:
        target = f"{tmux_session_name}:{window}"
    if target is None:
        return {}
    fmt = "#{pane_id}\t#{pane_pid}\t#{pane_dead}\t#{pane_dead_status}\t#{pane_current_command}\t#{window_name}"
    proc = run(
        [tmux_bin, "display-message", "-p", "-t", target, fmt],
        capture_output=True,
        text=True,
        check=False,
    )
    out: dict[str, Any] = {"tmux_target": target}
    if proc.returncode != 0:
        out["tmux_inspect_error"] = (proc.stderr or proc.stdout or f"exit status {proc.returncode}").strip()
        return out
    parts = (proc.stdout or "").strip().split("\t")
    keys = ("tmux_pane_id", "tmux_pane_pid", "tmux_pane_dead", "tmux_pane_dead_status", "tmux_pane_command", "tmux_window")
    for key, value in zip(keys, parts):
        out[key] = value
    cap = run(
        [tmux_bin, "capture-pane", "-p", "-t", target, "-S", "-80"],
        capture_output=True,
        text=True,
        check=False,
    )
    if cap.returncode == 0:
        out["tmux_pane_tail"] = (cap.stdout or "")[-4000:]
    else:
        out["tmux_capture_error"] = (cap.stderr or cap.stdout or f"exit status {cap.returncode}").strip()
    return out


def tmux_stamp_window_nonce(
    tmux_bin: str,
    *,
    window_id: str,
    nonce: str,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Record the launch nonce on a window so a rename cannot lose its identity."""
    proc = run(
        [tmux_bin, "set-option", "-w", "-t", window_id, "@codoxear_nonce", nonce],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or f"exit status {proc.returncode}").strip()
        raise TmuxCommandError("set-option", detail)


def tmux_rename_window(
    tmux_bin: str,
    *,
    window_id: str,
    name: str,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Title a window with the name the sidebar shows."""
    proc = run(
        [tmux_bin, "rename-window", "-t", window_id, name],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or f"exit status {proc.returncode}").strip()
        raise TmuxCommandError("rename-window", detail)


def tmux_renumber_windows(
    tmux_bin: str,
    *,
    tmux_session_name: str,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    """Close index gaps: `move-window -r` renumbers the session from base-index."""
    proc = run(
        [tmux_bin, "move-window", "-r", "-t", tmux_session_name],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or f"exit status {proc.returncode}").strip()
        raise TmuxCommandError("move-window", detail)


def tmux_base_index(
    tmux_bin: str,
    *,
    tmux_session_name: str,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> int | None:
    """The session's first window index, falling back to the global option.

    Read-only and best effort: an unreadable/older tmux returns None, and the
    mirror then skips renumbering rather than guessing a base.
    """
    for argv in (
        [tmux_bin, "show-options", "-v", "-t", tmux_session_name, "base-index"],
        [tmux_bin, "show-options", "-g", "-v", "base-index"],
    ):
        proc = run(argv, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            continue
        raw = (proc.stdout or "").strip()
        try:
            return int(raw)
        except ValueError:
            continue
    return None
