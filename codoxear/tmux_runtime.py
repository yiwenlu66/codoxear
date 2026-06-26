from __future__ import annotations

import subprocess
from typing import Any, Callable


def _clean_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    out = value.strip()
    return out or None


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
