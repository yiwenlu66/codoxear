"""Sidebar-order mirror for tmux windows.

Every codoxear session launched "in tmux" owns one window in a single tmux
session, and tmux's only window ordering is `window_index`. This module turns
the ordered listing rows into a window permutation and applies it with
`swap-window`, which permutes positions in place: no gaps, no renumbering, and
windows codoxear does not own keep their indices.

The ordering authority is the session-listing order (`session_listing.
sort_session_rows`) — the same array the browser renders — so this module never
re-derives a priority. It only decides *which* swaps realize the order it is
handed, and how to keep an attached viewer looking at the same window while the
numbering under it changes.

Pure planning (`plan_next_swap`) is separated from execution
(`reconcile_windows`) so both are directly testable with a fake tmux runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .tmux_runtime import TmuxCommandError


@dataclass(frozen=True)
class WindowTarget:
    """One session row projected onto the tmux window it lives in."""

    session_name: str
    window_name: str
    session_id: str


@dataclass(frozen=True)
class SwapStep:
    """One `swap-window` invocation, by window id."""

    src: str
    dst: str


def window_targets(rows: Iterable[Mapping[str, Any]], *, default_session: str) -> list[WindowTarget]:
    """Project ordered listing rows onto tmux windows, preserving row order.

    Rows without a `tmux_window` have no window (direct transport, or a
    terminal-launched broker that exported no window name) and are dropped. A
    window named by two rows keeps the first one, because the first is the one
    the sidebar ranks higher: the second row cannot own the same window without
    making the identity ambiguous, and reordering on a guess is worse than
    leaving one window where it is.
    """
    targets: list[WindowTarget] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        window_name = row.get("tmux_window")
        if not isinstance(window_name, str) or not window_name.strip():
            continue
        session_name = row.get("tmux_session")
        if not isinstance(session_name, str) or not session_name.strip():
            session_name = default_session
        key = (session_name, window_name)
        if key in seen:
            continue
        seen.add(key)
        session_id = row.get("session_id")
        targets.append(
            WindowTarget(
                session_name=session_name,
                window_name=window_name,
                session_id=session_id if isinstance(session_id, str) else str(session_id or ""),
            )
        )
    return targets


def group_targets(targets: Sequence[WindowTarget]) -> dict[str, list[WindowTarget]]:
    """Group targets by tmux session, preserving first-appearance order."""
    grouped: dict[str, list[WindowTarget]] = {}
    for target in targets:
        grouped.setdefault(target.session_name, []).append(target)
    return grouped


def target_signature(targets: Sequence[WindowTarget]) -> tuple[tuple[str, str], ...]:
    """A comparable identity for "the order we want", across generations."""
    return tuple((target.session_name, target.window_name) for target in targets)


def _window_id_by_index(windows: Sequence[Mapping[str, Any]], wanted_ids: set[str]) -> dict[int, str]:
    out: dict[int, str] = {}
    for window in windows:
        window_id = window.get("id")
        index = window.get("index")
        if isinstance(window_id, str) and isinstance(index, int) and window_id in wanted_ids:
            out[index] = window_id
    return out


def plan_next_swap(*, target_ids: Sequence[str], windows: Sequence[Mapping[str, Any]]) -> SwapStep | None:
    """Return the next swap that moves `target_ids` into index order, or None.

    The target windows occupy whatever index slots they happen to hold; the
    swap sequence permutes them among those slots only, so a window codoxear
    does not own never moves. Repeating this call with a fresh listing converges
    in at most `len(target_ids) - 1` swaps: each step parks one more window at
    its final slot.

    The session's current window is deliberately *not* consulted here: which
    window the viewer keeps is restored after the pass, not by the swap
    mechanics (see `reconcile_windows`).
    """
    ids = {window.get("id") for window in windows}
    present = [window_id for window_id in target_ids if window_id in ids]
    if len(present) < 2:
        return None
    id_by_index = _window_id_by_index(windows, set(present))
    if len(id_by_index) != len(present):
        # Two windows shared an index in the listing the caller passed in; refuse
        # to plan against a view tmux does not agree with.
        return None
    for index, wanted in zip(sorted(id_by_index), present):
        current = id_by_index[index]
        if current == wanted:
            continue
        return SwapStep(src=current, dst=wanted)
    return None


def reconcile_windows(
    *,
    session_name: str,
    targets: Sequence[WindowTarget],
    list_windows: Callable[[str], list[dict[str, Any]]],
    swap: Callable[[str, SwapStep], None],
    select: Callable[[str, str], None],
) -> dict[str, Any]:
    """Apply one tmux session's target order. Raises `TmuxCommandError` on tmux failures.

    `select(session_name, window_id)` restores the session's current window after
    the swaps: exchanging positions leaves the current-window pointer on the old
    index slot, which would show the attached terminal a different agent's pane.
    Restoring explicitly is the only version-independent way to do this — the
    `swap-window -d` flag means different things on tmux 3.5a and 3.8.
    """
    windows = list_windows(session_name)
    viewer_window = next((str(window.get("id")) for window in windows if window.get("active")), None)
    ids_by_name: dict[str, list[str]] = {}
    for window in windows:
        name = window.get("name")
        window_id = window.get("id")
        if isinstance(name, str) and isinstance(window_id, str):
            ids_by_name.setdefault(name, []).append(window_id)

    skipped: list[dict[str, str]] = []
    target_ids: list[str] = []
    for target in targets:
        matches = ids_by_name.get(target.window_name) or []
        if len(matches) != 1:
            skipped.append(
                {
                    "window": target.window_name,
                    "session_id": target.session_id,
                    "reason": "missing" if not matches else "ambiguous",
                }
            )
            continue
        if matches[0] in target_ids:
            skipped.append({"window": target.window_name, "session_id": target.session_id, "reason": "duplicate"})
            continue
        target_ids.append(matches[0])

    swaps = 0
    steps: list[SwapStep] = []
    converged = True
    while True:
        step = plan_next_swap(target_ids=target_ids, windows=windows)
        if step is None:
            break
        if swaps >= len(target_ids):
            # Each swap parks at least one window, so this cannot happen unless
            # tmux stopped honouring swaps; never spin on it.
            converged = False
            break
        swap(session_name, step)
        steps.append(step)
        swaps += 1
        windows = list_windows(session_name)

    restored = False
    if swaps and viewer_window is not None:
        present_ids = {str(window.get("id")) for window in windows}
        current = next((str(window.get("id")) for window in windows if window.get("active")), None)
        if viewer_window in present_ids and current != viewer_window:
            select(session_name, viewer_window)
            windows = list_windows(session_name)
            restored = True

    return {
        "session": session_name,
        "swaps": swaps,
        "steps": steps,
        "skipped": skipped,
        "converged": converged,
        "restored": restored,
        "order": [window.get("name") for window in windows],
    }


def reconcile(
    *,
    targets: Sequence[WindowTarget],
    list_windows: Callable[[str], list[dict[str, Any]]],
    swap: Callable[[str, SwapStep], None],
    select: Callable[[str, str], None],
) -> list[dict[str, Any]]:
    """Reconcile every tmux session named by `targets`, isolating per-session failures."""
    summaries: list[dict[str, Any]] = []
    for session_name, session_targets in group_targets(targets).items():
        try:
            summaries.append(
                reconcile_windows(
                    session_name=session_name,
                    targets=session_targets,
                    list_windows=list_windows,
                    swap=swap,
                    select=select,
                )
            )
        except TmuxCommandError as exc:
            summaries.append(
                {
                    "session": session_name,
                    "error": str(exc),
                    "swaps": 0,
                    "skipped": [],
                    "converged": False,
                    "restored": False,
                    "order": [],
                }
            )
    return summaries
