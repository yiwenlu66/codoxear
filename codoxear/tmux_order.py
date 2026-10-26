"""Sidebar → tmux window layout mirror.

Each codoxear session launched "in tmux" owns one window in a single tmux
session, and tmux's only window ordering is `window_index`. This module makes
the tmux window list a direct mirror of the sidebar:

* **position** — window `base + k` is sidebar row `k` (codoxear windows are kept
  contiguous at the head of the session, in sidebar order; windows no session
  names follow in their existing relative order);
* **name** — each window is titled with the name the sidebar shows for that
  session (alias, else cwd basename), so a status-bar entry and a sidebar row
  are the same string.

Both mirror properties are needed for the feature to be usable: ordering alone
leaves you matching `dexgem-litellm-gateway-85d667` to a sidebar row by
position, and names alone leave the two in different orders.

Identity is the one thing a rename can destroy, so every window carries a
launch-scoped token in the `@codoxear_nonce` tmux user option: the broker's
`spawn_nonce` when it has one, else a deterministic hash of the launch name. The
stamp is written before the first rename, which is what lets later passes find
the window after its title stops matching the sidecar.

Pure planning (`plan_next_swap`, `resolve_targets`) is separated from execution
(`reconcile_windows`, which receives a `TmuxOps` bundle) so both are directly
testable against a fake tmux.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .tmux_runtime import TmuxCommandError


@dataclass(frozen=True)
class WindowTarget:
    """One session row projected onto the tmux window it lives in."""

    session_name: str
    display_name: str
    launch_name: str
    session_id: str
    token: str


@dataclass(frozen=True)
class SwapStep:
    """One `swap-window` invocation, by window id."""

    src: str
    dst: str


@dataclass(frozen=True)
class TmuxOps:
    """The tmux side effects a reconcile pass may perform, by session name."""

    list_windows: Callable[[str], list[dict[str, Any]]]
    swap: Callable[[str, SwapStep], None]
    select: Callable[[str, str], None]
    rename: Callable[[str, str, str], None]
    stamp: Callable[[str, str, str], None]
    renumber: Callable[[str], None]
    base_index: Callable[[str], int | None]


def _clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def identity_token(launch_name: str, spawn_nonce: Any) -> str:
    """The rename-proof identity token for one window.

    The broker's launch nonce when the sidecar recorded one, otherwise a stable
    digest of the launch name — deterministic either way, so a later pass
    derives the same token from the same row after the window has been titled.
    """
    nonce = _clean_text(spawn_nonce)
    if nonce and len(nonce) <= 64 and all(ch in "0123456789abcdef" for ch in nonce):
        return nonce
    source = _clean_text(launch_name)
    return hashlib.sha1(source.encode("utf-8")).hexdigest()[:16] if source else ""


def window_targets(rows: Iterable[Mapping[str, Any]], *, default_session: str) -> list[WindowTarget]:
    """Project ordered listing rows onto tmux windows, preserving row order.

    Rows without a `tmux_window` have no window (direct transport, or a
    terminal-launched broker that exported no window name) and are dropped. A
    window named by two rows keeps the first one: the second cannot own the same
    window, and reordering on a guess is worse than leaving one window alone.
    """
    targets: list[WindowTarget] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        launch_name = _clean_text(row.get("tmux_window"))
        if not launch_name:
            continue
        session_name = _clean_text(row.get("tmux_session")) or default_session
        token = identity_token(launch_name, row.get("spawn_nonce"))
        if not token:
            continue
        key = (session_name, token)
        if key in seen:
            continue
        seen.add(key)
        session_id = row.get("session_id")
        targets.append(
            WindowTarget(
                session_name=session_name,
                display_name=_clean_text(row.get("display_name")),
                launch_name=launch_name,
                session_id=session_id if isinstance(session_id, str) else str(session_id or ""),
                token=token,
            )
        )
    return targets


def group_targets(targets: Sequence[WindowTarget]) -> dict[str, list[WindowTarget]]:
    """Group targets by tmux session, preserving first-appearance order."""
    grouped: dict[str, list[WindowTarget]] = {}
    for target in targets:
        grouped.setdefault(target.session_name, []).append(target)
    return grouped


def target_signature(targets: Sequence[WindowTarget]) -> tuple[tuple[str, str, str], ...]:
    """A comparable identity for "the layout we want", across generations.

    Includes the display name, so a rename alone (alias edit) counts as a change
    even when the order is untouched.
    """
    return tuple((target.session_name, target.token, target.display_name) for target in targets)


def resolve_targets(
    targets: Sequence[WindowTarget],
    windows: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[WindowTarget, str]], list[dict[str, str]]]:
    """Map targets to window ids: by identity token first, then by launch name.

    The name fallback is one-to-one over unclaimed windows, so a session created
    before the stamp existed still finds its own window, and two rows can never
    resolve to the same window.
    """
    by_token = {str(window.get("nonce")): str(window.get("id")) for window in windows if window.get("nonce")}
    claimed: set[str] = set()
    resolved: list[tuple[WindowTarget, str]] = []
    skipped: list[dict[str, str]] = []
    for target in targets:
        window_id = by_token.get(target.token)
        if window_id is not None and window_id not in claimed:
            claimed.add(window_id)
            resolved.append((target, window_id))
            continue
        candidates = [
            str(window.get("id"))
            for window in windows
            if window.get("name") == target.launch_name and str(window.get("id")) not in claimed
        ]
        if len(candidates) == 1:
            claimed.add(candidates[0])
            resolved.append((target, candidates[0]))
            continue
        skipped.append(
            {
                "window": target.launch_name,
                "session_id": target.session_id,
                "reason": "missing" if not candidates else "ambiguous",
            }
        )
    return resolved, skipped


def is_consecutive(indices: Sequence[int], base: int) -> bool:
    ordered = sorted(indices)
    return ordered == list(range(base, base + len(ordered)))


def plan_next_swap(*, target_ids: Sequence[str], windows: Sequence[Mapping[str, Any]]) -> SwapStep | None:
    """Return the next swap that moves `target_ids` into index order, or None.

    The target windows occupy whatever index slots they happen to hold; the
    swap sequence permutes them among those slots only, so the index set (and
    therefore contiguity) is preserved. Repeating this call with a fresh listing
    converges in at most `len(target_ids) - 1` swaps: each step parks one more
    window at its final slot.

    The session's current window is deliberately *not* consulted here: which
    window the viewer keeps is restored after the pass, not by the swap
    mechanics (see `reconcile_windows`).
    """
    ids = {window.get("id") for window in windows}
    present = [window_id for window_id in target_ids if window_id in ids]
    if len(present) < 2:
        return None
    id_by_index = {
        int(window["index"]): str(window["id"])
        for window in windows
        if isinstance(window.get("index"), int) and window.get("id") in set(present)
    }
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
    ops: TmuxOps,
) -> dict[str, Any]:
    """Apply one tmux session's names and order. Raises `TmuxCommandError` on tmux failures."""
    windows = ops.list_windows(session_name)
    resolved, skipped = resolve_targets(targets, windows)
    by_id = {str(window.get("id")): window for window in windows}

    # Identity before naming: a window that is renamed without a stamp could not
    # be found again, so stamping failures abort before any rename happens.
    stampable: set[str] = set()
    stamped = 0
    for target, window_id in resolved:
        current = by_id.get(window_id) or {}
        if current.get("nonce") == target.token:
            stampable.add(window_id)
            continue
        ops.stamp(session_name, window_id, target.token)
        stampable.add(window_id)
        stamped += 1

    renamed = 0
    for target, window_id in resolved:
        wanted = target.display_name
        current = by_id.get(window_id) or {}
        if not wanted or window_id not in stampable:
            continue
        if current.get("name") == wanted:
            continue
        # A name starting with "-" would be read by tmux as a flag.
        if wanted.startswith("-"):
            continue
        ops.rename(session_name, window_id, wanted)
        renamed += 1

    codoxear_ids = [window_id for _, window_id in resolved]

    windows = ops.list_windows(session_name) if (stamped or renamed) else windows
    owned = set(codoxear_ids)
    target_order = codoxear_ids + [str(window.get("id")) for window in windows if str(window.get("id")) not in owned]

    renumbered = False
    base = ops.base_index(session_name)
    if base is not None and not is_consecutive([int(window["index"]) for window in windows], base):
        ops.renumber(session_name)
        windows = ops.list_windows(session_name)
        renumbered = True

    viewer_window = next((str(window.get("id")) for window in windows if window.get("active")), None)

    swaps = 0
    steps: list[SwapStep] = []
    converged = True
    while True:
        step = plan_next_swap(target_ids=target_order, windows=windows)
        if step is None:
            break
        if swaps >= len(target_order):
            # Each swap parks at least one window, so this cannot happen unless
            # tmux stopped honouring swaps; never spin on it.
            converged = False
            break
        ops.swap(session_name, step)
        steps.append(step)
        swaps += 1
        windows = ops.list_windows(session_name)

    restored = False
    if swaps and viewer_window is not None:
        present = {str(window.get("id")) for window in windows}
        current = next((str(window.get("id")) for window in windows if window.get("active")), None)
        if viewer_window in present and current != viewer_window:
            ops.select(session_name, viewer_window)
            windows = ops.list_windows(session_name)
            restored = True

    return {
        "session": session_name,
        "swaps": swaps,
        "steps": steps,
        "renamed": renamed,
        "stamped": stamped,
        "renumbered": renumbered,
        "restored": restored,
        "skipped": skipped,
        "converged": converged,
        "order": [window.get("name") for window in windows],
    }


def reconcile(*, targets: Sequence[WindowTarget], ops: TmuxOps) -> list[dict[str, Any]]:
    """Reconcile every tmux session named by `targets`, isolating per-session failures."""
    summaries: list[dict[str, Any]] = []
    for session_name, session_targets in group_targets(targets).items():
        try:
            summaries.append(reconcile_windows(session_name=session_name, targets=session_targets, ops=ops))
        except TmuxCommandError as exc:
            summaries.append(
                {
                    "session": session_name,
                    "error": str(exc),
                    "swaps": 0,
                    "renamed": 0,
                    "stamped": 0,
                    "renumbered": False,
                    "restored": False,
                    "skipped": [],
                    "converged": False,
                    "order": [],
                }
            )
    return summaries
