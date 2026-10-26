"""Runtime driver that mirrors the sidebar order into tmux window order.

The pass is driven by *listing generations*, not by a timer: the browser renders
whatever `/api/sessions` last returned, so mirroring that same array is what
makes "tmux follows the sidebar" literally true. A timer would reorder against
stale priorities, because with nobody polling, `list_sessions` never runs and
`updated_ts` never advances.

`tick()` is the whole state machine and is synchronous, so the schedule
(settle window, floor between passes, no-op guard, relist interval) is testable
with an injected clock and no threads. `start()` only wraps it in the daemon
loop the server runs.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .tmux_order import TmuxOps, WindowTarget, reconcile, target_signature, window_targets
from .tmux_runtime import (
    tmux_base_index,
    tmux_list_windows,
    tmux_rename_window,
    tmux_renumber_windows,
    tmux_select_window,
    tmux_stamp_window_nonce,
    tmux_swap_windows,
)

# A burst of listing generations (an agent streaming, several clients polling)
# must settle into one pass instead of one pass per generation. The window is
# measured from the first generation that has not been applied yet, *not* from
# the newest arrival: a client polling faster than this window would otherwise
# restart it forever and starve the mirror.
SETTLE_SECONDS = 1.5

# Floor between acting passes. Exact mirroring means the top slot changes
# whenever any session records a message, so on a fleet of active sessions
# without a floor the status bar would permute continuously.
MIN_INTERVAL_SECONDS = 10.0

# Re-list even when the wanted order has not changed, so a window moved by hand
# in tmux is corrected without waiting for an unrelated ordering change.
RELIST_SECONDS = 60.0

TICK_SECONDS = 1.0


class TmuxOrderSync:
    """Coalescing sidebar-order → tmux-window-order mirror."""

    def __init__(
        self,
        *,
        default_session: str,
        tmux_available: Callable[[], bool],
        which_tmux: Callable[[str], str | None],
        run: Callable[..., Any],
        log: Callable[[str], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._default_session = default_session
        self._tmux_available = tmux_available
        self._which_tmux = which_tmux
        self._run = run
        self._log = log
        self._clock = clock
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Newest generation only: an older generation's order is superseded
        # before it is ever applied.
        self._generation = 0
        self._pending: tuple[int, list[WindowTarget]] | None = None
        self._unapplied_since: float | None = None
        # Worker-owned pass bookkeeping (read/written under the lock).
        self._applied_generation = 0
        self._applied_signature: tuple[tuple[str, str], ...] | None = None
        self._applied_ok = False
        self._last_pass_ts: float | None = None
        self._last_list_ts: float | None = None
        self._last_problem: str | None = None

    # -- producer side ---------------------------------------------------

    def note_generation(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Record the order of one `/api/sessions` response.

        Rows are projected immediately, on the calling (request) thread, so the
        worker never touches a listing row that the manager may still mutate.
        """
        targets = window_targets(rows, default_session=self._default_session)
        with self._lock:
            self._generation += 1
            self._pending = (self._generation, targets)
            if self._unapplied_since is None:
                self._unapplied_since = self._clock()

    # -- pass scheduling -------------------------------------------------

    def tick(self) -> list[dict[str, Any]] | None:
        """Run one scheduling decision. Returns the pass summary when one ran."""
        with self._lock:
            if self._pending is None or self._unapplied_since is None:
                return None
            generation, targets = self._pending
            if generation == self._applied_generation:
                self._unapplied_since = None
                return None
            now = self._clock()
            if now - self._unapplied_since < SETTLE_SECONDS:
                return None
            signature = target_signature(targets)
            if (
                signature == self._applied_signature
                and self._applied_ok
                and self._last_list_ts is not None
                and now - self._last_list_ts < RELIST_SECONDS
            ):
                # Same wanted order, still healthy, and re-listed recently: this
                # is the case a repeated identical poll lands in, and it must
                # cost no tmux subprocess at all.
                self._applied_generation = generation
                self._unapplied_since = None
                return None
            if self._last_pass_ts is not None and now - self._last_pass_ts < MIN_INTERVAL_SECONDS:
                return None
            self._last_pass_ts = now
        return self._run_pass(generation, targets, signature)

    def _run_pass(
        self,
        generation: int,
        targets: Sequence[WindowTarget],
        signature: tuple[tuple[str, str], ...],
    ) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        listed = False
        try:
            summaries, listed = self._apply(targets)
        except Exception as exc:  # the server must survive any tmux surprise
            self._log_once(f"tmux-order: pass failed: {type(exc).__name__}: {exc}")
        problem = next((s["error"] for s in summaries if isinstance(s.get("error"), str)), None)
        if problem:
            self._log_once(f"tmux-order: {problem}")
        unconverged = any(s.get("converged") is False for s in summaries if not s.get("error"))
        if unconverged:
            self._log_once("tmux-order: order did not converge; retrying on the next generation")
        if problem is None and not unconverged and not any(summary.get("skipped") for summary in summaries):
            with self._lock:
                # A fully clean pass clears the deduplication marker, so a
                # problem that recurs later is reported again.
                self._last_problem = None
        with self._lock:
            # Recording the generation on failure is deliberate: retries are
            # driven by the next generation (bounded by the floor), never by
            # spinning on the same one inside `tick`.
            self._applied_generation = generation
            # A generation that arrived while this pass was running must stay
            # marked unapplied, or it would wait for yet another generation
            # (which, with a client that has since stopped polling, never comes).
            if self._pending is None or self._pending[0] == generation:
                self._unapplied_since = None
            self._applied_signature = signature
            self._applied_ok = problem is None and not unconverged
            if listed:
                self._last_list_ts = self._clock()
        return summaries

    def _apply(self, targets: Sequence[WindowTarget]) -> tuple[list[dict[str, Any]], bool]:
        if not targets:
            return [], False
        if not self._tmux_available():
            return [], False
        tmux_bin = self._which_tmux("tmux")
        if tmux_bin is None:
            return [], False

        ops = TmuxOps(
            list_windows=lambda session_name: tmux_list_windows(tmux_bin, tmux_session_name=session_name, run=self._run),
            swap=lambda session_name, step: tmux_swap_windows(
                tmux_bin,
                src_window_id=step.src,
                dst_window_id=step.dst,
                run=self._run,
            ),
            select=lambda session_name, window_id: tmux_select_window(
                tmux_bin,
                window_id=window_id,
                tmux_session_name=session_name,
                run=self._run,
            ),
            rename=lambda session_name, window_id, name: tmux_rename_window(
                tmux_bin,
                window_id=window_id,
                name=name,
                run=self._run,
            ),
            stamp=lambda session_name, window_id, token: tmux_stamp_window_nonce(
                tmux_bin,
                window_id=window_id,
                nonce=token,
                run=self._run,
            ),
            renumber=lambda session_name: tmux_renumber_windows(
                tmux_bin,
                tmux_session_name=session_name,
                run=self._run,
            ),
            base_index=lambda session_name: tmux_base_index(
                tmux_bin,
                tmux_session_name=session_name,
                run=self._run,
            ),
        )

        summaries = reconcile(targets=targets, ops=ops)
        for summary in summaries:
            if summary.get("swaps") or summary.get("renamed") or summary.get("renumbered"):
                order = ",".join(str(name) for name in summary.get("order") or [])
                self._log(
                    f"tmux-order: session={summary['session']} swaps={summary['swaps']}"
                    f" renamed={summary['renamed']} renumbered={int(bool(summary['renumbered']))}"
                    f" restored={int(bool(summary['restored']))} order={order}"
                )
            skipped = summary.get("skipped") or []
            if skipped:
                detail = " ".join(f"{item['window']}({item['reason']})" for item in skipped)
                self._log_once(f"tmux-order: session={summary['session']} skipped={detail}")
        return summaries, True

    def _log_once(self, message: str) -> None:
        with self._lock:
            if message == self._last_problem:
                return
            self._last_problem = message
        self._log(message)

    # -- daemon loop -----------------------------------------------------

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            thread = threading.Thread(target=self._loop, name="tmux-order", daemon=True)
            self._thread = thread
        thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.tick()
            except Exception as exc:  # tick is total; this guards the thread itself
                self._log_once(f"tmux-order: tick failed: {type(exc).__name__}: {exc}")
            self._stop_event.wait(TICK_SECONDS)
