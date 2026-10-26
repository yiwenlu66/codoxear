"""Scheduling behaviour of the sidebar-order → tmux mirror.

The schedule is the part that decides whether the feature is usable: it must
coalesce bursts, never chase its own tail, cost nothing when nothing changed,
and still repair a window someone moved by hand. These tests drive `tick()`
with a fake clock and a fake tmux, so no thread and no real tmux is involved.
"""

from __future__ import annotations

import subprocess

from codoxear.tmux_order_sync import (
    MIN_INTERVAL_SECONDS,
    RELIST_SECONDS,
    SETTLE_SECONDS,
    TmuxOrderSync,
)


class Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeTmuxBin:
    """A tmux whose windows a swap mutates; records every invocation."""

    def __init__(self, order: list[str], *, active: str | None = None) -> None:
        self.windows = [
            {"index": index, "id": f"@{name}", "name": name, "active": name == active}
            for index, name in enumerate(order)
        ]
        self.calls: list[list[str]] = []
        self.fail_list = False

    def run(self, argv: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        if argv[1] == "list-windows":
            if self.fail_list:
                return subprocess.CompletedProcess(argv, 1, stdout="", stderr="no server running\n")
            text = "".join(
                f"{w['index']}\t{w['id']}\t{1 if w['active'] else 0}\t{w['name']}\n"
                for w in sorted(self.windows, key=lambda w: w["index"])
            )
            return subprocess.CompletedProcess(argv, 0, stdout=text, stderr="")
        if argv[1] == "swap-window":
            src_id = argv[argv.index("-s") + 1]
            dst_id = argv[argv.index("-t") + 1]
            src = next(w for w in self.windows if w["id"] == src_id)
            dst = next(w for w in self.windows if w["id"] == dst_id)
            active = next((w for w in self.windows if w["active"]), None)
            active_index = active["index"] if active else None
            src["index"], dst["index"] = dst["index"], src["index"]
            if active_index is not None:
                # tmux keeps the current-window pointer on its index slot.
                for window in self.windows:
                    window["active"] = window["index"] == active_index
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if argv[1] == "select-window":
            target = argv[argv.index("-t") + 1]
            window_id = target.split(":", 1)[-1]
            for window in self.windows:
                window["active"] = window["id"] == window_id
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        raise AssertionError(f"unexpected tmux invocation: {argv}")

    def order(self) -> list[str]:
        return [str(w["name"]) for w in sorted(self.windows, key=lambda w: w["index"])]

    def list_calls(self) -> int:
        return sum(1 for argv in self.calls if argv[1] == "list-windows")

    def active_name(self) -> str | None:
        active = next((window for window in self.windows if window["active"]), None)
        return str(active["name"]) if active else None

    def swap_calls(self) -> int:
        return sum(1 for argv in self.calls if argv[1] == "swap-window")


def rows(*names: str) -> list[dict[str, object]]:
    """Listing rows in sidebar order (highest-ranked first)."""
    return [
        {"session_id": f"sid-{name}", "tmux_session": "codoxear", "tmux_window": name, "final_priority": 1.0}
        for name in names
    ]


def make_sync(tmux: FakeTmuxBin, clock: Clock, *, available: bool = True, log: list[str] | None = None) -> TmuxOrderSync:
    return TmuxOrderSync(
        default_session="codoxear",
        tmux_available=lambda: available,
        which_tmux=lambda name: "/usr/bin/tmux",
        run=tmux.run,
        log=(log if log is not None else []).append,
        clock=clock,
    )


def test_first_generation_is_applied_only_after_the_settle_window() -> None:
    clock = Clock()
    tmux = FakeTmuxBin(["a", "b", "c"])
    sync = make_sync(tmux, clock)

    sync.note_generation(rows("c", "b", "a"))
    assert sync.tick() is None
    assert tmux.calls == []

    clock.advance(SETTLE_SECONDS)
    summaries = sync.tick()

    assert summaries is not None
    assert tmux.order() == ["c", "b", "a"]
    assert tmux.swap_calls() > 0


def test_a_burst_of_generations_settles_into_the_newest_order() -> None:
    clock = Clock()
    tmux = FakeTmuxBin(["a", "b", "c"])
    sync = make_sync(tmux, clock)

    sync.note_generation(rows("b", "a", "c"))
    clock.advance(0.5)
    sync.note_generation(rows("b", "c", "a"))
    clock.advance(0.5)
    sync.note_generation(rows("c", "b", "a"))
    # Still inside the settle window measured from the first unapplied
    # generation, so no pass runs over an order that is still being replaced.
    assert sync.tick() is None

    clock.advance(SETTLE_SECONDS)
    sync.tick()

    # The pass applies the newest generation, never the first one of the burst.
    assert tmux.order() == ["c", "b", "a"]


def test_an_unchanged_order_costs_no_tmux_invocation() -> None:
    clock = Clock()
    tmux = FakeTmuxBin(["a", "b", "c"])
    sync = make_sync(tmux, clock)

    sync.note_generation(rows("c", "b", "a"))
    clock.advance(SETTLE_SECONDS)
    sync.tick()
    calls_after_pass = len(tmux.calls)
    assert calls_after_pass > 1

    clock.advance(SETTLE_SECONDS)
    sync.note_generation(rows("c", "b", "a"))
    clock.advance(SETTLE_SECONDS)

    assert sync.tick() is None
    assert len(tmux.calls) == calls_after_pass


def test_the_floor_bounds_how_often_a_changed_order_is_applied() -> None:
    clock = Clock()
    tmux = FakeTmuxBin(["a", "b", "c"])
    sync = make_sync(tmux, clock)

    sync.note_generation(rows("c", "b", "a"))
    clock.advance(SETTLE_SECONDS)
    sync.tick()
    calls_after_first = len(tmux.calls)

    # A new message in another session reorders the sidebar immediately, but the
    # status bar is only permuted once per floor.
    clock.advance(SETTLE_SECONDS)
    sync.note_generation(rows("b", "a", "c"))
    clock.advance(SETTLE_SECONDS)
    assert sync.tick() is None
    assert len(tmux.calls) == calls_after_first

    clock.advance(MIN_INTERVAL_SECONDS)
    assert sync.tick() is not None
    assert tmux.order() == ["b", "a", "c"]


def test_a_window_moved_by_hand_is_repaired_after_the_relist_interval() -> None:
    clock = Clock()
    tmux = FakeTmuxBin(["a", "b", "c"])
    sync = make_sync(tmux, clock)

    sync.note_generation(rows("a", "b", "c"))
    clock.advance(SETTLE_SECONDS)
    sync.tick()
    calls_after_pass = len(tmux.calls)

    # The user drags a window in tmux; the wanted order never changed.
    tmux.windows[0]["index"], tmux.windows[2]["index"] = 2, 0
    clock.advance(SETTLE_SECONDS)
    sync.note_generation(rows("a", "b", "c"))
    clock.advance(SETTLE_SECONDS)
    assert sync.tick() is None
    assert len(tmux.calls) == calls_after_pass

    # Work is driven by listing generations, never by a timer, so the repair
    # happens on the first generation after the interval expires.
    clock.advance(RELIST_SECONDS)
    sync.note_generation(rows("a", "b", "c"))
    clock.advance(SETTLE_SECONDS)
    assert sync.tick() is not None
    assert tmux.order() == ["a", "b", "c"]


def test_a_host_without_tmux_does_nothing_and_recovers_when_tmux_appears() -> None:
    clock = Clock()
    tmux = FakeTmuxBin(["a", "b"])
    available = {"value": False}
    log: list[str] = []
    sync = TmuxOrderSync(
        default_session="codoxear",
        tmux_available=lambda: available["value"],
        which_tmux=lambda name: "/usr/bin/tmux",
        run=tmux.run,
        log=log.append,
        clock=clock,
    )

    sync.note_generation(rows("b", "a"))
    clock.advance(SETTLE_SECONDS)
    assert sync.tick() is not None
    assert tmux.calls == []

    available["value"] = True
    # Same wanted order, but nothing was ever listed, so the relist guard does
    # not suppress the retry: the next generation (or the floor, whichever comes
    # first) acts.
    clock.advance(MIN_INTERVAL_SECONDS)
    sync.note_generation(rows("b", "a"))
    clock.advance(SETTLE_SECONDS)
    assert sync.tick() is not None
    assert tmux.order() == ["b", "a"]


def test_a_failing_tmux_is_reported_once_and_retried_by_the_next_generation() -> None:
    clock = Clock()
    tmux = FakeTmuxBin(["a", "b"])
    tmux.fail_list = True
    log: list[str] = []
    sync = make_sync(tmux, clock, log=log)

    sync.note_generation(rows("b", "a"))
    clock.advance(SETTLE_SECONDS)
    summaries = sync.tick()
    assert summaries is not None and "no server running" in summaries[0]["error"]
    assert sum("no server running" in line for line in log) == 1

    clock.advance(MIN_INTERVAL_SECONDS)
    sync.note_generation(rows("b", "a"))
    clock.advance(SETTLE_SECONDS)
    sync.tick()
    # Same failure, same message: one line, not one per pass.
    assert sum("no server running" in line for line in log) == 1

    tmux.fail_list = False
    clock.advance(MIN_INTERVAL_SECONDS)
    sync.note_generation(rows("b", "a"))
    clock.advance(SETTLE_SECONDS)
    sync.tick()
    assert tmux.order() == ["b", "a"]


def test_the_viewer_window_is_restored_after_a_pass_that_moved_it() -> None:
    clock = Clock()
    tmux = FakeTmuxBin(["a", "b", "c"], active="c")
    sync = make_sync(tmux, clock)

    sync.note_generation(rows("c", "a", "b"))
    clock.advance(SETTLE_SECONDS)
    sync.tick()

    assert tmux.order() == ["c", "a", "b"]
    # `c` had to move from index 2 to index 0; the viewer is still looking at it.
    assert tmux.active_name() == "c"


def test_a_pass_logs_the_order_it_applied() -> None:
    clock = Clock()
    tmux = FakeTmuxBin(["a", "b", "c"])
    log: list[str] = []
    sync = make_sync(tmux, clock, log=log)

    sync.note_generation(rows("c", "b", "a"))
    clock.advance(SETTLE_SECONDS)
    sync.tick()

    assert any("session=codoxear swaps=" in line and "order=c,b,a" in line for line in log)


def test_fast_polling_cannot_starve_the_mirror() -> None:
    clock = Clock()
    tmux = FakeTmuxBin(["a", "b", "c"])
    sync = make_sync(tmux, clock)

    # A client polling faster than the settle window keeps producing
    # generations. The settle window is anchored to the first unapplied
    # generation, so it must still expire instead of restarting forever.
    for _ in range(6):
        sync.note_generation(rows("c", "b", "a"))
        clock.advance(1.0)
        sync.tick()

    assert tmux.order() == ["c", "b", "a"]


def test_a_pass_records_the_generation_it_applied() -> None:
    clock = Clock()
    tmux = FakeTmuxBin(["a", "b"])
    sync = make_sync(tmux, clock)

    sync.note_generation(rows("b", "a"))
    clock.advance(SETTLE_SECONDS)
    sync.tick()
    assert sync._applied_generation == 1

    # A newer generation that carries the same order is a no-op, and it must
    # retire the unapplied marker so a later relist check can run.
    clock.advance(SETTLE_SECONDS)
    sync.note_generation(rows("b", "a"))
    clock.advance(SETTLE_SECONDS)
    assert sync.tick() is None
    assert sync._unapplied_since is None


def test_a_generation_that_arrives_during_a_pass_is_still_applied() -> None:
    clock = Clock()
    tmux = FakeTmuxBin(["a", "b", "c"])
    sync = make_sync(tmux, clock)

    # The swap itself is the moment a new listing generation can land: a client
    # polls while the pass is running. That generation must survive the pass.
    original_swap = tmux.run

    def swap_then_note(argv: list[str], **kwargs):
        result = original_swap(argv, **kwargs)
        if argv[1] == "swap-window":
            sync.note_generation(rows("b", "a", "c"))
        return result

    sync._run = swap_then_note

    sync.note_generation(rows("c", "b", "a"))
    clock.advance(SETTLE_SECONDS)
    sync.tick()
    assert tmux.order() == ["c", "b", "a"]

    clock.advance(MIN_INTERVAL_SECONDS)
    assert sync.tick() is not None
    assert tmux.order() == ["b", "a", "c"]
