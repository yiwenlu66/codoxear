"""Behaviour tests for the sidebar-order → tmux-window-order mirror.

The tmux side is modelled as a window list that a swap mutates, so these tests
assert the *permutation* and the viewer-preservation rule rather than the shape
of the command strings (the argv contract lives in `test_tmux_runtime.py`, and
real tmux behaviour is pinned by `scripts/docker_tmux_order_verify.sh`).
"""

from __future__ import annotations

import pytest

from codoxear.tmux_order import (
    SwapStep,
    WindowTarget,
    group_targets,
    plan_next_swap,
    reconcile,
    reconcile_windows,
    target_signature,
    window_targets,
)
from codoxear.tmux_runtime import TmuxCommandError


class FakeTmux:
    """Window list with tmux's measured swap semantics.

    `swap-window` (with or without `-d`, on the versions checked) leaves the
    session's current-window pointer on its index slot, so a viewer looking at a
    window that moves ends up looking at a different window's pane.
    `select-window` is what puts it back. Modelling the pointer as
    index-following is the point: it is what makes a missing restore fail a test
    instead of passing silently.
    """

    def __init__(self, windows: list[dict[str, object]]) -> None:
        self.windows = [dict(window) for window in windows]
        self.swaps: list[SwapStep] = []
        self.selects: list[str] = []
        self.list_calls = 0

    def list_windows(self, session_name: str) -> list[dict[str, object]]:
        self.list_calls += 1
        return [dict(window) for window in sorted(self.windows, key=lambda w: int(w["index"]))]

    def _window(self, window_id: str) -> dict[str, object]:
        return next(window for window in self.windows if window["id"] == window_id)

    def swap(self, session_name: str, step: SwapStep) -> None:
        active = next((window for window in self.windows if window["active"]), None)
        active_index = int(active["index"]) if active else None
        src = self._window(step.src)
        dst = self._window(step.dst)
        src["index"], dst["index"] = dst["index"], src["index"]
        if active_index is not None:
            for window in self.windows:
                window["active"] = int(window["index"]) == active_index
        self.swaps.append(step)

    def select(self, session_name: str, window_id: str) -> None:
        for window in self.windows:
            window["active"] = window["id"] == window_id
        self.selects.append(window_id)

    def order(self) -> list[str]:
        return [str(window["name"]) for window in sorted(self.windows, key=lambda w: int(w["index"]))]

    def active_id(self) -> str | None:
        active = next((window for window in self.windows if window["active"]), None)
        return str(active["id"]) if active else None


def windows(*specs: tuple[str, str, bool]) -> list[dict[str, object]]:
    return [
        {"index": index, "id": window_id, "name": name, "active": active}
        for index, (window_id, name, active) in enumerate(specs)
    ]


def targets(*names: str, session: str = "codoxear") -> list[WindowTarget]:
    return [WindowTarget(session_name=session, window_name=name, session_id=f"sid-{name}") for name in names]


# -- projection --------------------------------------------------------


def test_window_targets_keeps_row_order_and_drops_rows_without_a_window() -> None:
    rows = [
        {"session_id": "a", "tmux_session": "codoxear", "tmux_window": "alpha"},
        {"session_id": "b", "transport": "direct", "tmux_window": None},
        {"session_id": "c", "tmux_window": "gamma"},
        {"session_id": "d", "tmux_session": "", "tmux_window": "delta"},
        {"session_id": "e", "tmux_session": "codoxear", "tmux_window": "  "},
    ]

    projected = window_targets(rows, default_session="codoxear")

    assert [(t.session_name, t.window_name) for t in projected] == [
        ("codoxear", "alpha"),
        # A row with a window but no session name belongs to the configured one.
        ("codoxear", "gamma"),
        ("codoxear", "delta"),
    ]


def test_window_targets_keeps_the_first_row_when_two_rows_name_one_window() -> None:
    rows = [
        {"session_id": "higher", "tmux_session": "codoxear", "tmux_window": "same"},
        {"session_id": "lower", "tmux_session": "codoxear", "tmux_window": "same"},
    ]

    projected = window_targets(rows, default_session="codoxear")

    assert [t.session_id for t in projected] == ["higher"]


def test_targets_group_by_session_in_first_appearance_order() -> None:
    grouped = group_targets(
        [
            WindowTarget("b", "w1", "s1"),
            WindowTarget("a", "w2", "s2"),
            WindowTarget("b", "w3", "s3"),
        ]
    )

    assert list(grouped) == ["b", "a"]
    assert [t.window_name for t in grouped["b"]] == ["w1", "w3"]
    assert target_signature(grouped["a"]) == (("a", "w2"),)


# -- swap planning -----------------------------------------------------


def test_plan_next_swap_returns_nothing_for_an_ordered_window_list() -> None:
    listing = windows(("W0", "a", False), ("W1", "b", True), ("W2", "c", False))

    assert plan_next_swap(target_ids=["W0", "W1", "W2"], windows=listing) is None


def test_plan_next_swap_swaps_the_slot_that_is_already_wrong() -> None:
    listing = windows(("W0", "a", False), ("W1", "b", False), ("W2", "c", False))

    step = plan_next_swap(target_ids=["W2", "W1", "W0"], windows=listing)

    assert step == SwapStep(src="W0", dst="W2")


@pytest.mark.parametrize("active_id", ["W0", "W1", "W2", None])
def test_plan_next_swap_is_independent_of_which_window_is_current(active_id: str | None) -> None:
    # The viewer is restored after the pass, so the plan must not depend on the
    # current window (that dependency is what made `-d` version-fragile).
    listing = windows(("W0", "a", False), ("W1", "b", False), ("W2", "c", False))
    for window in listing:
        window["active"] = window["id"] == active_id

    step = plan_next_swap(target_ids=["W2", "W1", "W0"], windows=listing)

    assert step == SwapStep(src="W0", dst="W2")


def test_plan_next_swap_ignores_windows_the_targets_do_not_cover() -> None:
    listing = windows(("USER", "mine", False), ("W0", "a", False), ("W1", "b", False))
    for window in listing:
        window["active"] = window["id"] == "USER"

    step = plan_next_swap(target_ids=["W1", "W0"], windows=listing)

    assert step == SwapStep(src="W0", dst="W1")


def test_plan_next_swap_survives_a_target_whose_window_disappeared() -> None:
    listing = windows(("W0", "a", False), ("W2", "c", False))

    step = plan_next_swap(target_ids=["W2", "GONE", "W0"], windows=listing)

    assert step == SwapStep(src="W0", dst="W2")


# -- reconcile ---------------------------------------------------------


def test_reconcile_converges_to_the_requested_order_and_is_idempotent() -> None:
    tmux = FakeTmux(windows(("W0", "a", False), ("W1", "b", False), ("W2", "c", False), ("W3", "d", False)))

    first = reconcile_windows(
        session_name="codoxear",
        targets=targets("d", "c", "b", "a"),
        list_windows=tmux.list_windows,
        swap=tmux.swap,
        select=tmux.select,
    )

    assert tmux.order() == ["d", "c", "b", "a"]
    assert first["converged"] is True
    # Two swaps, not three: the first parks `d` at index 0 and the second parks
    # `c`, which leaves `b` and `a` already correct.
    assert first["swaps"] == 2
    assert first["order"] == ["d", "c", "b", "a"]

    second = reconcile_windows(
        session_name="codoxear",
        targets=targets("d", "c", "b", "a"),
        list_windows=tmux.list_windows,
        swap=tmux.swap,
        select=tmux.select,
    )

    assert second["swaps"] == 0
    assert len(tmux.swaps) == 2


def test_reconcile_permutes_codoxear_windows_only_among_their_own_slots() -> None:
    tmux = FakeTmux(
        windows(
            ("USER1", "mine-1", False),
            ("W0", "a", False),
            ("W1", "b", True),
            ("W2", "c", False),
            ("USER2", "mine-2", False),
            ("W3", "d", False),
        )
    )
    viewer = tmux.active_id()

    reconcile_windows(
        session_name="codoxear",
        targets=targets("d", "c", "b", "a"),
        list_windows=tmux.list_windows,
        swap=tmux.swap,
        select=tmux.select,
    )

    assert tmux.order() == ["mine-1", "d", "c", "b", "mine-2", "a"]
    # The viewer keeps looking at its own window; only its number changed.
    assert tmux.active_id() == viewer
    # No swap ever exchanged a codoxear window with a foreign one.
    owned = {"W0", "W1", "W2", "W3"}
    assert all(step.src in owned and step.dst in owned for step in tmux.swaps)


def test_reconcile_restores_the_viewer_window_after_it_is_renumbered() -> None:
    tmux = FakeTmux(windows(("W0", "a", False), ("W1", "b", True), ("W2", "c", False), ("W3", "d", False)))

    summary = reconcile_windows(
        session_name="codoxear",
        targets=targets("c", "a", "b", "d"),
        list_windows=tmux.list_windows,
        swap=tmux.swap,
        select=tmux.select,
    )

    assert summary["converged"] is True
    assert summary["restored"] is True
    assert tmux.order() == ["c", "a", "b", "d"]
    # The viewer is back on its own pane, even though that window moved from
    # index 1 to index 2.
    assert tmux.active_id() == "W1"
    assert tmux.selects == ["W1"]


def test_reconcile_does_not_reselect_when_the_current_window_did_not_move() -> None:
    tmux = FakeTmux(windows(("W0", "a", False), ("W1", "b", True), ("W2", "c", False)))

    summary = reconcile_windows(
        session_name="codoxear",
        targets=targets("c", "b", "a"),
        list_windows=tmux.list_windows,
        swap=tmux.swap,
        select=tmux.select,
    )

    assert summary["swaps"] == 1
    assert summary["restored"] is False
    assert tmux.selects == []
    assert tmux.active_id() == "W1"


def test_reconcile_skips_missing_and_ambiguous_windows_instead_of_guessing() -> None:
    tmux = FakeTmux(
        windows(
            ("W0", "a", False),
            ("W1", "dup", False),
            ("W2", "dup", False),
            ("W3", "b", False),
        )
    )

    summary = reconcile_windows(
        session_name="codoxear",
        targets=targets("b", "dup", "gone", "a"),
        list_windows=tmux.list_windows,
        swap=tmux.swap,
        select=tmux.select,
    )

    assert summary["skipped"] == [
        {"window": "dup", "session_id": "sid-dup", "reason": "ambiguous"},
        {"window": "gone", "session_id": "sid-gone", "reason": "missing"},
    ]
    assert tmux.order() == ["b", "dup", "dup", "a"]
    assert summary["converged"] is True


def test_reconcile_isolates_a_failing_tmux_session() -> None:
    healthy = FakeTmux(windows(("W0", "a", False), ("W1", "b", False)))

    def list_windows(session_name: str) -> list[dict[str, object]]:
        if session_name == "broken":
            raise TmuxCommandError("list-windows", "no server running")
        return healthy.list_windows(session_name)

    summaries = reconcile(
        targets=[
            WindowTarget("broken", "x", "sid-x"),
            WindowTarget("codoxear", "b", "sid-b"),
            WindowTarget("codoxear", "a", "sid-a"),
        ],
        list_windows=list_windows,
        swap=healthy.swap,
        select=healthy.select,
    )

    assert [summary["session"] for summary in summaries] == ["broken", "codoxear"]
    assert "no server running" in summaries[0]["error"]
    assert summaries[1]["swaps"] == 1
    assert healthy.order() == ["b", "a"]


def test_reconcile_stops_instead_of_spinning_when_a_swap_does_not_take_effect() -> None:
    stuck = FakeTmux(windows(("W0", "a", False), ("W1", "b", False), ("W2", "c", False)))

    summary = reconcile_windows(
        session_name="codoxear",
        targets=targets("c", "b", "a"),
        list_windows=stuck.list_windows,
        swap=lambda session_name, step: None,
        select=stuck.select,
    )

    assert summary["converged"] is False
    # Bounded by the number of target windows: no unbounded retry loop.
    assert summary["swaps"] == 3
