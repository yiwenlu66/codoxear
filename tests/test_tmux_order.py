"""Behaviour tests for the sidebar → tmux window layout mirror.

The tmux side is modelled as a window list that the mirror's operations mutate,
so these tests assert the *layout* (positions, titles, identity) rather than the
shape of command strings (the argv contract lives in `test_tmux_runtime.py`, and
real tmux behaviour is pinned by `scripts/docker_tmux_order_verify.sh`).
"""

from __future__ import annotations

import pytest

from codoxear.tmux_order import (
    SwapStep,
    TmuxOps,
    WindowTarget,
    group_targets,
    identity_token,
    is_consecutive,
    plan_next_swap,
    reconcile,
    reconcile_windows,
    resolve_targets,
    target_signature,
    window_targets,
)
from codoxear.tmux_runtime import TmuxCommandError


class FakeTmux:
    """A tmux session whose windows the mirror's operations mutate.

    Models the measured behaviour: `swap-window` leaves the session's
    current-window pointer on its index slot (so a viewer looking at a window
    that moves would see another pane), and `select-window` is what puts it
    back.
    """

    def __init__(
        self,
        specs: list[tuple[str, str, bool, str | None]],
        *,
        base: int = 0,
        indices: list[int] | None = None,
    ) -> None:
        self.base = base
        slots = indices if indices is not None else [base + offset for offset in range(len(specs))]
        self.windows = [
            {"index": slot, "id": window_id, "name": name, "active": active, "nonce": nonce}
            for slot, (window_id, name, active, nonce) in zip(slots, specs)
        ]
        self.calls: list[tuple] = []

    # -- tmux verbs ----------------------------------------------------
    def list_windows(self, session_name: str) -> list[dict[str, object]]:
        return [dict(window) for window in sorted(self.windows, key=lambda w: int(w["index"]))]

    def swap(self, session_name: str, step) -> None:
        self.calls.append(("swap", step.src, step.dst))
        active = next((window for window in self.windows if window["active"]), None)
        active_index = int(active["index"]) if active else None
        src = self._window(step.src)
        dst = self._window(step.dst)
        src["index"], dst["index"] = dst["index"], src["index"]
        if active_index is not None:
            for window in self.windows:
                window["active"] = int(window["index"]) == active_index

    def select(self, session_name: str, window_id: str) -> None:
        self.calls.append(("select", window_id))
        for window in self.windows:
            window["active"] = window["id"] == window_id

    def rename(self, session_name: str, window_id: str, name: str) -> None:
        self.calls.append(("rename", window_id, name))
        self._window(window_id)["name"] = name

    def stamp(self, session_name: str, window_id: str, token: str) -> None:
        self.calls.append(("stamp", window_id, token))
        self._window(window_id)["nonce"] = token

    def renumber(self, session_name: str) -> None:
        self.calls.append(("renumber",))
        for offset, window in enumerate(sorted(self.windows, key=lambda w: int(w["index"]))):
            window["index"] = self.base + offset

    def base_index(self, session_name: str) -> int | None:
        return self.base

    # -- view helpers --------------------------------------------------
    def _window(self, window_id: str) -> dict[str, object]:
        return next(window for window in self.windows if window["id"] == window_id)

    def ops(self) -> TmuxOps:
        return TmuxOps(
            list_windows=self.list_windows,
            swap=self.swap,
            select=self.select,
            rename=self.rename,
            stamp=self.stamp,
            renumber=self.renumber,
            base_index=self.base_index,
        )

    def order(self) -> list[str]:
        return [str(window["name"]) for window in sorted(self.windows, key=lambda w: int(w["index"]))]

    def indices(self) -> list[int]:
        return [int(window["index"]) for window in sorted(self.windows, key=lambda w: int(w["index"]))]

    def active_id(self) -> str | None:
        active = next((window for window in self.windows if window["active"]), None)
        return str(active["id"]) if active else None

    def names_by_id(self) -> dict[str, str]:
        return {str(window["id"]): str(window["name"]) for window in self.windows}


def targets(*specs: tuple[str, str, str], session: str = "codoxear") -> list[WindowTarget]:
    """`specs` are (display_name, launch_name, token) in sidebar order."""
    return [
        WindowTarget(
            session_name=session,
            display_name=display,
            launch_name=launch,
            session_id=f"sid-{launch}",
            token=token,
        )
        for display, launch, token in specs
    ]


def reconcile_once(tmux: FakeTmux, session_targets: list[WindowTarget]) -> dict:
    return reconcile_windows(session_name="codoxear", targets=session_targets, ops=tmux.ops())


# -- projection --------------------------------------------------------


def test_window_targets_keeps_row_order_and_drops_rows_without_a_window() -> None:
    rows = [
        {"session_id": "a", "tmux_session": "codoxear", "tmux_window": "alpha-1", "display_name": "alpha", "spawn_nonce": "aaaaaaaaaaaa"},
        {"session_id": "b", "transport": "direct", "tmux_window": None},
        {"session_id": "c", "tmux_window": "gamma-3", "display_name": "gamma"},
        {"session_id": "d", "tmux_session": "", "tmux_window": "delta-4", "display_name": "delta"},
        {"session_id": "e", "tmux_session": "codoxear", "tmux_window": "   "},
    ]

    projected = window_targets(rows, default_session="codoxear")

    assert [(t.session_name, t.display_name, t.launch_name) for t in projected] == [
        ("codoxear", "alpha", "alpha-1"),
        # A row with a window but no session name belongs to the configured one.
        ("codoxear", "gamma", "gamma-3"),
        ("codoxear", "delta", "delta-4"),
    ]
    # The broker's nonce is the identity token when it is hex-shaped.
    assert projected[0].token == "aaaaaaaaaaaa"
    # Otherwise the token is a stable digest of the launch name, so a window with
    # no recorded nonce can still be found after it is renamed.
    assert projected[1].token == identity_token("gamma-3", None)
    assert identity_token("gamma-3", None) == identity_token("gamma-3", "")


def test_window_targets_keeps_the_first_row_for_a_shared_window() -> None:
    rows = [
        {"session_id": "higher", "tmux_window": "same-1", "spawn_nonce": "abc123abc123"},
        {"session_id": "lower", "tmux_window": "same-1", "spawn_nonce": "abc123abc123"},
    ]

    projected = window_targets(rows, default_session="codoxear")

    assert [t.session_id for t in projected] == ["higher"]


def test_targets_group_by_session_in_first_appearance_order() -> None:
    grouped = group_targets(
        [
            WindowTarget("b", "one", "w1", "s1", "t1"),
            WindowTarget("a", "two", "w2", "s2", "t2"),
            WindowTarget("b", "three", "w3", "s3", "t3"),
        ]
    )

    assert list(grouped) == ["b", "a"]
    assert [t.display_name for t in grouped["b"]] == ["one", "three"]
    # The signature covers the title too: an alias edit alone is a change.
    assert target_signature(grouped["a"]) == (("a", "t2", "two"),)


# -- identity ----------------------------------------------------------


def test_resolve_targets_prefers_the_stamp_and_falls_back_to_the_launch_name() -> None:
    windows = [
        {"index": 0, "id": "@0", "name": "renamed-by-us", "active": False, "nonce": "tok-a"},
        {"index": 1, "id": "@1", "name": "legacy-2", "active": False, "nonce": None},
        {"index": 2, "id": "@2", "name": "dup", "active": False, "nonce": None},
        {"index": 3, "id": "@3", "name": "dup", "active": False, "nonce": None},
    ]
    resolved, skipped = resolve_targets(
        [
            # Its title no longer matches the sidecar name; the stamp finds it.
            WindowTarget("codoxear", "alpha", "alpha-1", "sid-a", "tok-a"),
            # A pre-stamp window is still reachable by name when unambiguous.
            WindowTarget("codoxear", "beta", "legacy-2", "sid-b", "tok-b"),
            # ... but two candidates mean no candidate, and neither is claimed.
            WindowTarget("codoxear", "gamma", "dup", "sid-c", "tok-c"),
            WindowTarget("codoxear", "delta", "dup", "sid-d", "tok-d"),
        ],
        windows,
    )

    assert [(t.session_id, window_id) for t, window_id in resolved] == [("sid-a", "@0"), ("sid-b", "@1")]
    assert skipped == [
        {"window": "dup", "session_id": "sid-c", "reason": "ambiguous"},
        {"window": "dup", "session_id": "sid-d", "reason": "ambiguous"},
    ]


# -- layout ------------------------------------------------------------


def test_is_consecutive_respects_the_configured_base_index() -> None:
    assert is_consecutive([0, 1, 2], 0) is True
    assert is_consecutive([1, 2, 3], 1) is True
    assert is_consecutive([1, 2, 3], 0) is False
    assert is_consecutive([0, 2, 3], 0) is False
    assert is_consecutive([], 0) is True


def test_plan_next_swap_returns_nothing_for_an_ordered_window_list() -> None:
    windows = [
        {"index": 0, "id": "W0", "name": "a", "active": False, "nonce": None},
        {"index": 1, "id": "W1", "name": "b", "active": True, "nonce": None},
        {"index": 2, "id": "W2", "name": "c", "active": False, "nonce": None},
    ]

    assert plan_next_swap(target_ids=["W0", "W1", "W2"], windows=windows) is None


@pytest.mark.parametrize("active_id", ["W0", "W1", "W2", None])
def test_plan_next_swap_is_independent_of_which_window_is_current(active_id: str | None) -> None:
    # The viewer is restored after the pass, so the plan must not depend on the
    # current window.
    windows = [
        {"index": 0, "id": "W0", "name": "a", "active": active_id == "W0", "nonce": None},
        {"index": 1, "id": "W1", "name": "b", "active": active_id == "W1", "nonce": None},
        {"index": 2, "id": "W2", "name": "c", "active": active_id == "W2", "nonce": None},
    ]

    assert plan_next_swap(target_ids=["W2", "W1", "W0"], windows=windows) == SwapStep(src="W0", dst="W2")


def test_plan_next_swap_survives_a_target_whose_window_disappeared() -> None:
    windows = [
        {"index": 0, "id": "W0", "name": "a", "active": False, "nonce": None},
        {"index": 2, "id": "W2", "name": "c", "active": False, "nonce": None},
    ]

    assert plan_next_swap(target_ids=["W2", "GONE", "W0"], windows=windows) == SwapStep(src="W0", dst="W2")


# -- reconcile: names --------------------------------------------------


def test_reconcile_titles_each_window_with_the_sidebar_name() -> None:
    tmux = FakeTmux([("@1", "alpha-1", False, None), ("@2", "beta-2", False, None)])

    summary = reconcile_once(tmux, targets(("alpha", "alpha-1", "tok-a"), ("beta", "beta-2", "tok-b")))

    assert summary["renamed"] == 2
    assert tmux.order() == ["alpha", "beta"]
    assert tmux.active_id() is None


def test_each_window_is_stamped_before_it_is_renamed() -> None:
    tmux = FakeTmux([("@1", "alpha-1", False, None)])

    reconcile_once(tmux, targets(("alpha", "alpha-1", "tok-a")))

    # Order matters: a rename without a stamp would lose the window forever.
    assert tmux.calls == [("stamp", "@1", "tok-a"), ("rename", "@1", "alpha")]


def test_a_second_pass_finds_renamed_windows_by_their_stamp() -> None:
    tmux = FakeTmux([("@1", "alpha-1", False, None), ("@2", "beta-2", False, None)])
    session_targets = targets(("alpha", "alpha-1", "tok-a"), ("beta", "beta-2", "tok-b"))

    reconcile_once(tmux, session_targets)
    second = reconcile_once(tmux, session_targets)

    assert second["skipped"] == []
    assert second["stamped"] == 0
    assert second["renamed"] == 0
    assert second["swaps"] == 0
    assert tmux.order() == ["alpha", "beta"]


def test_an_alias_change_only_renames() -> None:
    tmux = FakeTmux([("@1", "alpha-1", False, "tok-a"), ("@2", "beta-2", False, "tok-b")])
    reconcile_once(tmux, targets(("alpha", "alpha-1", "tok-a"), ("beta", "beta-2", "tok-b")))
    tmux.calls.clear()

    summary = reconcile_once(tmux, targets(("alpha (renamed)", "alpha-1", "tok-a"), ("beta", "beta-2", "tok-b")))

    assert summary["renamed"] == 1
    assert summary["swaps"] == 0
    assert tmux.order() == ["alpha (renamed)", "beta"]
    assert tmux.calls == [("rename", "@1", "alpha (renamed)")]


def test_a_window_without_a_title_or_with_an_unusable_title_keeps_its_name() -> None:
    tmux = FakeTmux(
        [("@1", "alpha-1", False, "tok-a"), ("@2", "beta-2", False, "tok-b"), ("@3", "g-3", False, "tok-g")]
    )

    reconcile_once(
        tmux,
        targets(("", "alpha-1", "tok-a"), ("-danger", "beta-2", "tok-b"), ("two\nlines", "g-3", "tok-g")),
    )

    # Empty, flag-like and control-character names would break the call or the
    # status bar; those windows keep their launch names.
    assert tmux.order() == ["alpha-1", "beta-2", "g-3"]


# -- reconcile: positions ----------------------------------------------


def test_reconcile_puts_codoxear_windows_first_in_sidebar_order() -> None:
    tmux = FakeTmux(
        [
            ("@0", "my-shell", False, None),
            ("@1", "alpha-1", False, "tok-a"),
            ("@2", "beta-2", True, "tok-b"),
            ("@3", "my-other-shell", False, None),
            ("@4", "gamma-3", False, "tok-g"),
        ]
    )
    viewer = tmux.active_id()

    summary = reconcile_once(
        tmux,
        targets(("gamma", "gamma-3", "tok-g"), ("beta", "beta-2", "tok-b"), ("alpha", "alpha-1", "tok-a")),
    )

    # Sidebar order, then the windows no session owns in their existing order.
    assert tmux.order() == ["gamma", "beta", "alpha", "my-shell", "my-other-shell"]
    assert tmux.indices() == [0, 1, 2, 3, 4]
    assert summary["converged"] is True
    # The viewer kept its own pane while the numbering under it changed.
    assert tmux.active_id() == viewer == "@2"


def test_reconcile_closes_index_gaps_from_closed_windows() -> None:
    tmux = FakeTmux(
        [("@1", "alpha-1", False, "tok-a"), ("@9", "beta-2", False, "tok-b")],
        indices=[0, 9],
    )

    summary = reconcile_once(tmux, targets(("alpha", "alpha-1", "tok-a"), ("beta", "beta-2", "tok-b")))

    assert summary["renumbered"] is True
    assert tmux.indices() == [0, 1]
    assert tmux.order() == ["alpha", "beta"]


def test_reconcile_respects_a_non_zero_base_index() -> None:
    tmux = FakeTmux([("@1", "alpha-1", False, "tok-a"), ("@2", "beta-2", False, "tok-b")], base=1)

    reconcile_once(tmux, targets(("beta", "beta-2", "tok-b"), ("alpha", "alpha-1", "tok-a")))

    assert tmux.indices() == [1, 2]
    assert tmux.order() == ["beta", "alpha"]


def test_reconcile_skips_a_missing_window_and_leaves_its_slot_to_the_others() -> None:
    tmux = FakeTmux([("@1", "alpha-1", False, "tok-a"), ("@2", "beta-2", False, "tok-b")])

    summary = reconcile_once(
        tmux,
        targets(("alpha", "alpha-1", "tok-a"), ("gone", "gone-9", "tok-g"), ("beta", "beta-2", "tok-b")),
    )

    assert summary["skipped"] == [{"window": "gone-9", "session_id": "sid-gone-9", "reason": "missing"}]
    assert tmux.order() == ["alpha", "beta"]


def test_reconcile_is_idempotent() -> None:
    tmux = FakeTmux([("@0", "shell", False, None), ("@1", "alpha-1", False, "tok-a"), ("@2", "beta-2", False, "tok-b")])
    session_targets = targets(("beta", "beta-2", "tok-b"), ("alpha", "alpha-1", "tok-a"))

    reconcile_once(tmux, session_targets)
    tmux.calls.clear()
    second = reconcile_once(tmux, session_targets)

    assert second["swaps"] == 0
    assert second["renamed"] == 0
    assert second["stamped"] == 0
    assert tmux.calls == []


def test_reconcile_stops_instead_of_spinning_when_a_swap_does_not_take_effect() -> None:
    tmux = FakeTmux([("@1", "alpha-1", False, "tok-a"), ("@2", "beta-2", False, "tok-b"), ("@3", "g-3", False, "tok-g")])
    ops = tmux.ops()
    stuck = TmuxOps(
        list_windows=ops.list_windows,
        swap=lambda session_name, step: None,
        select=ops.select,
        rename=ops.rename,
        stamp=ops.stamp,
        renumber=ops.renumber,
        base_index=ops.base_index,
    )

    summary = reconcile_windows(
        session_name="codoxear",
        targets=targets(("g", "g-3", "tok-g"), ("beta", "beta-2", "tok-b"), ("alpha", "alpha-1", "tok-a")),
        ops=stuck,
    )

    assert summary["converged"] is False
    assert summary["swaps"] == 3


def test_reconcile_isolates_a_failing_tmux_session() -> None:
    healthy = FakeTmux([("@1", "alpha-1", False, "tok-a"), ("@2", "beta-2", False, "tok-b")])
    base_ops = healthy.ops()

    def list_windows(session_name: str) -> list[dict[str, object]]:
        if session_name == "broken":
            raise TmuxCommandError("list-windows", "no server running")
        return healthy.list_windows(session_name)

    summaries = reconcile(
        targets=[
            WindowTarget("broken", "x", "x-1", "sid-x", "tok-x"),
            WindowTarget("codoxear", "beta", "beta-2", "sid-b", "tok-b"),
            WindowTarget("codoxear", "alpha", "alpha-1", "sid-a", "tok-a"),
        ],
        ops=TmuxOps(
            list_windows=list_windows,
            swap=base_ops.swap,
            select=base_ops.select,
            rename=base_ops.rename,
            stamp=base_ops.stamp,
            renumber=base_ops.renumber,
            base_index=base_ops.base_index,
        ),
    )

    assert [summary["session"] for summary in summaries] == ["broken", "codoxear"]
    assert "no server running" in summaries[0]["error"]
    assert summaries[1]["renamed"] == 2
    assert healthy.order() == ["beta", "alpha"]


def test_reconcile_restores_the_viewer_window_after_it_is_renumbered() -> None:
    tmux = FakeTmux(
        [("@1", "alpha-1", False, "tok-a"), ("@2", "beta-2", True, "tok-b"), ("@3", "g-3", False, "tok-g")]
    )

    summary = reconcile_once(
        tmux,
        targets(("g", "g-3", "tok-g"), ("alpha", "alpha-1", "tok-a"), ("beta", "beta-2", "tok-b")),
    )

    assert summary["restored"] is True
    assert tmux.order() == ["g", "alpha", "beta"]
    assert tmux.active_id() == "@2"


def test_reconcile_does_not_reselect_when_the_current_window_did_not_move() -> None:
    # The current window is a shell that no session owns and no swap touches.
    tmux = FakeTmux([("@0", "alpha-1", False, "tok-a"), ("@1", "beta-2", False, "tok-b"), ("@2", "shell", True, None)])

    summary = reconcile_once(tmux, targets(("beta", "beta-2", "tok-b"), ("alpha", "alpha-1", "tok-a")))

    assert summary["swaps"] == 1
    assert summary["restored"] is False
    assert tmux.active_id() == "@2"
    assert tmux.order() == ["beta", "alpha", "shell"]
    assert not [call for call in tmux.calls if call[0] == "select"]
