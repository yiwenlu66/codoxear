from copy import deepcopy

import pytest

from codoxear.session_listing import listing_priority, reorder_target_offset, sort_session_rows
from test_snooze_indefinite import ui_harness


@pytest.mark.parametrize("tp,above,below,expected", [
    (0.2, 0.8, 0.4, 0.4),
    (0.2, None, 0.8, 0.605),
    (0.2, 0.4, None, 0.195),
    (1.0, None, 0.8, -0.195),
    (1.0, 0.4, None, -0.605),
])
def test_midpoint_top_bottom_and_fresh_dragged_offset(tp, above, below, expected):
    offset, applied = reorder_target_offset(time_priority=tp, above_priority=above, below_priority=below)
    assert offset == pytest.approx(expected)
    assert applied is True


def test_target_requires_at_least_one_neighbour():
    with pytest.raises(ValueError, match="above or below required"):
        reorder_target_offset(time_priority=0.2, above_priority=None, below_priority=None)


@pytest.mark.parametrize("tp,above,below,expected", [
    (0.0, None, 1.0, 1.0),
    (1.0, 0.0, None, -1.0),
])
def test_offset_clamp_reports_not_applied(tp, above, below, expected):
    assert reorder_target_offset(time_priority=tp, above_priority=above, below_priority=below) == (expected, False)


@pytest.mark.parametrize("tp,above,below,expected_offset,expected_applied", [
    (0.4, None, 1.0, 0.605, False),
    (1.0, None, 1.0, 0.005, False),
    (0.4, 1.0, 1.0, 0.6, False),
    (1.0, 1.0, None, -0.005, True),
    (0.4, 1.0, 0.5, 0.35, True),
])
def test_exact_one_neighbour_only_blocks_a_saturated_tie(tp, above, below, expected_offset, expected_applied):
    offset, applied = reorder_target_offset(time_priority=tp, above_priority=above, below_priority=below)
    assert offset == pytest.approx(expected_offset)
    assert applied is expected_applied


def project_rows(h):
    rows = []
    for sid, session in h.sessions.items():
        updated = session.last_chat_ts if session.last_chat_ts is not None else session.start_ts
        priority = listing_priority(
            now_ts=h.now, updated_ts=updated,
            priority_offset=h.store.sidebar_meta.get(sid, {}).get("priority_offset", 0.0),
            blocked=False, snoozed=False,
            half_life_seconds=h.listing.priority_half_life_seconds,
            bucket_seconds=h.listing.priority_bucket_seconds,
        )
        rows.append({"session_id": sid, "updated_ts": updated, "start_ts": session.start_ts,
                     "final_priority": priority.final_priority})
    sort_session_rows(rows)
    return [row["session_id"] for row in rows]


@pytest.mark.parametrize("above,below,wanted", [
    ("above", "below", ["above", "dragged", "below"]),
    (None, "above", ["dragged", "above", "below"]),
    ("below", None, ["above", "below", "dragged"]),
])
@pytest.mark.parametrize("fresh", [False, True])
def test_manager_changes_only_dragged_offset_and_real_sort_obeys_drop(tmp_path, above, below, wanted, fresh):
    h = ui_harness(tmp_path, half_life=10000.0, bucket=17.0)
    if fresh:
        h.sessions["dragged"].last_chat_ts = h.now
    h.store.sidebar_meta = {
        "dragged": {"priority_offset": 0.1, "snoozed_indefinitely": True, "dependency_session_id": "above"},
        "above": {"priority_offset": 0.2},
        "below": {"priority_offset": -0.005},
    }
    original = deepcopy(h.store.sidebar_meta)
    result = h.manager.reorder_session("dragged", above=above, below=below)
    assert result["applied"] is True
    assert h.store.sidebar_meta["above"] == original["above"]
    assert h.store.sidebar_meta["below"] == original["below"]
    assert h.store.sidebar_meta["dragged"] == {**original["dragged"], "priority_offset": result["priority_offset"]}
    assert h.saves == [True]
    assert h.store.load_sidebar_meta() == h.store.sidebar_meta
    # Reorder intentionally leaves the source asleep. These projections model
    # the Now ordering once the separate narrow wake operation clears snooze.
    assert project_rows(h) == wanted
    h.manager.snooze_session("dragged", indefinite=False)
    assert h.store.sidebar_meta["dragged"]["priority_offset"] == result["priority_offset"]


def test_last_chat_timestamp_and_existing_neighbour_offsets_drive_target(tmp_path):
    h = ui_harness(tmp_path, half_life=3600.0, bucket=600.0)
    h.sessions["above"].last_chat_ts = h.now - 3600
    h.sessions["below"].last_chat_ts = h.now - 7200
    h.sessions["dragged"].last_chat_ts = h.now - 10800
    h.store.sidebar_meta = {"above": {"priority_offset": 0.1}, "below": {"priority_offset": 0.05}}
    assert h.manager.reorder_session("dragged", above="above", below="below") == {
        "priority_offset": 0.325, "applied": True,
    }
    assert project_rows(h) == ["above", "dragged", "below"]


def test_saturated_neighbour_response_and_real_tiebreak(tmp_path):
    h = ui_harness(tmp_path)
    h.sessions["below"].last_chat_ts = h.now
    result = h.manager.reorder_session("dragged", above=None, below="below")
    assert result["applied"] is False
    assert project_rows(h).index("below") < project_rows(h).index("dragged")
    assert h.saves == [True]
