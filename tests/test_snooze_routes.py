from dataclasses import replace
import json

import pytest

from codoxear.control_routes import handle_control_post_route
from codoxear.server import _match_session_route
from test_control_routes import Handler, _deps
from test_snooze_indefinite import ui_harness


def dispatch(h, action, body, *, sid="dragged", auth=True):
    responses = []

    class RouteHandler(Handler):
        def _unauthorized(self):
            super()._unauthorized()
            responses.append((401, {"error": "unauthorized"}))

    handler = RouteHandler(body)
    deps = _deps(responses, auth=auth)
    if not auth:
        deps = replace(deps, read_json_body=lambda *args: pytest.fail("unauthorized body read"))
    assert handle_control_post_route(
        handler, path=f"/api/sessions/{sid}/{action}", manager=h.manager,
        deps=deps, match_session_route=_match_session_route,
    )
    assert len(responses) == 1
    return responses[0]


def test_snooze_set_clear_and_disk_keep_other_fields(tmp_path):
    h = ui_harness(tmp_path)
    h.store.aliases["dragged"] = "Keep name"
    h.store.sidebar_meta = {
        "dragged": {"priority_offset": 0.3, "snooze_until": h.now + 100, "dependency_session_id": "above"},
        "below": {"priority_offset": -0.3},
    }
    assert dispatch(h, "snooze", {"indefinite": True}) == (200, {
        "ok": True, "snooze_until": None, "snoozed_indefinitely": True,
    })
    assert json.loads(h.store.paths.sidebar_meta.read_text()) == {
        "dragged": {"priority_offset": 0.3, "snoozed_indefinitely": True, "dependency_session_id": "above"},
        "below": {"priority_offset": -0.3},
    }
    assert h.saves == [True]
    assert dispatch(h, "snooze", {"indefinite": False}) == (200, {
        "ok": True, "snooze_until": None, "snoozed_indefinitely": False,
    })
    assert json.loads(h.store.paths.sidebar_meta.read_text()) == {
        "dragged": {"priority_offset": 0.3, "dependency_session_id": "above"},
        "below": {"priority_offset": -0.3},
    }
    assert h.store.aliases == {"dragged": "Keep name"}
    assert h.saves == [True, True]


def test_snooze_clear_also_clears_finite_timestamp(tmp_path):
    h = ui_harness(tmp_path)
    h.store.sidebar_meta["dragged"] = {"snooze_until": h.now + 50}
    assert dispatch(h, "snooze", {"indefinite": False})[0] == 200
    assert h.store.load_sidebar_meta() == {"dragged": {"priority_offset": 0.0}}


@pytest.mark.parametrize("action,body", [
    ("snooze", {"indefinite": True}), ("reorder", {"above": "above", "below": "below"}),
])
def test_narrow_routes_require_auth_and_return_unknown_session(tmp_path, action, body):
    h = ui_harness(tmp_path)
    assert dispatch(h, action, body, auth=False)[0] == 401
    assert dispatch(h, action, body, sid="missing") == (404, {"error": "unknown session"})
    assert h.saves == []
    assert h.store.sidebar_meta == {}


@pytest.mark.parametrize("body", [{}, {"indefinite": None}, {"indefinite": 1}, {"indefinite": "true"}, {"indefinite": []}])
def test_snooze_non_boolean_is_400(tmp_path, body):
    h = ui_harness(tmp_path)
    assert dispatch(h, "snooze", body) == (400, {"error": "indefinite must be a boolean"})
    assert h.saves == []


@pytest.mark.parametrize("body,error", [
    ({"above": None, "below": None}, "above or below required"),
    ({"above": "missing", "below": None}, "unknown neighbour"),
    ({"above": None, "below": "missing"}, "unknown neighbour"),
    ({"above": 3, "below": None}, "unknown neighbour"),
    ({"above": [], "below": "below"}, "unknown neighbour"),
])
def test_reorder_bad_neighbours_are_400(tmp_path, body, error):
    h = ui_harness(tmp_path)
    assert dispatch(h, "reorder", body) == (400, {"error": error})
    assert h.saves == []


def test_reorder_200_shape_and_persistence(tmp_path):
    h = ui_harness(tmp_path)
    status, result = dispatch(h, "reorder", {"above": "above", "below": "below"})
    assert status == 200
    assert result == {"ok": True, "priority_offset": 0.0747, "applied": True}
    assert h.store.load_sidebar_meta() == {"dragged": {"priority_offset": result["priority_offset"]}}
    assert h.saves == [True]


def test_edit_accepts_flag_and_absence_resets_it(tmp_path):
    h = ui_harness(tmp_path)
    body = {"name": "New name", "priority_offset": 0.1, "snooze_until": None, "dependency_session_id": None}
    status, result = dispatch(h, "edit", {**body, "snoozed_indefinitely": True})
    assert status == 200
    assert result["snoozed_indefinitely"] is True
    assert h.store.load_sidebar_meta() == {"dragged": {"priority_offset": 0.1, "snoozed_indefinitely": True}}
    assert h.store.aliases == {"dragged": "New name"}
    status, result = dispatch(h, "edit", body)
    assert status == 200
    assert result["snoozed_indefinitely"] is False
    assert h.store.load_sidebar_meta() == {"dragged": {"priority_offset": 0.1}}


def test_edit_rejects_non_bool_and_conflicting_snooze(tmp_path):
    h = ui_harness(tmp_path)
    body = {"name": "New name", "priority_offset": 0.1, "snooze_until": None, "dependency_session_id": None}
    assert dispatch(h, "edit", {**body, "snoozed_indefinitely": 1}) == (
        400, {"error": "snoozed_indefinitely must be a boolean"},
    )
    assert dispatch(h, "edit", {**body, "snoozed_indefinitely": True, "snooze_until": h.now + 10}) == (
        400, {"error": "indefinite snooze cannot have snooze_until"},
    )
    assert h.saves == []
    assert h.store.aliases == {}


@pytest.mark.parametrize("clamp", [False, True])
def test_reorder_reports_saturation_and_persists_nearest_offset(tmp_path, clamp):
    h = ui_harness(tmp_path)
    h.sessions["below"].last_chat_ts = h.now
    if clamp:
        h.sessions["dragged"].start_ts = h.now - 1e9
    status, result = dispatch(h, "reorder", {"above": None, "below": "below"})
    assert status == 200
    assert result["ok"] is True
    assert result["applied"] is False
    assert -1.0 <= result["priority_offset"] <= 1.0
    if clamp:
        assert result["priority_offset"] == 1.0
    assert h.store.load_sidebar_meta() == {"dragged": {"priority_offset": result["priority_offset"]}}
    assert h.saves == [True]
