"""Behavioral VM coverage for the sidebar's pure drag decisions and geometry."""

import json
import subprocess
import textwrap

import pytest

from frontend_module_loader import module_path


DRAG_JS = module_path("app_sidebar_drag.js")
CARDS = [
    {"id": "a", "top": 100, "height": 40},
    {"id": "b", "top": 160, "height": 40},
    {"id": "c", "top": 220, "height": 40},
]


def run_drag(expression: str) -> object:
    script = textwrap.dedent(
        f"""
        const vm = require("vm");
        const ctx = {{ window: {{}} }};
        vm.createContext(ctx);
        vm.runInContext({json.dumps(DRAG_JS.read_text(encoding="utf-8"))}, ctx);
        const result = vm.runInContext(
          {json.dumps('const drag = window.CodoxearSidebarDrag; ' + expression)}, ctx
        );
        process.stdout.write(JSON.stringify(result));
        """
    )
    proc = subprocess.run(
        ["node", "-e", script], check=True, capture_output=True, text=True, timeout=15
    )
    return json.loads(proc.stdout)


@pytest.mark.parametrize(
    "options, expected",
    [
        ({"sourceGroup": "now", "targetGroup": "now"}, "reorder"),
        ({"sourceGroup": "now", "targetGroup": "later"}, "snooze"),
        ({"sourceGroup": "now", "targetGroup": "waiting"}, "refuse"),
        ({"sourceGroup": "later", "targetGroup": "now"}, "wake"),
        ({"sourceGroup": "later", "targetGroup": "later"}, "none"),
        ({"sourceGroup": "later", "targetGroup": "waiting"}, "refuse"),
        ({"sourceGroup": "waiting", "targetGroup": "now"}, "none"),
        ({"sourceGroup": "unknown", "targetGroup": "now"}, "none"),
        ({"targetGroup": "now"}, "none"),
        ({"sourceGroup": "now"}, "none"),
        ({"sourceGroup": "later"}, "none"),
        ({"sourceGroup": "now", "targetGroup": "unknown"}, "none"),
        ({}, "none"),
    ],
)
def test_drop_kind_gesture_map(options: dict, expected: str) -> None:
    assert run_drag(f"drag.dropKind({json.dumps(options)})") == expected


@pytest.mark.parametrize(
    "cards, pointer, expected",
    [
        (CARDS, "50", 0),
        (CARDS, "300", 3),
        (CARDS, "170", 1),
        (CARDS, "190", 2),
        (CARDS, "120", 0),  # Exactly at a midpoint stays before the card.
        (CARDS, "150", 1),  # A gap has the same slot as its two neighbours.
        (CARDS[:1], "110", 0),
        (CARDS[:1], "130", 1),
        ([], "100", 0),
        (CARDS, "NaN", 3),
        (CARDS, "Infinity", 3),
        (CARDS, "-Infinity", 3),
    ],
)
def test_insertion_index(cards: list, pointer: str, expected: int) -> None:
    assert run_drag(f"drag.insertionIndex({json.dumps(cards)}, {pointer})") == expected


@pytest.mark.parametrize(
    "dragged, index, order, above, below",
    [
        ("d", 0, ["d", "a", "b", "c"], None, "a"),
        ("d", 1, ["a", "d", "b", "c"], "a", "b"),
        ("d", 3, ["a", "b", "c", "d"], "c", None),
        ("b", 0, ["b", "a", "c"], None, "a"),
        ("b", 2, ["a", "c", "b"], "c", None),
        ("d", -10, ["d", "a", "b", "c"], None, "a"),
        ("d", 100, ["a", "b", "c", "d"], "c", None),
    ],
)
def test_next_order_moves_one_id_and_resolves_neighbours(
    dragged: str, index: int, order: list, above: str | None, below: str | None
) -> None:
    assert run_drag(f'drag.nextOrder(["a", "b", "c"], {json.dumps(dragged)}, {index})') == {
        "order": order, "above": above, "below": below,
    }


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"pointerY": 190}, {"kind": "none"}),
        (
            {"pointerY": 50},
            {"kind": "reorder", "order": ["b", "a", "c"], "above": None, "below": "a", "index": 0},
        ),
        (
            {"sourceId": "d", "pointerY": 150},
            {"kind": "reorder", "order": ["a", "d", "b", "c"], "above": "a", "below": "b", "index": 1},
        ),
        (
            {"pointerY": 300},
            {"kind": "reorder", "order": ["a", "c", "b"], "above": "c", "below": None, "index": 2},
        ),
        (
            {"targetGroup": "later"},
            {"kind": "snooze", "order": None, "above": None, "below": None, "index": None},
        ),
        (
            {"sourceGroup": "later", "sourceId": "d", "pointerY": 210},
            {"kind": "wake", "order": ["a", "b", "d", "c"], "above": "b", "below": "c", "index": 2},
        ),
        (
            {"overHeader": True, "pointerY": 300},
            {"kind": "reorder", "order": ["b", "a", "c"], "above": None, "below": "a", "index": 0},
        ),
        (
            {"sourceGroup": "later", "sourceId": "d", "overHeader": True, "pointerY": 300},
            {"kind": "wake", "order": ["d", "a", "b", "c"], "above": None, "below": "a", "index": 0},
        ),
        ({"targetGroup": "waiting"}, {"kind": "none"}),
        ({"sourceGroup": "later", "targetGroup": "waiting"}, {"kind": "none"}),
        ({"sourceGroup": "later", "targetGroup": "later"}, {"kind": "none"}),
        ({"targetGroup": None}, {"kind": "none"}),
    ],
)
def test_drag_payload(overrides: dict, expected: dict) -> None:
    options = {
        "sourceGroup": "now", "targetGroup": "now", "sourceId": "b",
        "targetIds": ["a", "b", "c"], "cards": CARDS, "pointerY": 190,
    }
    options.update(overrides)
    assert run_drag(f"drag.dragPayload({json.dumps(options)})") == expected


@pytest.mark.parametrize("pointer", ["250", "148", "352", "NaN", "Infinity", "-Infinity"])
def test_auto_scroll_stays_still_outside_edge_bands(pointer: str) -> None:
    assert run_drag(f"drag.autoScrollStep(100, 400, {pointer})") == 0


@pytest.mark.parametrize(
    "pointer, direction",
    [(100, -1), (110, -1), (147, -1), (90, -1), (400, 1), (390, 1), (353, 1), (410, 1)],
)
def test_auto_scroll_moves_toward_nearest_edge_with_bounded_speed(pointer: int, direction: int) -> None:
    step = run_drag(f"drag.autoScrollStep(100, 400, {pointer})")
    assert 6 <= direction * step <= 14


def test_auto_scroll_accelerates_as_pointer_approaches_edge() -> None:
    assert run_drag("[drag.autoScrollStep(100, 400, 147), drag.autoScrollStep(100, 400, 100), "
                    "drag.autoScrollStep(100, 400, 353), drag.autoScrollStep(100, 400, 400)]") == [-6, -14, 6, 14]


@pytest.mark.parametrize(
    "cards, index, expected",
    [
        ([], 0, 84),  # Empty block uses headerBottom plus the line inset.
        (CARDS, 0, 96),
        (CARDS, 3, 264),
        (CARDS, 1, 150),
        (CARDS, 2, 210),
        (CARDS[:1], 0, 96),
        (CARDS[:1], 1, 144),
    ],
)
def test_insertion_line_y(cards: list, index: int, expected: int) -> None:
    assert run_drag(f"drag.insertionLineY({json.dumps(cards)}, {index}, 80)") == expected
