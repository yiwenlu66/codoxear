"""Pin the staged-attachment chip's geometry contract on its known selectors.

The chip family is a small, flat selector set, so these assertions read exact
selectors through the shared css_tokens helpers instead of reimplementing a
cascade. The selectors the contract is about, and the state each one matches:

  .stagedAttachmentChip                                 every chip
  .stagedAttachmentChip[data-pending]                   an in-flight chip
  .stagedAttachmentChip[data-pending]:not([data-failed])  in flight, not the recovery layout
  .stagedAttachmentChip[data-failed]                    the failure/recovery layout
  .stagedAttachmentRemove, .…Remove::after              its cancel control and invisible hit lane

Each test names the state it reads. scripts/docker_upload_pill_geometry_verify.sh
is authoritative for the real cascade and for hit testing; these assertions pin
the two declarations the defect came from — an in-flight `height: auto` that made
the pill 14px taller than the completed one, and a 44px *visual* cancel box left
on the in-flight state instead of the failure layout.
"""

from __future__ import annotations

import pytest

from css_tokens import APP_CSS, all_rules, base_tokens, declarations, parse_stylesheet, resolve, selector_text

# Every selector that reaches a plain in-flight chip or its cancel control. A
# rule declaring a box here is the defect: the chip must keep the base height,
# and the cancel control must keep the base 18x18 visual box.
IN_FLIGHT_CHIP_SELECTORS = (
    ".stagedAttachmentChip[data-pending]",
    ".stagedAttachmentChip[data-pending]:not([data-failed])",
)
IN_FLIGHT_CANCEL_SELECTORS = (
    ".stagedAttachmentChip[data-pending] .stagedAttachmentRemove",
    ".stagedAttachmentChip[data-pending]:not([data-failed]) .stagedAttachmentRemove",
)
IN_FLIGHT_LANE_SELECTOR = ".stagedAttachmentChip[data-pending]:not([data-failed]) .stagedAttachmentRemove::after"
FAILED_CANCEL_SELECTOR = ".stagedAttachmentChip[data-failed] .stagedAttachmentRemove"
FAILED_LANE_SELECTOR = ".stagedAttachmentChip[data-failed] .stagedAttachmentRemove::after"
BOX_PROPERTIES = ("height", "min-height", "width", "min-width")


@pytest.fixture(scope="module")
def styles() -> dict[str, dict[str, str]]:
    """Declared properties per exact selector, in source order."""
    table: dict[str, dict[str, str]] = {}
    for rule in all_rules(parse_stylesheet(APP_CSS)):
        table.setdefault(selector_text(rule), {}).update(declarations(rule))
    return table


def px(value: str, tokens: dict[str, str]) -> float:
    resolved = resolve(value, tokens).strip()
    if resolved in {"0", "0.0"}:
        return 0.0  # a unitless zero is a valid length
    assert resolved.endswith("px"), f"expected a px length, got {resolved!r}"
    return float(resolved[:-2])


def insets(value: str, tokens: dict[str, str]) -> list[float]:
    """Inset shorthand expanded to (top, right, bottom, left)."""
    parts = [px(part, tokens) for part in resolve(value, tokens).split()]
    assert 1 <= len(parts) <= 4, f"not an inset shorthand: {value!r}"
    if len(parts) == 1:
        return parts * 4
    if len(parts) == 2:
        return [parts[0], parts[1], parts[0], parts[1]]
    if len(parts) == 3:
        return [parts[0], parts[1], parts[2], parts[1]]
    return parts


def test_chip_height_is_the_shared_control_token(styles) -> None:
    table = base_tokens()
    assert styles[".stagedAttachmentChip"]["height"] == "var(--ctl-chrome)"
    assert px(styles[".stagedAttachmentChip"]["height"], table) == 32


def test_no_rule_gives_an_in_flight_chip_its_own_box(styles) -> None:
    for selector in IN_FLIGHT_CHIP_SELECTORS:
        declared = styles.get(selector, {})
        assert not set(BOX_PROPERTIES) & set(declared), f"{selector} must not size the chip: {declared}"
    # The recovery layout is the deliberate exception: it grows to carry its
    # wrapped reason plus Retry.
    failed = styles[".stagedAttachmentChip[data-failed]"]
    assert failed["height"] == "auto" and failed["flex-wrap"] == "wrap"
    assert px(failed["min-height"], base_tokens()) == 32


def test_the_44px_visual_cancel_box_stays_in_the_recovery_layout(styles) -> None:
    table = base_tokens()
    failed = styles[FAILED_CANCEL_SELECTOR]
    assert px(failed["width"], table) == 44 and px(failed["min-height"], table) == 44
    for selector in IN_FLIGHT_CANCEL_SELECTORS:
        declared = styles.get(selector, {})
        assert not set(BOX_PROPERTIES) & set(declared), f"{selector} must not resize the cancel control: {declared}"


def test_an_in_flight_cancel_reaches_44px_through_its_hit_lane(styles) -> None:
    table = base_tokens()
    base = styles[".stagedAttachmentRemove"]
    assert px(base["width"], table) == 18 and px(base["height"], table) == 18
    # In flight: 18px of box plus the lane's outset, measured over both axes.
    lane = insets(styles[IN_FLIGHT_LANE_SELECTOR]["inset"], table)
    top, right, bottom, left = lane
    assert 18 - right - left >= 44
    assert 18 - top - bottom >= 44
    # The lane reaches upward: the composer's entry row below the tray owns the
    # pixels its own surface covers, and the browser verifier proves the lane
    # still ends above it while staying inside the composer band.
    assert -top > -bottom
    # The recovery layout needs no lane; its own 44px box is the target.
    assert insets(styles[FAILED_LANE_SELECTOR]["inset"], table) == [0, 0, 0, 0]
