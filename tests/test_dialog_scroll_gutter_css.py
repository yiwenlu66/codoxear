"""One gutter mechanism for every dialog scroll body.

A scroll container clips ink overflow at its padding box. Focus outlines and
the selected-swatch ring paint ``outline-width + outline-offset`` outside a
control's border box, so an edge-to-edge control inside a bare scroll body
loses the outer slice of its ring; overlay scrollbars (iOS, macOS) paint over
content that reaches the right edge. Every dialog scroll body therefore
carries one shared treatment: padding on every side at least as large as the
ring outset, a wider right-side lane for overlay scrollbars, and a stable
scrollbar gutter so classic scrollbars do not reflow the body when it starts
to scroll. Grids inside those bodies no longer need their own outset padding.

The stylesheet is parsed and cascaded per selector; nothing here inspects raw
source text.
"""

from __future__ import annotations

import re

from css_tokens import APP_CSS, base_tokens, declarations, parse_stylesheet, resolve, selector_text, top_level_rules

PX = re.compile(r"^(-?\d+(?:\.\d+)?)px$")

# Every scroll body that lives inside a dialog and holds edge-to-edge
# controls or copy: Settings/New session/Edit (.formBody), Help (.helpBody),
# and the queue viewer (.queueList).
DIALOG_SCROLL_BODIES = (".formBody", ".helpBody", ".queueList")


def px(value: str) -> float:
    match = PX.match(value.strip())
    assert match, f"expected a px length, got {value!r}"
    return float(match.group(1))


def cascaded(selector: str) -> dict[str, str]:
    """Declarations from every top-level rule whose selector list names ``selector``."""
    result: dict[str, str] = {}
    for rule in top_level_rules(parse_stylesheet(APP_CSS)):
        if selector in [part.strip() for part in selector_text(rule).split(",")]:
            result.update(declarations(rule))
    assert result, f"no rule for {selector}"
    return result


def box_padding(style: dict[str, str], tokens: dict[str, str]) -> dict[str, float]:
    parts = [px(part) for part in resolve(style["padding"], tokens).split()]
    top = parts[0]
    right = parts[1] if len(parts) > 1 else top
    bottom = parts[2] if len(parts) > 2 else top
    left = parts[3] if len(parts) > 3 else right
    return {"top": top, "right": right, "bottom": bottom, "left": left}


def focus_ring_outset(tokens: dict[str, str]) -> float:
    ring = cascaded("button:focus-visible")
    return px(resolve(ring["outline"].split()[0], tokens)) + px(resolve(ring["outline-offset"], tokens))


def test_every_dialog_scroll_body_absorbs_the_focus_ring_on_all_sides() -> None:
    tokens = base_tokens()
    outset = focus_ring_outset(tokens)
    assert outset >= 3
    for selector in DIALOG_SCROLL_BODIES:
        style = cascaded(selector)
        overflow = style.get("overflow-y") or style.get("overflow")
        assert overflow in {"auto", "scroll"}, f"{selector} is not a scroll container"
        padding = box_padding(style, tokens)
        for side, value in padding.items():
            assert value >= outset, f"{selector} {side} padding {value}px clips a {outset}px focus ring"


def test_every_dialog_scroll_body_leaves_a_lane_for_overlay_scrollbars() -> None:
    tokens = base_tokens()
    lane = px(resolve("var(--space-3)", tokens))
    for selector in DIALOG_SCROLL_BODIES:
        style = cascaded(selector)
        assert box_padding(style, tokens)["right"] >= lane, f"{selector} content runs under an overlay scrollbar"
        assert style["scrollbar-gutter"] == "stable", f"{selector} reflows when a classic scrollbar appears"


def test_the_swatch_grid_relies_on_the_body_gutter_instead_of_its_own_outset() -> None:
    tokens = base_tokens()
    active = cascaded(".themeSwatch.active")
    ring = px(resolve(active["outline"].split()[0], tokens)) + px(resolve(active["outline-offset"], tokens))
    assert ring <= min(box_padding(cascaded(".formBody"), tokens).values())
    assert "padding" not in cascaded(".themeSwatches")


def test_the_details_table_rows_carry_their_own_inset() -> None:
    # The Details dialog scrolls a bordered table rather than a bare body;
    # its rows inset their content past the overlay-scrollbar lane, and a
    # body-level gutter would detach the row separators from the border.
    tokens = base_tokens()
    grid = cascaded(".diagViewer .detailsGrid")
    assert grid["overflow-y"] == "auto"
    assert "padding" not in grid
    row = box_padding(cascaded(".detailsRow"), tokens)
    assert min(row["left"], row["right"]) >= px(resolve("var(--space-3)", tokens))
