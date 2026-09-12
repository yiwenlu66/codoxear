"""Computed CSS contract for transcript subagent activity bubbles.

The activity detail row participates in the bubble's intrinsic inline size. A
100%-wide flex child makes even short content stretch the bubble to its maximum;
nowrap then hides long telemetry instead of wrapping it. A spanning grid item
can also enlarge either intrinsic track, displacing the label even when the
marker-to-label gap itself remains correct. The dots and label therefore share
an independent inline-flex header while the detail row occupies its own grid
row. These checks parse the cascade and assert that layout mechanism rather than
matching stylesheet source.
"""

from __future__ import annotations

from css_tokens import APP_CSS, declarations, parse_stylesheet, selector_text, top_level_rules


def cascaded(selector: str) -> dict[str, str]:
    style: dict[str, str] = {}
    for rule in top_level_rules(parse_stylesheet(APP_CSS)):
        selectors = [part.strip() for part in selector_text(rule).split(",")]
        if selector in selectors:
            style.update(declarations(rule))
    assert style, f"no rule for {selector}"
    return style


def test_busy_and_idle_activity_bubbles_use_intrinsic_grid_width() -> None:
    message = cascaded(".msg")
    assert message["max-width"] not in {"none", "initial", "unset"}

    for selector in (".msg.typing", ".subagentActivity"):
        style = cascaded(selector)
        assert style["display"] == "inline-grid"
        assert style["grid-template-columns"] == "minmax(0, auto)"
        assert "width" not in style
        assert "max-width" not in style

    header = cascaded(".subagentActivityHeader")
    assert header["display"] == "inline-flex"
    assert header["justify-self"] == "start"
    assert header["gap"] == "var(--space-3)"
    assert header["max-width"] == "100%"

    details = cascaded(".subagentDetails")
    assert details["grid-column"] == "1 / -1"
    assert details["min-width"] == "0"
    assert "width" not in details
    assert "flex" not in details
    assert "flex-basis" not in details


def merged_cascade(*selectors: str) -> dict[str, str]:
    """Cascaded declarations for selectors applied in source order.

    Later (more specific or simply later) rules override earlier ones, and
    selectors with no rule contribute nothing.
    """
    style: dict[str, str] = {}
    rules = parse_stylesheet(APP_CSS)
    wanted = set(selectors)
    for rule in top_level_rules(rules):
        for part in selector_text(rule).split(","):
            if part.strip() in wanted:
                style.update(declarations(rule))
    return style


def test_activity_bubbles_share_one_internal_layout() -> None:
    """Busy and idle bubbles must be the same padded box with one left edge.

    The busy typing bubble and the idle activity bubble render the same
    header + child-lines content, so their internal padding comes from one
    token pair, child lines share the marker group's left edge (bubble
    content-left) in both states, and no per-state indent override exists.
    """
    busy = cascaded(".msg.typing")
    idle = cascaded(".subagentActivity")
    shared_padding = "var(--space-3) var(--space-4)"
    assert busy["padding"] == shared_padding
    assert idle["padding"] == shared_padding
    assert "row-gap" not in idle
    assert "row-gap" not in busy

    # Child lines sit at bubble content-left: no state-specific indent.
    details = merged_cascade(".subagentDetails", ".msg.typing .subagentDetails", ".subagentActivity .subagentDetails")
    assert "padding-left" not in details
    assert "padding" not in details


def test_child_lines_own_their_vertical_rhythm() -> None:
    """Details attach tighter to the summary than children sit from each other.

    The header-to-details gap (one shared rule, not per-bubble overrides)
    tightens the block under its summary, while a smaller flex gap separates
    distinct children so a wrapped continuation stays visually inside its row.
    """
    details = cascaded(".subagentDetails")
    assert details["margin-top"] == "var(--space-2)"
    assert details["display"] == "flex"
    assert details["flex-direction"] == "column"
    assert details["gap"] == "var(--space-1)"


def test_child_lines_use_the_ui_font_matching_their_summary() -> None:
    """Detail telemetry renders in the shared UI font, not the data mono face.

    User preference supersedes the mono-for-data rule inside these bubbles
    only: child lines share the summary's `--font-ui` family (inherited from
    the body by the summary labels) at their own smaller size, and no rule
    reintroduces a monospace family or a hardcoded font stack here.
    """
    details = cascaded(".subagentDetails")
    assert details["font"] == "var(--font-2xs)/1.4 var(--font-ui)"
    assert "font-family" not in details
    line = cascaded(".subagentDetailLine")
    assert "font" not in line
    assert "font-family" not in line
    for label in (".typingStats", ".subagentActivityText"):
        style = cascaded(label)
        assert "font-family" not in style
        assert "font" not in style


def test_subagent_detail_lines_wrap_instead_of_clipping() -> None:
    line = cascaded(".subagentDetailLine")
    assert line["white-space"] == "normal"
    assert line["overflow-wrap"] == "anywhere"
    assert line.get("overflow", "visible") == "visible"
    assert line.get("text-overflow", "clip") == "clip"
