"""Computed CSS contract for transcript subagent activity bubbles.

The activity detail row participates in the bubble's intrinsic inline size. A
100%-wide flex child makes even short content stretch the bubble to its maximum;
nowrap then hides long telemetry instead of wrapping it. These checks parse the
cascade and assert the layout mechanism rather than matching stylesheet source.
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
        assert style["grid-template-columns"] == "auto minmax(0, auto)"
        assert "width" not in style
        assert "max-width" not in style

    details = cascaded(".subagentDetails")
    assert details["grid-column"] == "1 / -1"
    assert details["min-width"] == "0"
    assert "width" not in details
    assert "flex" not in details
    assert "flex-basis" not in details


def test_subagent_detail_lines_wrap_instead_of_clipping() -> None:
    line = cascaded(".subagentDetailLine")
    assert line["white-space"] == "normal"
    assert line["overflow-wrap"] == "anywhere"
    assert line.get("overflow", "visible") == "visible"
    assert line.get("text-overflow", "clip") == "clip"
