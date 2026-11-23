"""The source shortcut is compact while its touch lane avoids the cwd field."""
from css_tokens import APP_CSS, base_tokens, declarations, parse_stylesheet, resolve, selector_text, top_level_rules


def style(selector):
    result = {}
    for rule in top_level_rules(parse_stylesheet(APP_CSS)):
        if selector in [part.strip() for part in selector_text(rule).split(",")]:
            result.update(declarations(rule))
    assert result
    return result


def test_source_shortcut_hit_lane_does_not_expand_row_or_cover_input():
    link = style(".sourceWorkspaceLink")
    lane = style(".sourceWorkspaceLink::after")
    assert link["min-height"] == "0"
    assert link["padding"] == "0"
    assert link["border"] == "0"
    assert link["background"] == "transparent"
    assert link["min-width"] == "0"
    assert lane["height"] == "44px"
    assert lane["left"] == lane["right"] == "0"
    # The field's gap separates the heading from the following input.
    gap = int(resolve(style(".field")["gap"], base_tokens()).removesuffix("px"))
    assert int(lane["bottom"].removesuffix("px")) > -gap
    assert style(".sourceWorkspaceHeading .fieldLabel")["min-width"] == "0"
