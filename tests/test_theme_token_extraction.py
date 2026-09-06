"""Appearance-neutral token extraction: paper light computes exactly as before.

app.css expresses geometry, type, focus, icon muting, and elevation through
``:root`` tokens so theme families can retune them. These tests parse the
stylesheet and resolve every declaration against the paper-light token table,
pinning the values the literals had before extraction: every radius is 0,
chrome text is the generic sans-serif stack, focus rings are ink, backend
icons are muted to ink, and nothing casts a shadow.
"""

from __future__ import annotations

import re

import pytest

from css_tokens import APP_CSS, all_rules, base_tokens, declarations, parse_stylesheet, resolve, selector_text

PAPER_INK = "#2f2b26"
PAPER_DANGER = "#b91c1c"
PAPER_ICON_FILTER = "grayscale(1) brightness(0) opacity(0.62)"

# Component roles behind each radius token. A selector listed here must resolve
# its border-radius through that token so a family retunes it as one unit.
RADIUS_ROLE_SELECTORS = {
    "--radius-control": ["button", ".icon-btn", ".choiceChip", ".md code", ".code-copy-btn", ".filePickerBtn", ".jumpBtn"],
    "--radius-card": [".session", ".sessionContent", ".toast", ".fileViewer", ".queueViewer", ".md pre", ".md table", ".login", ".sendChoice"],
    "--radius-bubble": [".msg"],
    "--radius-pill": [".badge", ".day-sep", ".attachBadge", ".stagedAttachmentChip", ".sessionGroupCount"],
    "--radius-dot": [".stateDot", ".ownDot", ".typingDot", ".subagentActivitySquare"],
}


def _rules_for(selector: str):
    matches = []
    for rule in all_rules(parse_stylesheet(APP_CSS)):
        if selector in [part.strip() for part in selector_text(rule).split(",")]:
            matches.append(declarations(rule))
    assert matches, f"missing CSS rule for {selector}"
    return matches


def test_paper_light_tokens_carry_the_pre_extraction_literals() -> None:
    tokens = base_tokens()
    for name in ("--radius-control", "--radius-card", "--radius-bubble", "--radius-pill", "--radius-dot"):
        assert resolve(tokens[name], tokens) == "0"
    assert resolve(tokens["--font-ui"], tokens) == "sans-serif"
    assert resolve(tokens["--font-prose"], tokens) == "sans-serif"
    assert resolve(tokens["--focus-ring"], tokens) == PAPER_INK
    assert resolve(tokens["--icon-muted-filter"], tokens) == PAPER_ICON_FILTER
    assert resolve(tokens["--shadow-pop"], tokens) == "none"


# The nested-fill rule (.sessionContent) expresses its concentric radius as
# max(0px, calc(<card> - 1px)); evaluate the arithmetic before comparing.
CONCENTRIC_RADIUS = re.compile(r"max\(0px, calc\((\d+(?:\.\d+)?)(?:px)? - 1px\)\)")


def resolve_radius_square(value: str, tokens: dict[str, str]) -> str:
    resolved = resolve(value, tokens)
    match = CONCENTRIC_RADIUS.fullmatch(resolved)
    if match:
        return str(max(0.0, float(match.group(1)) - 1)).removesuffix(".0")
    return resolved


def test_every_border_radius_in_base_resolves_to_square() -> None:
    tokens = base_tokens()
    seen = 0
    for rule in all_rules(parse_stylesheet(APP_CSS)):
        # Settings swatches are miniatures of the other families and carry
        # their geometry on purpose.
        if selector_text(rule).startswith(".themeSwatch"):
            continue
        for name, value in declarations(rule).items():
            if name != "border-radius":
                continue
            seen += 1
            resolved = resolve_radius_square(value, tokens)
            assert set(resolved.split()) == {"0"}, f"{selector_text(rule)} resolves border-radius to {resolved!r}"
    assert seen >= 81


@pytest.mark.parametrize(("token", "selectors"), sorted(RADIUS_ROLE_SELECTORS.items()))
def test_radius_roles_retune_through_their_token(token: str, selectors: list[str]) -> None:
    # Resolving against a table where only this token is non-zero proves each
    # selector's radius is wired to that token and no other.
    probe = dict(base_tokens())
    probe[token] = "7px"
    for selector in selectors:
        radii = [resolve(rule["border-radius"], probe) for rule in _rules_for(selector) if "border-radius" in rule]
        assert radii, f"{selector} declares no border-radius"
        assert all("7px" in radius for radius in radii), f"{selector} does not follow {token}: {radii}"


def test_body_and_meta_text_follow_the_ui_font_token() -> None:
    tokens = base_tokens()
    probe = dict(tokens)
    probe["--font-ui"] = "Probe UI"
    for selector in ("body", ".sessionMetaLine", ".sidebarMetaLabel"):
        families = [resolve(rule["font-family"], probe) for rule in _rules_for(selector) if "font-family" in rule]
        assert families == ["Probe UI"], selector
        assert [resolve(rule["font-family"], tokens) for rule in _rules_for(selector) if "font-family" in rule] == ["sans-serif"]


def test_markdown_headings_follow_the_prose_font_token() -> None:
    tokens = base_tokens()
    probe = dict(tokens)
    probe["--font-prose"] = "Probe Serif"
    families = [resolve(rule["font-family"], probe) for rule in _rules_for(".md h1") if "font-family" in rule]
    assert families == ["Probe Serif"]
    assert [resolve(rule["font-family"], tokens) for rule in _rules_for(".md h1") if "font-family" in rule] == ["sans-serif"]


def test_backend_icons_are_muted_through_the_filter_token() -> None:
    tokens = base_tokens()
    probe = dict(tokens)
    probe["--icon-muted-filter"] = "invert(1)"
    for selector in (".sessionBackendStatusIcon", ".session.active .sessionBackendStatusIcon"):
        filters = [rule["filter"] for rule in _rules_for(selector) if "filter" in rule]
        assert [resolve(value, tokens) for value in filters] == [PAPER_ICON_FILTER], selector
        assert [resolve(value, probe) for value in filters] == ["invert(1)"], selector


def test_focus_outlines_resolve_to_ink_and_follow_the_focus_token() -> None:
    tokens = base_tokens()
    probe = dict(tokens)
    probe["--focus-ring"] = "#123456"
    focus_rules = 0
    for rule in all_rules(parse_stylesheet(APP_CSS)):
        selector = selector_text(rule)
        outline = declarations(rule).get("outline")
        if not outline or ":focus-visible" not in selector:
            continue
        focus_rules += 1
        resolved = resolve(outline, tokens)
        assert resolved in (f"2px solid {PAPER_INK}", f"2px solid {PAPER_DANGER}", "none"), f"{selector}: {resolved}"
        if resolved == f"2px solid {PAPER_INK}":
            assert resolve(outline, probe) == "2px solid #123456", selector
    assert focus_rules >= 8


def test_floating_surfaces_cast_no_shadow_on_paper_but_follow_the_elevation_token() -> None:
    tokens = base_tokens()
    probe = dict(tokens)
    probe["--shadow-pop"] = "0 1px 2px black"
    for selector in (".toast", ".fileViewer", ".queueViewer", ".formViewer", ".helpViewer", ".diagViewer", ".sendChoice", ".filePickerMenu", ".unattendedMenu", ".composer .modelPicker"):
        shadows = [rule["box-shadow"] for rule in _rules_for(selector) if "box-shadow" in rule]
        assert shadows, f"{selector} declares no box-shadow"
        assert {resolve(value, tokens) for value in shadows} == {"none"}, selector
        assert {resolve(value, probe) for value in shadows} == {"0 1px 2px black"}, selector


def test_base_stylesheet_keeps_colors_in_root_tokens() -> None:
    color_literal = re.compile(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(")
    for rule in all_rules(parse_stylesheet(APP_CSS)):
        if selector_text(rule) == ":root":
            continue
        for name, value in declarations(rule).items():
            # Custom properties are tokens by definition (swatch palettes
            # scope theirs to the swatch); ordinary properties must not carry
            # a color literal.
            if name.startswith("--"):
                continue
            assert not color_literal.search(value), f"{selector_text(rule)} {name}: {value}"
