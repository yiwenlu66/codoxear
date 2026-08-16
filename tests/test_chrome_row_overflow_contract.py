"""Parsed stylesheet contract: chrome rows are never scroll containers.

Chrome .icon-btn hit areas are absolutely positioned ::after pseudo elements
with a negative inset. Any ancestor scroll container inherits that phantom
outset as scrollable overflow, which yields permanent scrollbars (and clips
painted focus/dirty rings) even when the row visually fits. This test pins
the invariant that .actions chrome rows carry no overflow machinery, and
that the one genuine chip scroller absorbs the hit-area outset in padding.
"""

from __future__ import annotations

from pathlib import Path

from tinycss2 import parse_declaration_list, parse_rule_list, parse_stylesheet, serialize


APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


def _declarations(tokens: list[object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for declaration in parse_declaration_list(tokens, skip_comments=True, skip_whitespace=True):
        if declaration.type != "declaration":
            continue
        value = serialize(declaration.value).strip()
        result[declaration.lower_name] = f"{value} !important" if declaration.important else value
    return result


def _rules(rules: list[object], *, media: str | None = None) -> list[tuple[str, dict[str, str], str | None]]:
    result: list[tuple[str, dict[str, str], str | None]] = []
    for rule in rules:
        if rule.type == "at-rule" and rule.at_keyword == "media" and rule.content is not None:
            result.extend(_rules(parse_rule_list(rule.content, skip_comments=True, skip_whitespace=True), media=serialize(rule.prelude).strip()))
        elif rule.type == "qualified-rule":
            declarations = _declarations(rule.content)
            for selector in serialize(rule.prelude).split(","):
                result.append((selector.strip(), declarations, media))
    return result


def _all_rules() -> list[tuple[str, dict[str, str], str | None]]:
    return _rules(parse_stylesheet(APP_CSS.read_text(encoding="utf-8"), skip_comments=True, skip_whitespace=True))


def _computed(rules: list[tuple[str, dict[str, str], str | None]], selector: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for candidate, declarations, media in rules:
        if candidate == selector and media is None:
            result.update(declarations)
    if not result:
        raise AssertionError(f"missing CSS rule for {selector}")
    return result


def test_actions_chrome_rows_are_not_scroll_containers() -> None:
    rules = _all_rules()
    actions = _computed(rules, ".actions")
    for prop in ("overflow", "overflow-x", "overflow-y"):
        assert prop not in actions, f".actions must not set {prop}: chrome rows fit or wrap, never scroll"
    assert not any(selector == ".actions::-webkit-scrollbar" for selector, _, _ in rules), (
        "the webkit scrollbar hack existed only to hide the phantom scrollbars of a scrollable .actions row"
    )
    # Crowded dialog headers wrap instead of scrolling.
    assert _computed(rules, ".queueHeader .actions")["flex-wrap"] == "wrap"
    assert _computed(rules, ".fileViewerHeader .actions")["flex-wrap"] == "wrap"


def test_choice_chips_scroller_absorbs_hit_area_outset() -> None:
    rules = _all_rules()
    chips = _computed(rules, ".choiceChips")
    # Chips genuinely scroll, but the 2px form-control hit-area outset must be
    # absorbed by padding so scrollbars appear only on real chip overflow.
    assert chips["overflow-x"] == "auto"
    assert chips["padding"] == "2px"
    hit_area = _computed(rules, ".formViewer .choiceChip::after")
    assert hit_area["inset"] == "-2px"
