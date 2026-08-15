"""Regression coverage for the desktop session-card hover action buttons.

The inline Edit/Duplicate/Delete group has two reported behaviors under test:

1. The buttons must be opaque paper-filled so session title/meta text does not
   bleed through them while revealed.
2. The group must reveal on hover and hide when the pointer leaves. The reveal
   condition must not include plain ``:focus-within``: a mouse click leaves DOM
   focus on the clicked button, which would pin the group visible after the
   pointer leaves. Keyboard accessibility is preserved through
   ``:has(... :focus-visible)``, which mouse clicks never trigger.

Per project policy this parses the stylesheet into per-selector declarations
instead of asserting raw source text.
"""

from __future__ import annotations

import re
from pathlib import Path


APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"

HOVER_MEDIA_RE = re.compile(r"@media\s*\(hover:\s*hover\)\s*and\s*\(pointer:\s*fine\)\s*and\s*\(min-width:\s*881px\)")


def _strip_css_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)


def _matching_brace(css: str, opening_brace: int) -> int:
    depth = 1
    for index in range(opening_brace + 1, len(css)):
        if css[index] == "{":
            depth += 1
        elif css[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    raise AssertionError("unclosed CSS block")


def _declarations(body: str) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for declaration in body.split(";"):
        if ":" not in declaration:
            continue
        property_name, value = declaration.split(":", 1)
        property_name = property_name.strip()
        value = value.strip()
        if property_name and value:
            declarations[property_name] = value
    return declarations


def _hover_media_rules() -> list[tuple[list[str], dict[str, str]]]:
    """Return (selectors, declarations) pairs inside the desktop hover media block."""

    css = _strip_css_comments(APP_CSS.read_text(encoding="utf-8"))
    match = HOVER_MEDIA_RE.search(css)
    if not match:
        raise AssertionError("missing desktop hover media block")
    opening_brace = css.find("{", match.end() - 1)
    closing_brace = _matching_brace(css, opening_brace)
    block = css[opening_brace + 1 : closing_brace]

    rules: list[tuple[list[str], dict[str, str]]] = []
    cursor = 0
    while cursor < len(block):
        inner_open = block.find("{", cursor)
        if inner_open < 0:
            break
        header = block[cursor:inner_open].strip()
        inner_close = _matching_brace(block, inner_open)
        body = block[inner_open + 1 : inner_close]
        cursor = inner_close + 1
        if not header or header.startswith("@"):
            continue
        rules.append(([selector.strip() for selector in header.split(",")], _declarations(body)))
    return rules


def _rules_revealing_inline_actions() -> list[list[str]]:
    """Selector lists that turn the inline action group visible."""

    revealing = []
    for selectors, declarations in _hover_media_rules():
        if declarations.get("opacity") == "1" and declarations.get("visibility") == "visible":
            if any("sessionActionsInline" in selector for selector in selectors):
                revealing.append(selectors)
    if not revealing:
        raise AssertionError("no rule reveals .sessionActionsInline")
    return revealing


def test_hover_action_buttons_are_paper_filled() -> None:
    for selectors, declarations in _hover_media_rules():
        if ".session.desktop .sessionActionsInline .icon-btn" in selectors:
            assert declarations.get("background") == "var(--paper)"
            return
    raise AssertionError("missing rule for .session.desktop .sessionActionsInline .icon-btn")


def test_hover_reveal_activates_on_hover() -> None:
    for selectors in _rules_revealing_inline_actions():
        assert any(
            selector.startswith(".session.desktop:hover") for selector in selectors
        ), f"hover reveal missing :hover activation: {selectors}"


def test_hover_reveal_ignores_mouse_click_focus() -> None:
    """A mouse click leaves DOM focus on the button; the reveal must not track it.

    Only keyboard focus (``:focus-visible``) may keep the group pinned, so the
    allowed non-hover activation is ``:has(.sessionActionsInline :focus-visible)``.
    """

    for selectors in _rules_revealing_inline_actions():
        for selector in selectors:
            assert ":focus-within" not in selector, (
                f"reveal selector {selector!r} uses :focus-within, which stays active "
                "after a mouse click and pins the buttons after the pointer leaves"
            )
        assert any(
            ":focus-visible" in selector and ":has(" in selector for selector in selectors
        ), f"keyboard focus accessibility lost; expected :has(... :focus-visible) in {selectors}"
