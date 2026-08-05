"""Regression coverage for the responsive Details/diagnostics row layout.

This test parses the stylesheet into active declarations at a viewport width instead
of asserting source text. It deliberately checks the rendered CSS cascade for the
three selectors that make up a diagnostics row.
"""

from __future__ import annotations

import re
from pathlib import Path


APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


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


def _rules_at_width(css: str, width: int) -> list[tuple[str, dict[str, str]]]:
    """Return stylesheet rules whose media conditions match *width*.

    The application stylesheet uses ordinary rules plus nested ``@media`` blocks;
    this small structural parser preserves source order, which is the CSS cascade
    relevant to the diagnostics selectors.
    """

    def walk(block: str, active: bool = True) -> list[tuple[str, dict[str, str]]]:
        rules: list[tuple[str, dict[str, str]]] = []
        cursor = 0
        while cursor < len(block):
            opening_brace = block.find("{", cursor)
            if opening_brace < 0:
                break
            header = block[cursor:opening_brace].strip()
            closing_brace = _matching_brace(block, opening_brace)
            body = block[opening_brace + 1 : closing_brace]
            cursor = closing_brace + 1
            if not header:
                continue
            if header.startswith("@media"):
                max_widths = [int(value) for value in re.findall(r"max-width\s*:\s*(\d+)px", header)]
                min_widths = [int(value) for value in re.findall(r"min-width\s*:\s*(\d+)px", header)]
                media_active = all(width <= value for value in max_widths) and all(width >= value for value in min_widths)
                rules.extend(walk(body, active and media_active))
                continue
            if header.startswith("@"):
                continue
            if active:
                for selector in header.split(","):
                    rules.append((selector.strip(), _declarations(body)))
        return rules

    return walk(_strip_css_comments(css))


def _computed_rule(width: int, selector: str) -> dict[str, str]:
    computed: dict[str, str] = {}
    for rule_selector, declarations in _rules_at_width(APP_CSS.read_text(encoding="utf-8"), width):
        if rule_selector == selector:
            computed.update(declarations)
    if not computed:
        raise AssertionError(f"missing CSS rule for {selector}")
    return computed


def test_diagnostics_rows_stack_through_sidebar_layout_width() -> None:
    phone = _computed_rule(390, ".detailsRow")
    phone_layout = _computed_rule(521, ".detailsRow")
    drawer_layout = _computed_rule(880, ".detailsRow")
    desktop = _computed_rule(881, ".detailsRow")

    assert phone["display"] == "grid"
    for style in (phone, phone_layout, drawer_layout):
        assert style["grid-template-columns"] == "minmax(0, 1fr)"
        assert style["gap"] == "var(--space-1)"
    assert desktop["grid-template-columns"] == "110px minmax(0, 1fr)"
    assert desktop["gap"] == "var(--space-5)"


def test_diagnostics_drawer_layout_preserves_row_treatment() -> None:
    drawer_row = _computed_rule(880, ".detailsRow")
    desktop_row = _computed_rule(881, ".detailsRow")
    drawer_label = _computed_rule(880, ".detailsLabel")
    desktop_label = _computed_rule(881, ".detailsLabel")

    assert {name for name in drawer_row if drawer_row[name] != desktop_row.get(name)} == {
        "grid-template-columns",
        "gap",
    }
    assert drawer_row["padding"] == desktop_row["padding"]
    assert drawer_row["border-bottom"] == desktop_row["border-bottom"]
    assert drawer_label["color"] == desktop_label["color"]
    assert drawer_label["font-size"] == desktop_label["font-size"]
    assert drawer_label["padding-top"] == "0"
    assert desktop_label["padding-top"] == "2px"
