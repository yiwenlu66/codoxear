"""Behavioral CSS cascade coverage for responsive diagnostics rows.

The stylesheet is parsed with tinycss2 and evaluated at representative viewport
widths.  The tests model only width media features; pointer media features do
not change the diagnostics declarations under test.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pytest
import tinycss2
from tinycss2.ast import AtRule, Declaration, QualifiedRule


APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"
TWO_COLUMN = "110px minmax(0, 1fr)"
SINGLE_COLUMN = "minmax(0, 1fr)"


def _media_matches_width(prelude: Iterable[object], width: int) -> bool:
    """Evaluate the width clauses in an @media rule for a viewport width."""

    for query in tinycss2.serialize(prelude).split(","):
        max_widths = [int(value) for value in re.findall(r"max-width\s*:\s*(\d+)px", query)]
        min_widths = [int(value) for value in re.findall(r"min-width\s*:\s*(\d+)px", query)]
        if (max_widths or min_widths) and all(width <= value for value in max_widths) and all(
            width >= value for value in min_widths
        ):
            return True
    return False


def _rules_at_width(rules: Iterable[object], width: int, active: bool = True) -> list[QualifiedRule]:
    matching: list[QualifiedRule] = []
    for rule in rules:
        if isinstance(rule, AtRule) and rule.lower_at_keyword == "media" and rule.content is not None:
            nested_rules = tinycss2.parse_rule_list(rule.content, skip_comments=True, skip_whitespace=True)
            matching.extend(_rules_at_width(nested_rules, width, active and _media_matches_width(rule.prelude, width)))
        elif active and isinstance(rule, QualifiedRule):
            matching.append(rule)
    return matching


def _computed_style(width: int, selector: str) -> dict[str, str]:
    stylesheet = tinycss2.parse_stylesheet(APP_CSS.read_text(encoding="utf-8"), skip_comments=True, skip_whitespace=True)
    computed: dict[str, str] = {}

    for rule in _rules_at_width(stylesheet, width):
        selectors = [part.strip() for part in tinycss2.serialize(rule.prelude).split(",")]
        if selector not in selectors:
            continue
        declarations = tinycss2.parse_declaration_list(rule.content, skip_comments=True, skip_whitespace=True)
        for declaration in declarations:
            if isinstance(declaration, Declaration):
                computed[declaration.lower_name] = tinycss2.serialize(declaration.value).strip()

    if not computed:
        raise AssertionError(f"missing CSS rule for {selector}")
    return computed


@pytest.mark.parametrize(
    ("width", "columns"),
    [
        (390, SINGLE_COLUMN),
        (521, SINGLE_COLUMN),
        (880, TWO_COLUMN),
        (1280, TWO_COLUMN),
    ],
)
def test_diagnostics_details_row_layout_at_viewport(width: int, columns: str) -> None:
    details_row = _computed_style(width, ".detailsRow")

    assert details_row["display"] == "grid"
    # A one-column CSS grid places the label then value in source order: label above value.
    assert details_row["grid-template-columns"] == columns


@pytest.mark.parametrize("width", [390, 521, 880, 1280])
def test_diagnostics_layout_preserves_paper_geometry(width: int) -> None:
    for selector in (".detailsGrid", ".detailsRow", ".detailsLabel", ".detailsValue"):
        style = _computed_style(width, selector)
        assert style.get("box-shadow", "none") == "none"
        assert style.get("border-radius", "0") == "0"
