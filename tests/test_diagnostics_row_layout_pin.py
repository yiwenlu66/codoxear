"""Pin the computed responsive layout of Details/diagnostics rows.

The test evaluates the stylesheet cascade with tinycss2 at representative viewport
widths.  It does not inspect raw source strings: only declarations from rules
whose media queries match the viewport participate in the resulting style.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pytest
import tinycss2

from css_tokens import base_tokens, resolve
from tinycss2.ast import AtRule, Declaration, QualifiedRule


APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"
PHONE_COLUMNS = "minmax(0, 1fr)"
DESKTOP_COLUMNS = "110px minmax(0, 1fr)"


def _media_matches_viewport(prelude: Iterable[object], width: int) -> bool:
    """Return whether any comma-separated width query applies at *width*."""

    for query in tinycss2.serialize(prelude).split(","):
        maximums = [int(value) for value in re.findall(r"max-width\s*:\s*(\d+)px", query)]
        minimums = [int(value) for value in re.findall(r"min-width\s*:\s*(\d+)px", query)]
        if (maximums or minimums) and all(width <= maximum for maximum in maximums) and all(
            width >= minimum for minimum in minimums
        ):
            return True
    return False


def _active_rules(rules: Iterable[object], width: int, *, active: bool = True) -> list[QualifiedRule]:
    """Flatten qualified rules that participate in the cascade at *width*."""

    matching: list[QualifiedRule] = []
    for rule in rules:
        if isinstance(rule, AtRule) and rule.lower_at_keyword == "media" and rule.content is not None:
            nested_rules = tinycss2.parse_rule_list(rule.content, skip_comments=True, skip_whitespace=True)
            matching.extend(
                _active_rules(
                    nested_rules,
                    width,
                    active=active and _media_matches_viewport(rule.prelude, width),
                )
            )
        elif active and isinstance(rule, QualifiedRule):
            matching.append(rule)
    return matching


def _computed_style(width: int, selector: str) -> dict[str, str]:
    stylesheet = tinycss2.parse_stylesheet(
        APP_CSS.read_text(encoding="utf-8"),
        skip_comments=True,
        skip_whitespace=True,
    )
    computed: dict[str, str] = {}
    tokens = base_tokens()

    for rule in _active_rules(stylesheet, width):
        selectors = [part.strip() for part in tinycss2.serialize(rule.prelude).split(",")]
        if selector not in selectors:
            continue
        declarations = tinycss2.parse_declaration_list(rule.content, skip_comments=True, skip_whitespace=True)
        for declaration in declarations:
            if isinstance(declaration, Declaration):
                computed[declaration.lower_name] = resolve(tinycss2.serialize(declaration.value).strip(), tokens)

    if not computed:
        raise AssertionError(f"missing CSS rule for {selector}")
    return computed


@pytest.mark.parametrize(
    ("width", "columns"),
    [
        # minmax(0, 1fr) is one flexible track: labels and values occupy
        # consecutive grid rows while retaining a zero minimum for long values.
        (390, PHONE_COLUMNS),
        (880, PHONE_COLUMNS),
        (881, DESKTOP_COLUMNS),
        (1280, DESKTOP_COLUMNS),
    ],
)
def test_diagnostics_row_grid_layout_is_pinned_at_representative_viewports(
    width: int,
    columns: str,
) -> None:
    style = _computed_style(width, ".detailsRow")

    assert style["display"] == "grid"
    assert style["grid-template-columns"] == columns


def test_diagnostics_row_desktop_layout_has_no_later_width_override() -> None:
    assert _computed_style(881, ".detailsRow") == _computed_style(1280, ".detailsRow")


@pytest.mark.parametrize("width", [390, 880, 1280])
def test_diagnostics_row_components_preserve_flat_paper_treatment(width: int) -> None:
    for selector in (".detailsGrid", ".detailsRow", ".detailsLabel", ".detailsValue"):
        style = _computed_style(width, selector)
        assert style.get("border-radius", "0") == "0"
        assert style.get("box-shadow", "none") == "none"
        assert style.get("backdrop-filter", "none") == "none"


def test_diagnostics_label_lifts_to_its_own_drawer_grid_row() -> None:
    assert _computed_style(390, ".detailsLabel")["padding-top"] == "0"
    assert _computed_style(880, ".detailsLabel")["padding-top"] == "0"
    assert _computed_style(881, ".detailsLabel")["padding-top"] == "2px"
