"""Pin the role-based typography contract for shared form dialogs.

The stylesheet is parsed and cascaded per selector; these tests never inspect
raw source text.
"""

from pathlib import Path
import re

import tinycss2
from tinycss2.ast import AtRule, Declaration, QualifiedRule


APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"
ENTRY_SELECTORS = (
    'input[type="text"]',
    'input[type="password"]',
    'input[type="search"]',
    'input[type="date"]',
    'input[type="time"]',
    'input[type="datetime-local"]',
    'input[type="number"]',
    "select",
    "textarea",
)


def _media_query_matches(prelude: str, width: int) -> bool:
    """Evaluate app.css media forms for a fine-pointer viewport."""
    for query in prelude.split(","):
        terms = [term.strip() for term in query.strip().split(" and ")]
        matches = True
        for term in terms:
            max_width = re.fullmatch(r"\(\s*max-width:\s*(\d+)px\s*\)", term)
            min_width = re.fullmatch(r"\(\s*min-width:\s*(\d+)px\s*\)", term)
            if max_width:
                matches = matches and width <= int(max_width.group(1))
            elif min_width:
                matches = matches and width >= int(min_width.group(1))
            elif term in {"(hover: hover)", "(pointer: fine)"}:
                matches = matches and True
            elif term == "(pointer: coarse)" or term.startswith("(prefers-"):
                matches = False
            else:
                raise AssertionError(f"unsupported media query in test harness: {query}")
        if matches:
            return True
    return False


def _active_rules(nodes, width: int):
    for rule in nodes:
        if isinstance(rule, QualifiedRule):
            yield rule
        elif isinstance(rule, AtRule) and rule.lower_at_keyword == "media":
            prelude = tinycss2.serialize(rule.prelude).strip()
            if _media_query_matches(prelude, width):
                nested = tinycss2.parse_rule_list(
                    rule.content,
                    skip_comments=True,
                    skip_whitespace=True,
                )
                yield from _active_rules(nested, width)


def _stylesheet():
    return tinycss2.parse_stylesheet(
        APP_CSS.read_text(encoding="utf-8"),
        skip_comments=True,
        skip_whitespace=True,
    )


def _computed_style(width: int, selector: str) -> dict[str, str]:
    computed: dict[str, str] = {}
    important: set[str] = set()
    for rule in _active_rules(_stylesheet(), width):
        selectors = [part.strip() for part in tinycss2.serialize(rule.prelude).split(",")]
        if selector not in selectors:
            continue
        declarations = tinycss2.parse_declaration_list(
            rule.content,
            skip_comments=True,
            skip_whitespace=True,
        )
        for declaration in declarations:
            if not isinstance(declaration, Declaration):
                continue
            name = declaration.lower_name
            if declaration.important or name not in important:
                computed[name] = tinycss2.serialize(declaration.value).strip()
            if declaration.important:
                important.add(name)
    if not computed:
        raise AssertionError(f"missing CSS rule for {selector} at width {width}")
    return computed


def _font_size_rules():
    for rule in _active_rules(_stylesheet(), 1280):
        selectors = [part.strip() for part in tinycss2.serialize(rule.prelude).split(",")]
        declarations = tinycss2.parse_declaration_list(
            rule.content,
            skip_comments=True,
            skip_whitespace=True,
        )
        for declaration in declarations:
            if isinstance(declaration, Declaration) and declaration.lower_name == "font-size":
                yield selectors, tinycss2.serialize(declaration.value).strip()


def test_entry_controls_share_the_desktop_value_scale():
    for selector in ENTRY_SELECTORS:
        style = _computed_style(1280, selector)
        assert style["font-family"] == "inherit"
        assert style["font-size"] == "var(--font-lg)"


def test_entry_controls_share_the_mobile_anti_zoom_floor():
    for selector in ENTRY_SELECTORS:
        style = _computed_style(390, selector)
        assert style["font-size"] == "var(--font-xl)"


def test_label_and_action_plane_uses_one_token():
    for selector in ("button", ".fieldLabel", ".choiceChip", ".checkField", ".icon-btn.text-btn"):
        assert _computed_style(1280, selector)["font-size"] == "var(--font-md)"


def test_value_and_title_plane_uses_one_token():
    for selector in (".dialogPickerBtn", ".pickerButtonPrimary", ".queueHeader .title"):
        assert _computed_style(1280, selector)["font-size"] == "var(--font-lg)"


def test_dialog_meta_plane_uses_one_token():
    for selector in (".fieldHint", ".pickerButtonSecondary"):
        assert _computed_style(1280, selector)["font-size"] == "var(--font-sm)"
    assert _computed_style(1280, ".rangeValue")["font"] == "var(--font-sm)/1.2 var(--font-mono)"


def test_form_containers_do_not_own_entry_typography():
    forbidden_prefixes = (
        ".formViewer input",
        ".formViewer textarea",
        ".unattendedMenu input",
        ".unattendedMenu textarea",
    )
    offenders = []
    for selectors, value in _font_size_rules():
        for selector in selectors:
            if selector.startswith(forbidden_prefixes) and "checkbox" not in selector:
                offenders.append((selector, value))
    assert offenders == []
