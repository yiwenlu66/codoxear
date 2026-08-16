"""Pin dialog date/time controls to the dialog's control type scale.

The stylesheet is parsed and cascaded per selector; these tests never inspect
raw source text.
"""

from pathlib import Path
import re

import tinycss2
from tinycss2.ast import AtRule, Declaration, QualifiedRule


APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


def _media_query_matches(prelude: str, width: int) -> bool:
    """Evaluate app.css media forms for a fine-pointer viewport."""
    for query in prelude.split(","):
        query = query.strip()
        terms = [term.strip() for term in query.split(" and ")]
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


def _computed_style(width: int, selector: str) -> dict[str, str]:
    stylesheet = tinycss2.parse_stylesheet(
        APP_CSS.read_text(encoding="utf-8"),
        skip_comments=True,
        skip_whitespace=True,
    )
    computed: dict[str, str] = {}
    important: set[str] = set()
    for rule in _active_rules(stylesheet, width):
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


def test_custom_snooze_date_time_controls_use_dialog_control_type_scale():
    for selector in ('.formViewer input[type="date"]', '.formViewer input[type="time"]'):
        style = _computed_style(1280, selector)
        assert style["font-family"] == "inherit"
        assert style["font-size"] == "var(--font-lg)"


def test_custom_snooze_date_time_controls_keep_mobile_anti_zoom_size():
    for selector in ('input[type="date"]', 'input[type="time"]'):
        style = _computed_style(390, selector)
        assert style["font-size"] == "var(--font-xl)"
