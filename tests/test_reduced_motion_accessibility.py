"""Behavioral stylesheet coverage for the global reduced-motion contract."""

from pathlib import Path

import tinycss2
from tinycss2.ast import AtRule, Declaration, QualifiedRule


APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


def test_reduced_motion_disables_all_animation_and_transition() -> None:
    stylesheet = tinycss2.parse_stylesheet(APP_CSS.read_text(encoding="utf-8"), skip_comments=True, skip_whitespace=True)
    reduced_motion_rules = [
        rule
        for rule in stylesheet
        if isinstance(rule, AtRule)
        and rule.lower_at_keyword == "media"
        and "prefers-reduced-motion: reduce" in tinycss2.serialize(rule.prelude)
    ]

    assert reduced_motion_rules
    universal_rules = [
        nested_rule
        for media_rule in reduced_motion_rules
        for nested_rule in tinycss2.parse_rule_list(media_rule.content, skip_comments=True, skip_whitespace=True)
        if isinstance(nested_rule, QualifiedRule) and tinycss2.serialize(nested_rule.prelude).strip() == "*"
    ]
    assert len(universal_rules) == 1

    declarations = {
        declaration.lower_name: (tinycss2.serialize(declaration.value).strip(), declaration.important)
        for declaration in tinycss2.parse_declaration_list(universal_rules[0].content, skip_comments=True, skip_whitespace=True)
        if isinstance(declaration, Declaration)
    }
    assert declarations["animation"] == ("none", True)
    assert declarations["transition"] == ("none", True)
