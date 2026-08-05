"""Behavioral stylesheet coverage for the global reduced-motion contract."""

from pathlib import Path

import tinycss2
from tinycss2.ast import AtRule, Declaration, QualifiedRule


APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


def test_reduced_motion_disables_attention_animations() -> None:
    stylesheet = tinycss2.parse_stylesheet(APP_CSS.read_text(encoding="utf-8"), skip_comments=True, skip_whitespace=True)
    reduced_motion_rules = [
        rule
        for rule in stylesheet
        if isinstance(rule, AtRule)
        and rule.lower_at_keyword == "media"
        and "prefers-reduced-motion: reduce" in tinycss2.serialize(rule.prelude)
    ]

    assert len(reduced_motion_rules) == 1
    reduced_rules = {
        tinycss2.serialize(nested_rule.prelude).strip(): {
            declaration.lower_name: (tinycss2.serialize(declaration.value).strip(), declaration.important)
            for declaration in tinycss2.parse_declaration_list(nested_rule.content, skip_comments=True, skip_whitespace=True)
            if isinstance(declaration, Declaration)
        }
        for nested_rule in tinycss2.parse_rule_list(
            reduced_motion_rules[0].content, skip_comments=True, skip_whitespace=True
        )
        if isinstance(nested_rule, QualifiedRule)
    }
    assert reduced_rules == {
        ".stateDot.busy,\n        .stateDot.pending": {"animation": ("none", False)},
        ".msg-row.nav-pulse .msg": {"animation": ("none", False)},
        ".typingDot": {"animation": ("none", False)},
    }
