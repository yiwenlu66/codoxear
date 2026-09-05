"""Shared CSS token helpers for cascade tests.

The stylesheet expresses appearance through ``:root`` custom properties. Tests
that assert computed geometry or color must resolve ``var()`` references the
way a browser would (recursively, honoring fallbacks) instead of comparing
raw declaration text.
"""

from __future__ import annotations

import re
from pathlib import Path

import tinycss2
from tinycss2.ast import AtRule, Declaration, QualifiedRule

STATIC_DIR = Path(__file__).resolve().parents[1] / "codoxear" / "static"
APP_CSS = STATIC_DIR / "app.css"
THEMES_DIR = STATIC_DIR / "themes"

_VAR_RE = re.compile(r"var\(\s*(--[\w-]+)\s*(?:,([^()]*))?\)")


def parse_stylesheet(path: Path) -> list[object]:
    return tinycss2.parse_stylesheet(path.read_text(encoding="utf-8"), skip_comments=True, skip_whitespace=True)


def declarations(rule: QualifiedRule) -> dict[str, str]:
    result: dict[str, str] = {}
    for declaration in tinycss2.parse_declaration_list(rule.content, skip_comments=True, skip_whitespace=True):
        if isinstance(declaration, Declaration):
            result[declaration.lower_name] = tinycss2.serialize(declaration.value).strip()
    return result


def selector_text(rule: QualifiedRule) -> str:
    return re.sub(r"\s+", " ", tinycss2.serialize(rule.prelude).strip())


def top_level_rules(rules: list[object]) -> list[QualifiedRule]:
    """Qualified rules outside any at-rule, in source order."""
    return [rule for rule in rules if isinstance(rule, QualifiedRule)]


def all_rules(rules: list[object]) -> list[QualifiedRule]:
    """Every qualified rule including those nested in @media/@supports."""
    flattened: list[QualifiedRule] = []
    for rule in rules:
        if isinstance(rule, QualifiedRule):
            flattened.append(rule)
        elif isinstance(rule, AtRule) and rule.content is not None:
            flattened.extend(all_rules(tinycss2.parse_rule_list(rule.content, skip_comments=True, skip_whitespace=True)))
    return flattened


def custom_properties(rules: list[object], selector: str) -> dict[str, str]:
    """Custom properties declared by top-level rules whose selector matches exactly."""
    tokens: dict[str, str] = {}
    for rule in top_level_rules(rules):
        if selector_text(rule) != selector:
            continue
        for name, value in declarations(rule).items():
            if name.startswith("--"):
                tokens[name] = value
    return tokens


def resolve(value: str, tokens: dict[str, str], depth: int = 0) -> str:
    """Resolve ``var()`` references recursively against a token table."""
    if depth > 32:
        raise AssertionError(f"custom property cycle while resolving {value!r}")

    def substitute(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2)
        if name in tokens:
            return resolve(tokens[name], tokens, depth + 1)
        if fallback is not None:
            return resolve(fallback.strip(), tokens, depth + 1)
        return match.group(0)

    return _VAR_RE.sub(substitute, value)


def base_tokens() -> dict[str, str]:
    """The paper-light token table: app.css ``:root`` outside media queries."""
    return custom_properties(parse_stylesheet(APP_CSS), ":root")
