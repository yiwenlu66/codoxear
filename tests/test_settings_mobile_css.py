"""Form-dialog type contracts on coarse pointers and small viewports.

1. iOS Safari zooms the page when a focused text entry computes under 16px
   and keeps that zoom after the keyboard closes. Every text-entry kind the
   app can render must resolve to the 16px floor on coarse pointers — the
   rule is matched against element descriptors, not against selector text.

2. That floor would invert a form dialog's hierarchy (entry 16 > title 14 >
   label 13), so form-dialog type planes retune as a unit under the same
   media condition: the title plane meets the floor, labels sit one step
   below, hints one step below that. Desktop planes stay untouched.

(The selected swatch's ring clipping is covered by the dialog scroll-body
gutter in test_dialog_scroll_gutter_css.py.)

Per project policy this parses the stylesheet and resolves tokens the way a
browser would instead of asserting on raw source.
"""

from __future__ import annotations

import re

import pytest
import tinycss2
from tinycss2.ast import AtRule, Declaration

from css_tokens import APP_CSS, all_rules, base_tokens, declarations, parse_stylesheet, resolve, selector_text

PX = re.compile(r"^(-?\d+(?:\.\d+)?)px$")


def px(value: str) -> float:
    match = PX.match(value.strip())
    assert match, f"expected a px length, got {value!r}"
    return float(match.group(1))


def rule_declarations(selector: str) -> dict[str, str]:
    for rule in all_rules(parse_stylesheet(APP_CSS)):
        if selector_text(rule) == selector:
            return declarations(rule)
    raise AssertionError(f"no rule for {selector}")


# (tag, type attribute) for every text-entry kind that can gain focus and
# raise the iOS keyboard; ``None`` is an <input> without a type attribute.
TEXT_ENTRIES = [
    ("input", None),
    ("input", "text"),
    ("input", "password"),
    ("input", "email"),
    ("input", "url"),
    ("input", "tel"),
    ("input", "search"),
    ("input", "number"),
    ("input", "date"),
    ("input", "time"),
    ("input", "datetime-local"),
    ("select", None),
    ("textarea", None),
]

SIMPLE_SELECTOR = re.compile(r'^(?P<tag>[a-z]+)(?::not\(\[type\]\)|\[type="(?P<type>[a-z-]+)"\])?$')


def selector_matches(selector: str, tag: str, type_attr: str | None) -> bool:
    match = SIMPLE_SELECTOR.match(selector.strip())
    if not match or match.group("tag") != tag:
        return False
    if ":not([type])" in selector:
        return type_attr is None
    if match.group("type") is not None:
        return match.group("type") == type_attr
    return True


def coarse_pointer_blocks() -> list[AtRule]:
    blocks = []
    for node in parse_stylesheet(APP_CSS):
        if not isinstance(node, AtRule) or node.lower_at_keyword != "media":
            continue
        prelude = re.sub(r"\s+", " ", "".join(token.serialize() for token in node.prelude)).strip()
        if "(pointer: coarse)" in prelude:
            blocks.append(node)
    assert blocks, "no coarse-pointer media block"
    return blocks


def coarse_pointer_font_rules() -> list[tuple[list[str], str, bool]]:
    """(selectors, font-size value, important) inside every coarse-pointer media block."""
    rules = []
    for node in coarse_pointer_blocks():
        for rule in all_rules([node]):
            for decl in tinycss2.parse_declaration_list(rule.content, skip_comments=True, skip_whitespace=True):
                if isinstance(decl, Declaration) and decl.lower_name == "font-size":
                    selectors = [s.strip() for s in selector_text(rule).split(",")]
                    rules.append((selectors, tinycss2.serialize(decl.value).strip(), bool(decl.important)))
    assert rules, "no coarse-pointer font-size rule"
    return rules


@pytest.mark.parametrize("tag,type_attr", TEXT_ENTRIES, ids=[f"{t}[{ty}]" for t, ty in TEXT_ENTRIES])
def test_every_text_entry_kind_resolves_to_the_ios_no_zoom_floor(tag: str, type_attr: str | None) -> None:
    tokens = base_tokens()
    sizes = [
        (value, important)
        for selectors, value, important in coarse_pointer_font_rules()
        if any(selector_matches(selector, tag, type_attr) for selector in selectors)
    ]
    assert sizes, f"no coarse-pointer font-size rule matches <{tag} type={type_attr}>"
    for value, important in sizes:
        assert important, "the floor must beat component font-size rules"
        assert px(resolve(value, tokens)) >= 16


def coarse_pointer_form_dialog_tokens() -> dict[str, str]:
    """Custom properties the form-dialog scope retunes on coarse pointers."""
    tokens: dict[str, str] = {}
    for node in coarse_pointer_blocks():
        for rule in all_rules([node]):
            if selector_text(rule) == ".formViewer":
                tokens.update({name: value for name, value in declarations(rule).items() if name.startswith("--")})
    assert tokens, "form dialogs do not retune their type planes on coarse pointers"
    return tokens


def test_form_dialog_type_planes_retune_as_a_unit_under_the_entry_floor() -> None:
    desktop = base_tokens()
    mobile = {**desktop, **coarse_pointer_form_dialog_tokens()}
    floor = px(resolve("var(--font-xl)", mobile))
    title = px(resolve("var(--font-lg)", mobile))
    label = px(resolve("var(--font-md)", mobile))
    hint = px(resolve("var(--font-sm)", mobile))
    assert floor >= 16
    assert title == floor, "the section-title plane must meet the entry floor, never sit below it"
    assert floor > label > hint, "labels and hints step down from the title plane"
    # Desktop planes are untouched: the retune lives only in the coarse-pointer scope.
    assert [px(resolve(f"var(--font-{name})", desktop)) for name in ("sm", "md", "lg", "xl")] == [12, 13, 14, 16]
    assert set(coarse_pointer_form_dialog_tokens()) == {"--font-sm", "--font-md", "--font-lg"}
