"""Settings dialog geometry contracts behind two iPhone reports.

1. The selected theme swatch's selection ring paints outside the button box
   (``outline`` + ``outline-offset``). The swatch grid sits full-width inside
   the scroll-bounded ``.formBody`` whose ``overflow-x: hidden`` clips at its
   padding box, so the grid must reserve at least the ring's outset as
   padding or the first/last card's outer edge is cut off.

2. iOS Safari zooms the page when a focused text entry computes under 16px
   and keeps that zoom after the keyboard closes. Every text-entry kind the
   app can render must resolve to the 16px floor on coarse pointers — the
   rule is matched against element descriptors, not against selector text.

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


def test_swatch_grid_reserves_the_selection_ring_outset() -> None:
    tokens = base_tokens()
    active = rule_declarations(".themeSwatch.active")
    outline = active["outline"].split()
    ring_outset = px(resolve(outline[0], tokens)) + px(resolve(active["outline-offset"], tokens))
    assert ring_outset > 0

    form_body = rule_declarations(".formBody")
    assert form_body["overflow-x"] == "hidden"  # the clip that made the padding necessary

    grid = rule_declarations(".themeSwatches")
    padding = [px(resolve(part, tokens)) for part in grid["padding"].split()]
    horizontal = (padding[1] if len(padding) > 1 else padding[0], padding[-1] if len(padding) == 4 else padding[1] if len(padding) > 1 else padding[0])
    assert min(horizontal) >= ring_outset


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


def coarse_pointer_font_rules() -> list[tuple[list[str], str, bool]]:
    """(selectors, font-size value, important) inside every coarse-pointer media block."""
    rules = []
    for node in parse_stylesheet(APP_CSS):
        if not isinstance(node, AtRule) or node.lower_at_keyword != "media":
            continue
        prelude = re.sub(r"\s+", " ", "".join(token.serialize() for token in node.prelude)).strip()
        if "(pointer: coarse)" not in prelude:
            continue
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
