"""Theme file contract: every family palette overrides every semantic color.

A theme is ``:root[data-theme="<family>"]`` (light palette; paper light is the
base stylesheet itself) plus ``:root[data-theme="<family>"][data-mode="dark"]``
(dark palette). A token missing from a dark block would leak the paper-light
color into a dark surface, so coverage is asserted here rather than reviewed
by eye. Adding a color token to app.css ``:root`` obliges every family block.
"""

from __future__ import annotations

import re

import pytest

from css_tokens import APP_CSS, THEMES_DIR, all_rules, base_tokens, custom_properties, declarations, parse_stylesheet, resolve, selector_text

FAMILIES = ("paper", "clay", "slate")
# Colors that are the same in every theme by design.
THEME_INVARIANT_COLORS = {
    "--video-bg",  # video letterboxing is always black
}
COLOR_LITERAL = re.compile(r"^(#[0-9a-fA-F]{3,8}|rgba?\(.*\)|hsla?\(.*\)|transparent)$")


def semantic_color_tokens() -> set[str]:
    # Tokens whose base value is a color literal. Derived tokens such as
    # --text: var(--ink) follow their source through the cascade and need no
    # per-family override of their own.
    tokens = base_tokens()
    colors = {
        name
        for name, value in tokens.items()
        if COLOR_LITERAL.match(value) and name not in THEME_INVARIANT_COLORS
    }
    assert len(colors) > 60
    return colors


def family_blocks(family: str) -> tuple[dict[str, str], dict[str, str]]:
    rules = parse_stylesheet(THEMES_DIR / f"{family}.css")
    light = custom_properties(rules, f':root[data-theme="{family}"]')
    dark = custom_properties(rules, f':root[data-theme="{family}"][data-mode="dark"]')
    return light, dark


@pytest.mark.parametrize("family", FAMILIES)
def test_dark_block_overrides_every_semantic_color(family: str) -> None:
    _light, dark = family_blocks(family)
    missing = semantic_color_tokens() - set(dark)
    assert not missing, f"{family} dark block leaves paper-light colors for: {sorted(missing)}"


@pytest.mark.parametrize("family", ("clay", "slate"))
def test_light_block_overrides_every_semantic_color(family: str) -> None:
    light, _dark = family_blocks(family)
    missing = semantic_color_tokens() - set(light)
    assert not missing, f"{family} light block leaves paper-light colors for: {sorted(missing)}"


def test_paper_file_holds_only_the_dark_block() -> None:
    rules = parse_stylesheet(THEMES_DIR / "paper.css")
    assert [selector_text(rule) for rule in all_rules(rules)] == [':root[data-theme="paper"][data-mode="dark"]']


@pytest.mark.parametrize("family", FAMILIES)
def test_dark_blocks_flip_the_icon_filter_and_color_scheme(family: str) -> None:
    _light, dark = family_blocks(family)
    assert "invert(" in dark["--icon-muted-filter"], f"{family}: dark icons must not stay black-on-black"
    rules = parse_stylesheet(THEMES_DIR / f"{family}.css")
    dark_rule = next(rule for rule in all_rules(rules) if selector_text(rule) == f':root[data-theme="{family}"][data-mode="dark"]')
    assert declarations(dark_rule).get("color-scheme") == "dark"


@pytest.mark.parametrize("family", FAMILIES)
def test_family_tokens_resolve_to_concrete_values(family: str) -> None:
    # A family block may reference its own tokens (--text: var(--ink)) but the
    # cascade base + family must resolve every color without dangling var().
    light, dark = family_blocks(family)
    for label, block in (("light", light), ("dark", dark)):
        table = dict(base_tokens())
        table.update(light)
        if label == "dark":
            table.update(dark)
        for name in semantic_color_tokens():
            value = resolve(table[name], table)
            assert "var(" not in value, f"{family} {label} {name} -> {value}"


def test_family_component_rules_keep_base_specificity() -> None:
    # Component rules use :where(:root[data-theme=...]) so they only outrank
    # the exact base rule they replace by source order; a bare :root[...]
    # prefix would silently beat more specific base variants (error bubbles,
    # active badges, disabled primaries).
    for family in ("clay", "slate"):
        rules = parse_stylesheet(THEMES_DIR / f"{family}.css")
        for rule in all_rules(rules):
            selector = selector_text(rule)
            if selector.startswith(":root["):
                assert set(declarations(rule)) <= {"color-scheme"} | {name for name in declarations(rule) if name.startswith("--")}, selector
                continue
            for part in selector.split(","):
                assert part.strip().startswith(f':where(:root[data-theme="{family}"]) '), f"{family}: {part.strip()}"


def test_family_stylesheets_retune_geometry_through_tokens_only() -> None:
    # A family may set literal radii only for surfaces a role token cannot
    # express (documented in the file); everything else must go through tokens.
    allowed_literal_radius = {
        "clay": {".ownDot"},
        "slate": {".composer form"},
    }
    for family in ("clay", "slate"):
        rules = parse_stylesheet(THEMES_DIR / f"{family}.css")
        light = custom_properties(rules, f':root[data-theme="{family}"]')
        for name in ("--radius-control", "--radius-card", "--radius-bubble", "--radius-pill", "--radius-dot"):
            assert name in light, f"{family} does not retune {name}"
        literal_selectors = set()
        for rule in all_rules(rules):
            radius = declarations(rule).get("border-radius")
            if radius and "var(" not in radius:
                literal_selectors.add(selector_text(rule).replace(f':where(:root[data-theme="{family}"]) ', ""))
        assert literal_selectors == allowed_literal_radius[family]
