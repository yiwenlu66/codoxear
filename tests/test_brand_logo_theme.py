"""Rendered brand-logo CSS contract across theme families.

The logo DOM is constructed by the shell test.  This test executes the actual
stylesheets through tinycss2 and resolves the semantic-token cascade each
family exposes to that DOM, rather than searching CSS source text.
"""

from __future__ import annotations

from css_tokens import all_rules, base_tokens, custom_properties, declarations, parse_stylesheet, resolve, selector_text
from css_tokens import APP_CSS, THEMES_DIR


LOGO_SELECTORS = {
    "page": ".brandLogo .brandLogoPage",
    "fold": ".brandLogo .brandLogoFold",
    "terminal": ".brandLogo .brandLogoTerminal",
}


def _base_logo_styles() -> dict[str, dict[str, str]]:
    styles: dict[str, dict[str, str]] = {}
    for rule in all_rules(parse_stylesheet(APP_CSS)):
        selector = selector_text(rule)
        for name, logo_selector in LOGO_SELECTORS.items():
            if selector == logo_selector:
                styles[name] = declarations(rule)
    assert styles.keys() == LOGO_SELECTORS.keys()
    return styles


def _theme_logo_styles(family: str, mode: str) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    tokens = dict(base_tokens())
    rules = parse_stylesheet(THEMES_DIR / f"{family}.css")
    tokens.update(custom_properties(rules, f':root[data-theme="{family}"]'))
    if mode == "dark":
        tokens.update(custom_properties(rules, f':root[data-theme="{family}"][data-mode="dark"]'))

    styles = _base_logo_styles()
    for rule in all_rules(rules):
        selector = selector_text(rule)
        if f'[data-theme="{family}"]' not in selector:
            continue
        if mode == "light" and '[data-mode="dark"]' in selector:
            continue
        for name, logo_selector in LOGO_SELECTORS.items():
            if selector.endswith(f" {logo_selector}"):
                styles[name].update(declarations(rule))
    return tokens, styles


def _resolved_brand_signature(family: str, mode: str) -> tuple[str, str, str, str]:
    tokens, styles = _theme_logo_styles(family, mode)
    return (
        resolve(styles["page"]["fill"], tokens),
        resolve(styles["page"]["stroke"], tokens),
        resolve(styles["fold"]["fill"], tokens),
        resolve(styles["terminal"]["stroke"], tokens),
    )


def test_brand_logo_is_token_rendered_and_family_styles_are_visibly_distinct() -> None:
    _tokens, styles = _theme_logo_styles("paper", "light")
    assert styles["page"] == {"fill": "var(--brand-logo-page)", "stroke": "var(--brand-logo-outline)"}
    assert styles["fold"] == {"fill": "var(--brand-logo-fold)", "stroke": "var(--brand-logo-outline)"}
    assert styles["terminal"] == {"stroke": "var(--brand-logo-terminal)"}

    signatures = {family: _resolved_brand_signature(family, "light") for family in ("paper", "clay", "slate")}
    assert len(set(signatures.values())) == 3
    assert signatures["paper"] == ("#ffffff", "#2f2b26", "#f6f5f1", "#2f2b26")
    assert signatures["clay"] == ("#fffdf9", "#c96442", "#f3e3da", "#35302a")
    assert signatures["slate"] == ("#ffffff", "#0d0d0d", "transparent", "#0d0d0d")


def test_paper_logo_resolves_to_ink_and_paper_roles_in_both_modes() -> None:
    for mode in ("light", "dark"):
        tokens, _styles = _theme_logo_styles("paper", mode)
        paper = resolve(tokens["--paper"], tokens)
        ink = resolve(tokens["--ink"], tokens)
        background = resolve(tokens["--bg"], tokens)
        assert _resolved_brand_signature("paper", mode) == (paper, ink, background, ink)


def test_brand_logo_tokens_remain_concrete_in_every_theme_mode() -> None:
    for family in ("paper", "clay", "slate"):
        for mode in ("light", "dark"):
            assert all("var(" not in value for value in _resolved_brand_signature(family, mode))
