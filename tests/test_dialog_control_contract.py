"""Dialog control contract: icon-only chrome is square in every modal surface.

Dialog icon buttons (header close buttons, the backend logo tab) once
collapsed to glyph width because a content-sizing rule written for
text-bearing dialog controls also hit icon-only buttons. Each dialog then
carried its own rescue rule — an ID override for New session, a surface block
for Details — and dialogs added later without a rescue (Settings, Edit
conversation) rendered their close button as a glyph-hugging vertical pill.
This test pins the contract where it belongs: one square-chrome rule covering
all four modal surfaces, content-sizing only for text-bearing controls, one
borderless-header rule, and no per-dialog rescue selectors.
"""

from __future__ import annotations

from css_tokens import APP_CSS, all_rules, base_tokens, declarations, parse_stylesheet, resolve, selector_text

SURFACES = (".formViewer", ".queueViewer", ".diagViewer", ".helpViewer")
SQUARE_SELECTOR = (
    ".formViewer .icon-btn, .queueViewer .icon-btn, .diagViewer .icon-btn, "
    ".helpViewer .icon-btn, .formViewer .agentBackendTab"
)
CLOSE_BUTTON_IDS = (
    "#newSessionCloseBtn",
    "#settingsCloseBtn",
    "#editCloseBtn",
    "#diagCloseBtn",
    "#queueCloseBtn",
    "#helpCloseBtn",
)


def app_rules():
    return all_rules(parse_stylesheet(APP_CSS))


def rule_declarations(rules, selector):
    matches = [rule for rule in rules if selector_text(rule) == selector]
    assert len(matches) == 1, f"expected exactly one rule for {selector!r}, found {len(matches)}"
    return declarations(matches[0])


def test_dialog_icon_chrome_is_square_in_every_surface() -> None:
    tokens = base_tokens()
    decls = rule_declarations(app_rules(), SQUARE_SELECTOR)
    for prop in ("width", "min-width", "height", "min-height"):
        assert decls[prop] == "var(--dialog-control-h)", f"{prop}: {decls[prop]}"
        assert resolve(decls[prop], tokens) == "32px"


def test_text_bearing_dialog_controls_size_to_content() -> None:
    decls = rule_declarations(app_rules(), ".formViewer .choiceChip")
    assert decls["width"] == "auto"
    assert decls["min-width"] == "0"


def test_dialog_header_chrome_is_borderless_on_every_surface() -> None:
    decls = rule_declarations(app_rules(), ".queueHeader .icon-btn")
    assert decls["border"] == "0"


def test_no_per_dialog_close_button_rescue_selectors() -> None:
    selectors = [selector_text(rule) for rule in app_rules()]
    rescues = [sel for sel in selectors if any(fragment in sel for fragment in CLOSE_BUTTON_IDS)]
    assert not rescues, f"close-button rescue selectors reappeared: {rescues}"


def test_no_later_rule_content_sizes_dialog_icon_buttons() -> None:
    rules = app_rules()
    for rule in rules:
        sel = selector_text(rule)
        if sel == SQUARE_SELECTOR or "::" in sel or "text-btn" in sel:
            continue
        if any(surface in sel for surface in SURFACES) and ".icon-btn" in sel:
            decls = declarations(rule)
            width = decls.get("width")
            assert width != "auto", f"{sel} re-breaks the square contract (width: auto)"
