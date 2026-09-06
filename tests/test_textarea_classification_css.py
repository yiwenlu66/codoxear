"""Every textarea in the app is classified: single-line composer or multi-line field.

The composer starts at control height and its auto-grow code raises the
inline height as lines accumulate. That single-line squash used to live on
the bare ``textarea`` element, so every dialog field had to escape it
individually and the ones that did not (the unattended prompt) rendered as
a one-line 44px box. The squash now lives only on ``.composer textarea``;
dialog textareas share one multi-line treatment under ``.formViewer``, and
the remaining textareas each declare their own sizing.

The stylesheet is parsed and cascaded per selector; nothing here inspects
raw source text.
"""

from __future__ import annotations

import re

from css_tokens import APP_CSS, base_tokens, declarations, parse_stylesheet, resolve, selector_text, top_level_rules

PX = re.compile(r"^(-?\d+(?:\.\d+)?)px$")

SINGLE_LINE_PROPERTIES = ("height", "min-height", "max-height", "resize", "overflow-y")

# Textareas outside .formViewer that own their sizing: queued-message editors,
# the paste-into-file field, the unattended per-session request, and the plain
# file editor fallback.
SELF_SIZED_TEXTAREAS = (".queueText", ".filePasteInput", ".unattendedMenu textarea", ".filePlainEditTextarea")


def px(value: str) -> float:
    match = PX.match(value.strip())
    assert match, f"expected a px length, got {value!r}"
    return float(match.group(1))


def cascaded(selector: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for rule in top_level_rules(parse_stylesheet(APP_CSS)):
        if selector in [part.strip() for part in selector_text(rule).split(",")]:
            result.update(declarations(rule))
    assert result, f"no rule for {selector}"
    return result


def test_the_bare_textarea_element_carries_no_single_line_squash() -> None:
    base = cascaded("textarea")
    leaked = [name for name in SINGLE_LINE_PROPERTIES if name in base]
    assert leaked == [], f"textarea element rule squashes every field app-wide via {leaked}"


def test_the_composer_owns_its_single_line_start_and_auto_grow_contract() -> None:
    composer = cascaded(".composer textarea")
    assert composer["height"] == composer["min-height"] == "var(--composerCtl)"
    assert px(composer["max-height"]) > 44  # the single-line control height
    assert composer["resize"] == "none"
    assert composer["overflow-y"] == "hidden"  # auto-grow flips it to auto only past max-height


def test_dialog_textareas_share_one_multi_line_treatment() -> None:
    tokens = base_tokens()
    field = cascaded(".formViewer textarea")
    assert "height" not in field, "the rows attribute sets the starting height"
    assert px(resolve(field["min-height"], tokens)) >= 2 * px(resolve("var(--ctl)", tokens))
    assert field["max-height"].startswith("min(")
    assert field["overflow-y"] == "auto"
    assert field["resize"] == "vertical"


def test_custom_css_field_adds_only_its_code_entry_bits() -> None:
    custom = cascaded(".customCssInput")
    assert set(custom) == {"font-family", "font-size", "white-space", "overflow"}
    assert custom["font-family"] == "var(--font-mono)"
    assert custom["white-space"] == "pre"


def test_every_other_textarea_declares_its_own_sizing() -> None:
    for selector in SELF_SIZED_TEXTAREAS:
        style = cascaded(selector)
        assert "resize" in style, f"{selector} inherits an unclassified resize behavior"
        assert "min-height" in style, f"{selector} has no starting height of its own"
        assert style.get("overflow-y") or style.get("overflow"), f"{selector} has no overflow behavior of its own"
