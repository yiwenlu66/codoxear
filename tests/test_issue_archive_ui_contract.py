"""Parsed stylesheet contracts for the issue-archive UI requirements."""

from __future__ import annotations

from pathlib import Path

from tinycss2 import parse_declaration_list, parse_rule_list, parse_stylesheet, serialize


APP_CSS = Path(__file__).resolve().parents[1] / "codoxear" / "static" / "app.css"


def _declarations(tokens: list[object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for declaration in parse_declaration_list(tokens, skip_comments=True, skip_whitespace=True):
        if declaration.type != "declaration":
            continue
        value = serialize(declaration.value).strip()
        result[declaration.lower_name] = f"{value} !important" if declaration.important else value
    return result


def _rules(rules: list[object], *, media: str | None = None) -> list[tuple[str, dict[str, str], str | None]]:
    result: list[tuple[str, dict[str, str], str | None]] = []
    for rule in rules:
        if rule.type == "at-rule" and rule.at_keyword == "media" and rule.content is not None:
            result.extend(_rules(parse_rule_list(rule.content, skip_comments=True, skip_whitespace=True), media=serialize(rule.prelude).strip()))
        elif rule.type == "qualified-rule":
            declarations = _declarations(rule.content)
            for selector in serialize(rule.prelude).split(","):
                result.append((selector.strip(), declarations, media))
    return result


def _computed(selector: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for candidate, declarations, media in _rules(parse_stylesheet(APP_CSS.read_text(encoding="utf-8"), skip_comments=True, skip_whitespace=True)):
        if candidate == selector and media is None:
            result.update(declarations)
    if not result:
        raise AssertionError(f"missing CSS rule for {selector}")
    return result


def test_archive_visual_contracts_preserve_focusable_layout_and_data_typography() -> None:
    topbar = _computed(".topbar")
    top_actions = _computed(".topActions")
    time_chip = _computed(".chatTimeChip")
    composer = _computed(".composer")
    chrome_buttons = _computed(".topActions .icon-btn")
    tab = _computed(".agentBackendTab.active")
    underline = _computed(".agentBackendTab.active::before")

    # Header content stays within its shell rather than making the page scroll;
    # each user-visible utility control retains an ink border.
    assert topbar["overflow"] == "hidden"
    assert top_actions["overflow"] == "hidden"
    assert chrome_buttons["border"] == "1px solid var(--border)"
    # The date chip has no shaded fill and the composer remains a one-hairline
    # input band, rather than an extra boxed textarea.
    assert time_chip["background"] == "transparent"
    assert composer["border-top"] == "1px solid var(--border)"
    # Backend selection is an ink outline plus a geometric underline, not a
    # filled black tab.
    assert tab["border"] == "2px solid var(--ink)"
    assert tab["background"] == "var(--paper)"
    assert underline == {
        "content": '""',
        "position": "absolute",
        "right": "var(--space-2)",
        "bottom": "var(--space-2)",
        "left": "var(--space-2)",
        "height": "2px",
        "background": "var(--ink)",
    }
    # State, cwd, and branch are ordinary compact prose; only the model and
    # effort unit carries monospace data typography.
    assert _computed(".sessionMetaLine")["font-family"] == "sans-serif"
    # sidebarMetaData carries model/effort; user explicitly requested proportional
    # (not monospace) for the entire sidebar secondary line.
    assert _computed(".sidebarMetaData").get("font-family", "sans-serif") != "var(--font-mono)"
    assert _computed(".sidebarMetaLabel")["font-family"] == "sans-serif"
    assert _computed(".sidebarMetaSeparator")["font-family"] == "sans-serif"


def test_archive_reduced_motion_contract_disables_attention_animations() -> None:
    reduced_rules = {
        selector: declarations
        for selector, declarations, media in _rules(
            parse_stylesheet(APP_CSS.read_text(encoding="utf-8"), skip_comments=True, skip_whitespace=True)
        )
        if media is not None and "prefers-reduced-motion" in media and "reduce" in media
    }
    assert reduced_rules == {
        ".stateDot.busy": {"animation": "none"},
        ".stateDot.pending": {"animation": "none"},
        ".msg-row.nav-pulse .msg": {"animation": "none"},
        ".typingDot": {"animation": "none"},
    }
