"""Parse the shared Markdown styles: visual wraps must not collapse code whitespace.

Real line geometry, byte-exact clipboard round trips and responsive/theme cascade
coverage live in scripts/docker_markdown_codeblock_wrap_verify.py.
"""
from css_tokens import APP_CSS, declarations, parse_stylesheet, selector_text, top_level_rules


def cascaded(selector: str) -> dict[str, str]:
    result = {}
    for rule in top_level_rules(parse_stylesheet(APP_CSS)):
        if selector in [part.strip() for part in selector_text(rule).split(",")]:
            result.update(declarations(rule))
    assert result, f"no rule for {selector}"
    return result


def test_markdown_blocks_preserve_whitespace_and_wrap_unbroken_tokens() -> None:
    pre = cascaded(".md pre")
    assert pre["white-space"] == "pre-wrap"
    assert pre["overflow-wrap"] == "anywhere"
    assert pre["max-width"] == "100%"
    assert pre["box-sizing"] == "border-box"
    assert cascaded(".md pre code")["white-space"] == "inherit"


def test_inline_code_keeps_its_independent_whitespace_rule() -> None:
    inline = cascaded(".md code")
    assert inline["white-space"] == "pre-wrap"
    assert inline["word-break"] == "break-word"


def test_source_editor_keeps_preformatted_horizontal_scrolling() -> None:
    source = cascaded(".filePlainEditTextarea")
    assert source["white-space"] == "pre"
    assert source["overflow"] == "auto"
