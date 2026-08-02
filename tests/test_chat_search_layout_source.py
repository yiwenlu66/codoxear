from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "codoxear/static/app.css").read_text(encoding="utf-8")
SHELL = (ROOT / "codoxear/static/app_shell.js").read_text(encoding="utf-8")
SEARCH = (ROOT / "codoxear/static/app_chat_search.js").read_text(encoding="utf-8")


def test_search_bar_is_one_compact_header_row() -> None:
    block = CSS.split(".chatSearchBar {", 1)[1].split("}", 1)[0]
    assert "height: 40px" in block
    assert "width: 100%" in block
    assert "border: 0" in block
    assert "border-bottom: 1px solid var(--hairline)" in block
    assert "margin: 0" in block


def test_search_input_owns_remaining_width_and_status_cannot_overlap() -> None:
    input_block = CSS.split(".chatSearchInput {", 1)[1].split("}", 1)[0]
    status_block = CSS.split(".chatSearchStatus {", 1)[1].split("}", 1)[0]
    assert "flex: 1 1 auto" in input_block
    assert "min-width: 0" in input_block
    assert "overflow: hidden" in status_block
    assert "text-overflow: ellipsis" in status_block
    assert "white-space: nowrap" in status_block
    icon_block = CSS.split(".chatSearchBar .icon-btn {", 1)[1].split("}", 1)[0]
    assert "flex: 0 0 32px" in icon_block
    assert "min-height: 32px" in icon_block
    assert 'inset: -6px' in CSS.split(".chatSearchBar .icon-btn::after {", 1)[1].split("}", 1)[0]


def test_search_bar_has_glyph_and_compact_mobile_status() -> None:
    assert "chatSearchGlyph" in SHELL
    assert "chatSearchGlyph, chatSearchInput, chatSearchStatus" in SHELL
    assert "`${position}/${total}`" in SEARCH
