import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "codoxear" / "static" / "app.css").read_text(encoding="utf-8")
SHELL = (ROOT / "codoxear" / "static" / "app_shell.js").read_text(encoding="utf-8")
ROWS = (ROOT / "codoxear" / "static" / "app_message_rows.js").read_text(encoding="utf-8")
APP = (ROOT / "codoxear" / "static" / "app.js").read_text(encoding="utf-8")


RULE_RE = re.compile(r"(?P<selectors>[^{}]+)\{(?P<body>[^{}]*)\}", re.MULTILINE)


def media_block(css: str, marker: str) -> str:
    start = css.index(marker)
    brace = css.index("{", start)
    depth = 1
    index = brace + 1
    while depth:
        if css[index] == "{":
            depth += 1
        elif css[index] == "}":
            depth -= 1
        index += 1
    return css[brace + 1 : index - 1]


def rule_body(css: str, selector: str, contains: str | None = None) -> str:
    for match in RULE_RE.finditer(css):
        if selector not in {part.strip() for part in match.group("selectors").split(",")}:
            continue
        body = match.group("body")
        if contains is None or contains in body:
            return body
    suffix = f" containing {contains!r}" if contains else ""
    raise AssertionError(f"No rule for {selector}{suffix}")


class TestChromeLayoutSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.touch = media_block(CSS, "@media (max-width: 700px), (pointer: coarse)")

    def test_message_copy_control_uses_the_shell_gutter(self) -> None:
        self.assertIn("shell.appendChild(copyBtn)", ROWS)
        self.assertNotIn('class: "msg-meta"', ROWS)
        shell = rule_body(CSS, ".msg-shell")
        self.assertIn("calc(100% - 36px)", shell)
        copy = rule_body(CSS, ".msg-copy-btn")
        self.assertIn("position: absolute", copy)
        self.assertIn("right: -36px", copy)
        self.assertIn("width: 30px", copy)
        self.assertIn("height: 30px", copy)
        self.assertIn("opacity: 0", copy)
        self.assertIn("pointer-events: none", copy)
        user_copy = rule_body(CSS, ".msg-shell.user .msg-copy-btn")
        self.assertIn("left: -36px", user_copy)
        self.assertIn("@media (hover: hover) and (pointer: fine) {\n        .msg-row:hover .msg-copy-btn", CSS)
        self.assertIn(".msg-row:focus-within .msg-copy-btn", CSS)
        self.assertIn(".msg-row.show-copy .msg-copy-btn", CSS)

    def test_gutter_copy_geometry_stays_inside_a_390px_viewport(self) -> None:
        viewport = 390
        chat_padding = 10
        button = 30
        offset = 36
        content_width = viewport - (2 * chat_padding)
        shell_width = min(760, content_width * 0.82, content_width - offset)
        assistant_left = chat_padding + shell_width + (offset - button)
        user_left = chat_padding + (content_width - shell_width) - offset
        self.assertGreaterEqual(assistant_left, chat_padding)
        self.assertLessEqual(assistant_left + button, viewport - chat_padding)
        self.assertGreaterEqual(user_left, chat_padding)
        self.assertLessEqual(user_left + button, viewport - chat_padding)
        self.assertEqual(offset - button, 6)

    def test_touch_secondary_controls_keep_visual_size_and_gain_hit_slop(self) -> None:
        copy = rule_body(self.touch, ".msg-copy-btn")
        self.assertIn("width: 30px", copy)
        self.assertIn("height: 30px", copy)
        self.assertNotIn("opacity: 1", copy)
        self.assertNotIn("44px", copy)
        copy_slop = rule_body(self.touch, ".msg-copy-btn::after")
        self.assertIn("position: absolute", copy_slop)
        self.assertIn("inset: -7px", copy_slop)
        jump = rule_body(self.touch, ".jumpBtn")
        self.assertIn("width: 32px", jump)
        self.assertIn("height: 32px", jump)
        self.assertNotIn("44px", jump)
        remove = rule_body(self.touch, ".stagedAttachmentRemove")
        self.assertIn("min-width: 18px", remove)
        self.assertIn("min-height: 18px", remove)
        remove_slop = rule_body(self.touch, ".stagedAttachmentRemove::after")
        self.assertIn("inset: -7px", remove_slop)

    def test_staged_tray_is_a_full_width_composer_row(self) -> None:
        self.assertIn('el("div", { class: "composerInputRow" }', SHELL)
        self.assertIn("stagedTray,\n      el(\"div\", { class: \"composerInputRow\" }", SHELL)
        tray = rule_body(CSS, ".stagedAttachments")
        self.assertIn("width: 100%", tray)
        self.assertIn("flex-wrap: wrap", tray)
        chip = rule_body(CSS, ".stagedAttachmentChip")
        self.assertIn("height: 32px", chip)
        clear = rule_body(CSS, ".stagedAttachmentsClear", "height: 32px")
        self.assertIn("height: 32px", clear)
        self.assertIn("margin-left: auto", clear)
        self.assertIn("middleEllipsis(name)", APP)
        form = rule_body(CSS, ".composer form")
        self.assertNotIn("border:", form)
        self.assertNotIn("background:", form)
        drop = rule_body(CSS, ".composer.drop-active form")
        self.assertIn("background: var(--wash)", drop)
        self.assertNotIn("outline:", drop)

    def test_chat_header_keeps_time_and_navigation_in_flow(self) -> None:
        self.assertIn('const chatHeader = el("div", { class: "chatHeader", id: "chatHeader" }, [chatTimeChip, chatNavRail]);', SHELL)
        header = rule_body(CSS, ".chatHeader")
        self.assertIn("display: grid", header)
        self.assertIn("grid-template-columns: minmax(0, 1fr) auto auto", header)
        self.assertIn("padding: 6px 12px 8px", header)
        self.assertIn("--chat-nav-rail-w: 120px", header)
        time = rule_body(CSS, ".chatTimeChip", "grid-column: 1 / 3")
        self.assertIn("grid-column: 1 / 3", time)
        self.assertIn("max-width: min(220px, calc(100vw - 24px - var(--chat-nav-rail-w)))", time)
        self.assertIn("overflow: hidden", time)
        self.assertIn("text-overflow: ellipsis", time)
        self.assertIn("text-align: center", time)
        rail = rule_body(CSS, ".chatNavRail")
        self.assertIn("grid-column: 3", rail)
        self.assertNotIn("align-self", rail)
        self.assertIn("isTouchCopyMode", APP)
        self.assertIn("window.getSelection().toString()", APP)
        self.assertIn("a, button, input, select, textarea, [role='link'], mark", APP)
        self.assertNotIn("display: none !important", self.touch)


if __name__ == "__main__":
    unittest.main()
